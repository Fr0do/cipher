"""
extract_key.py — key extraction from edited video for video-captcha-solver.

Tries in order: QR/barcode → OCR → VLM → audio morse/tones → LSB steg.
Returns the first non-empty key found, or None.
"""
import io
import subprocess
import tempfile
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------

def sample_frames(video_path: str, fps: float = 1.0) -> list:
    """Extract frames as PIL Images at given fps."""
    from PIL import Image
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps},scale=1280:-2",
        "-f", "image2pipe", "-pix_fmt", "rgb24", "-vcodec", "rawvideo", "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not result.stdout:
        return []

    # probe for dimensions
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", video_path],
        capture_output=True, text=True
    )
    try:
        w, h = map(int, probe.stdout.strip().split(","))
    except Exception:
        return []

    raw = result.stdout
    frame_size = w * h * 3
    frames = []
    for i in range(0, len(raw), frame_size):
        chunk = raw[i:i + frame_size]
        if len(chunk) == frame_size:
            arr = np.frombuffer(chunk, dtype=np.uint8).reshape((h, w, 3))
            frames.append(Image.fromarray(arr))
    return frames


def sample_scene_frames(video_path: str, threshold: float = 0.3) -> list:
    """Extract frames at scene changes."""
    from PIL import Image
    with tempfile.TemporaryDirectory() as td:
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", f"select='gt(scene,{threshold})',scale=1280:-2",
            "-vsync", "vfr",
            f"{td}/%04d.jpg"
        ]
        subprocess.run(cmd, capture_output=True)
        frames = []
        for p in sorted(Path(td).glob("*.jpg")):
            from PIL import Image
            frames.append(Image.open(p).copy())
    return frames


# ---------------------------------------------------------------------------
# 1. QR / barcode
# ---------------------------------------------------------------------------

def try_qr(frames: list) -> str | None:
    try:
        from pyzbar.pyzbar import decode
    except ImportError:
        return None
    for frame in frames:
        results = decode(frame)
        for r in results:
            data = r.data.decode("utf-8", errors="ignore").strip()
            if data:
                return data
    return None


# ---------------------------------------------------------------------------
# 2. OCR
# ---------------------------------------------------------------------------

def try_ocr(frames: list) -> str | None:
    try:
        import easyocr
        reader = easyocr.Reader(["en"], verbose=False)
        use_easy = True
    except ImportError:
        use_easy = False

    candidates = []
    for frame in frames:
        if use_easy:
            results = reader.readtext(np.array(frame))
            for _, text, conf in results:
                text = text.strip()
                if conf > 0.5 and 4 <= len(text) <= 32:
                    candidates.append((conf, text))
        else:
            try:
                import pytesseract
                text = pytesseract.image_to_string(frame).strip()
                for line in text.splitlines():
                    line = line.strip()
                    if 4 <= len(line) <= 32 and any(c.isalnum() for c in line):
                        candidates.append((0.5, line))
            except Exception:
                pass

    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]
    return None


# ---------------------------------------------------------------------------
# 3. VLM
# ---------------------------------------------------------------------------

def try_vlm(frames: list, vlm_fn) -> str | None:
    if not vlm_fn or not frames:
        return None

    # pick up to 8 evenly spaced frames
    step = max(1, len(frames) // 8)
    selected = frames[::step][:8]

    prompt = """These are frames from a video that has been edited according to specific instructions.
A key or cipher is hidden in or revealed by this video. Look carefully for:
- Text strings (especially short alphanumeric codes, 4-16 characters)
- QR codes or barcodes
- Morse code patterns (visual dots and dashes)
- Number sequences or hidden symbols
- Any non-natural visual element that looks like an embedded message

What key, code, or cipher do you see? Return ONLY the key string, nothing else.
If you see nothing, return NONE."""

    return vlm_fn(prompt, images=selected)


# ---------------------------------------------------------------------------
# 4. Audio morse / tones
# ---------------------------------------------------------------------------

def try_audio(video_path: str) -> str | None:
    try:
        from scipy.io import wavfile
        from scipy.signal import find_peaks
        import scipy.fft
    except ImportError:
        return None

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name

    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-ar", "8000", "-ac", "1", wav_path],
        capture_output=True
    )
    try:
        rate, data = wavfile.read(wav_path)
    except Exception:
        return None

    if data.dtype != np.float32:
        data = data.astype(np.float32) / np.iinfo(data.dtype).max

    # energy envelope: detect on/off bursts → morse
    window = rate // 20  # 50ms
    energy = np.array([np.mean(data[i:i+window]**2) for i in range(0, len(data), window)])
    threshold = np.max(energy) * 0.1
    bits = (energy > threshold).astype(int)

    # very naive morse decoder: count run lengths
    runs = []
    current = bits[0]
    count = 0
    for b in bits:
        if b == current:
            count += 1
        else:
            runs.append((current, count))
            current = b
            count = 1
    runs.append((current, count))

    dot_len = min((c for v, c in runs if v == 1), default=1)
    morse = ""
    for val, cnt in runs:
        if val == 1:
            morse += "." if cnt <= dot_len * 1.5 else "-"
        else:
            morse += " " if cnt <= dot_len * 2 else "/"

    return _decode_morse(morse.strip()) or None


_MORSE_CODE = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E",
    "..-.": "F", "--.": "G", "....": "H", "..": "I", ".---": "J",
    "-.-": "K", ".-..": "L", "--": "M", "-.": "N", "---": "O",
    ".--.": "P", "--.-": "Q", ".-.": "R", "...": "S", "-": "T",
    "..-": "U", "...-": "V", ".--": "W", "-..-": "X", "-.--": "Y",
    "--..": "Z", "-----": "0", ".----": "1", "..---": "2",
    "...--": "3", "....-": "4", ".....": "5", "-....": "6",
    "--...": "7", "---..": "8", "----.": "9",
}


def _decode_morse(morse: str) -> str:
    words = morse.split("/")
    decoded = []
    for word in words:
        chars = []
        for code in word.strip().split():
            chars.append(_MORSE_CODE.get(code, "?"))
        decoded.append("".join(chars))
    result = " ".join(decoded)
    return result if "?" not in result and result.strip() else None


# ---------------------------------------------------------------------------
# 5. LSB steganography
# ---------------------------------------------------------------------------

def try_lsb(frames: list) -> str | None:
    if not frames:
        return None
    try:
        arr = np.array(frames[0])
        flat = arr.flatten()
        bits = flat & 1
        # try to decode as ASCII bytes
        chars = []
        for i in range(0, min(len(bits), 1024), 8):
            byte = int("".join(str(b) for b in bits[i:i+8]), 2)
            if 32 <= byte <= 126:
                chars.append(chr(byte))
            elif byte == 0:
                break
            else:
                return None
        text = "".join(chars).strip()
        return text if len(text) >= 4 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------

def extract_key(video_path: str, vlm_fn=None) -> str | None:
    frames_1fps = sample_frames(video_path, fps=1.0)
    frames_scene = sample_scene_frames(video_path)
    all_frames = frames_1fps + frames_scene

    for method, fn in [
        ("QR",    lambda: try_qr(all_frames)),
        ("OCR",   lambda: try_ocr(all_frames)),
        ("VLM",   lambda: try_vlm(all_frames, vlm_fn)),
        ("AUDIO", lambda: try_audio(video_path)),
        ("LSB",   lambda: try_lsb(frames_1fps[:1])),
    ]:
        result = fn()
        if result and result.strip().upper() != "NONE":
            print(f"[extract_key] found via {method}: {result!r}")
            return result.strip()

    return None
