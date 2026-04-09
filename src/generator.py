"""
generator.py — procedural captcha benchmark generator for CIPHER.

Each sample = (scrambled_video, nl_instructions, key, metadata)
Fully deterministic from (seed, difficulty_config, background_track).

Difficulty axes:
  key_visibility : 1-4  (large plain → small → low-contrast → brief flash)
  op_count       : 1-4  (number of edit operations chained)
  nl_ambiguity   : 1-4  (exact → paraphrase → reversed description → compositional)

Background tracks (independent of difficulty, crossed with L1-L5):
  clean   — animation / studio footage, no on-screen text
             sources: Blender open movies (BBB, Sintel, Elephants Dream)
  natural — real-world human activity, incidental text only
             sources: Kinetics-400 / ActivityNet clips
  text    — high on-screen text density (slides, news tickers, screencasts)
             sources: TED talks, Khan Academy, newscast recordings

Paper structure: 3 tracks × 5 levels × N samples = full CIPHER dataset.
Per-track results reveal whether background text confounds key extraction (S3)
independently of instruction-following (S1) and execution (S2).
"""
import random
import string
import subprocess
import tempfile
import json
import hashlib
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Difficulty presets (used in the paper)
# ---------------------------------------------------------------------------

LEVELS = {
    "L1": dict(key_visibility=1, op_count=1, nl_ambiguity=1, distractor=0),
    "L2": dict(key_visibility=2, op_count=2, nl_ambiguity=1, distractor=1),
    "L3": dict(key_visibility=3, op_count=2, nl_ambiguity=2, distractor=1),
    "L4": dict(key_visibility=3, op_count=3, nl_ambiguity=3, distractor=2),
    "L5": dict(key_visibility=4, op_count=4, nl_ambiguity=4, distractor=2),
    # L6 (QR/morse): reserved for future work
}


# ---------------------------------------------------------------------------
# Op library — each op has: ffmpeg_filter, inverse_filter, nl_phrases[]
# ---------------------------------------------------------------------------

OPS = [
    {
        "name": "reverse",
        "vf": "reverse", "af": "areverse",
        "inverse_vf": "reverse", "inverse_af": "areverse",
        "nl": [
            "reverse the video",
            "play it backwards",
            "play the video in reverse",
            "flip the timeline",
            "the video is played in reverse order",
        ],
    },
    {
        "name": "hflip",
        "vf": "hflip", "af": None,
        "inverse_vf": "hflip", "inverse_af": None,
        "nl": [
            "flip the video horizontally",
            "mirror the video left to right",
            "apply a horizontal mirror",
            "flip it left-right",
            "the video has been horizontally mirrored",
        ],
    },
    {
        "name": "vflip",
        "vf": "vflip", "af": None,
        "inverse_vf": "vflip", "inverse_af": None,
        "nl": [
            "flip the video vertically",
            "flip it upside down",
            "apply a vertical flip",
            "the video is upside down, correct it",
            "flip the video top to bottom",
        ],
    },
    {
        "name": "rotate90",
        "vf": "transpose=1", "af": None,
        "inverse_vf": "transpose=2", "inverse_af": None,
        "nl": [
            "rotate the video 90 degrees clockwise",
            "rotate it 90° to the right",
            "turn the video clockwise by 90 degrees",
            "the video was rotated 90° counter-clockwise, fix it",
            "rotate 90 degrees clockwise",
        ],
    },
    {
        "name": "rotate270",
        "vf": "transpose=2", "af": None,
        "inverse_vf": "transpose=1", "inverse_af": None,
        "nl": [
            "rotate the video 270 degrees clockwise",
            "rotate it 90° to the left",
            "turn the video counter-clockwise by 90 degrees",
            "the video was rotated 90° clockwise, undo it",
            "rotate 270 degrees clockwise",
        ],
    },
    {
        "name": "grayscale",
        "vf": "format=gray,format=yuv420p", "af": None,
        "inverse_vf": None, "inverse_af": None,  # irreversible — only used in distractor mode
        "nl": [
            "convert the video to grayscale",
            "desaturate the video",
            "make it black and white",
            "remove all color from the video",
            "apply grayscale filter",
        ],
    },
    {
        "name": "speed2x",
        "vf": "setpts=0.5*PTS", "af": "atempo=2.0",
        "inverse_vf": "setpts=2.0*PTS", "inverse_af": "atempo=0.5",
        "nl": [
            "speed up the video 2x",
            "double the playback speed",
            "make the video play twice as fast",
            "the video is at half speed, fix it",
            "apply 2x speed",
        ],
    },
]

# Ops safe for chaining (reversible, no irreversible transforms)
CHAIN_OPS = [op for op in OPS if op["name"] != "grayscale"]


# ---------------------------------------------------------------------------
# NL instruction builder
# ---------------------------------------------------------------------------

def build_instructions(ops: list[dict], nl_ambiguity: int, rng: random.Random) -> str:
    """
    nl_ambiguity:
      1 = exact canonical phrase, sequential "first … then …"
      2 = random paraphrase, sequential
      3 = reversed description order ("the video has been X then Y, undo")
      4 = compositional / underspecified ("undo all edits that were applied")
    """
    if nl_ambiguity == 4:
        return "The video has been edited. Undo all the transformations to reveal the original content."

    phrases = []
    for op in ops:
        idx = 0 if nl_ambiguity == 1 else rng.randint(1, len(op["nl"]) - 1)
        phrases.append(op["nl"][idx])

    if nl_ambiguity == 3:
        phrases = list(reversed(phrases))
        if len(phrases) == 1:
            return f"The video was modified: {phrases[0]}. Undo this."
        steps = "; then ".join(phrases)
        return f"The following edits were applied in order: {steps}. Apply the inverse operations to restore it."

    if len(phrases) == 1:
        return phrases[0].capitalize() + "."
    connector = rng.choice(["First", "Step 1:"])
    parts = [f"{connector} {phrases[0]}"]
    for i, p in enumerate(phrases[1:], 2):
        parts.append(f"then {p}")
    return ", ".join(parts) + "."


# ---------------------------------------------------------------------------
# Key embedding
# ---------------------------------------------------------------------------

def _random_key(rng: random.Random, length: int = 7) -> str:
    chars = string.ascii_uppercase + string.digits
    # avoid ambiguous chars
    chars = chars.replace("0", "").replace("O", "").replace("I", "").replace("1", "")
    return "".join(rng.choices(chars, k=length))


def embed_key_overlay(
    input_video: str,
    output_video: str,
    key: str,
    key_visibility: int,
    start_t: float,
    end_t: float,
    rng: random.Random,
) -> None:
    """Burn key text onto video frames during [start_t, end_t]."""
    # Probe video dimensions
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", input_video],
        capture_output=True, text=True
    )
    try:
        w, h = map(int, probe.stdout.strip().split(","))
    except Exception:
        w, h = 640, 360

    # Key rendering params by visibility level
    vis_params = {
        1: dict(fontsize=int(h * 0.12), color=(255, 255, 0), alpha=255, x_frac=0.5, y_frac=0.5),
        2: dict(fontsize=int(h * 0.08), color=(255, 255, 255), alpha=220, x_frac=rng.uniform(0.3, 0.7), y_frac=rng.uniform(0.3, 0.7)),
        3: dict(fontsize=int(h * 0.06), color=(200, 200, 200), alpha=180, x_frac=rng.uniform(0.2, 0.8), y_frac=rng.uniform(0.2, 0.8)),
        4: dict(fontsize=int(h * 0.05), color=(180, 180, 180), alpha=140, x_frac=rng.uniform(0.1, 0.9), y_frac=rng.uniform(0.1, 0.9)),
        5: None,  # QR / morse — handled separately
    }
    params = vis_params[min(key_visibility, 4)]

    # Create overlay frame (RGBA)
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Try to load a monospace font; fall back to default
    font = None
    for font_path in ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
                      "/System/Library/Fonts/Menlo.ttc",
                      "/usr/share/fonts/TTF/DejaVuSansMono.ttf"]:
        if Path(font_path).exists():
            try:
                font = ImageFont.truetype(font_path, params["fontsize"])
                break
            except Exception:
                pass

    r, g, b = params["color"]
    color_rgba = (r, g, b, params["alpha"])

    # Estimate text size
    test_img = Image.new("RGBA", (1, 1))
    test_draw = ImageDraw.Draw(test_img)
    try:
        bbox = test_draw.textbbox((0, 0), key, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th = params["fontsize"] * len(key) * 0.6, params["fontsize"]

    x = int(params["x_frac"] * w - tw / 2)
    y = int(params["y_frac"] * h - th / 2)
    x = max(5, min(x, w - int(tw) - 5))
    y = max(5, min(y, h - int(th) - 5))

    # Drop shadow for readability
    draw.text((x + 2, y + 2), key, fill=(0, 0, 0, params["alpha"]), font=font)
    draw.text((x, y), key, fill=color_rgba, font=font)

    overlay_path = tempfile.mktemp(suffix="_overlay.png")
    overlay.save(overlay_path)

    # Overlay on video
    duration = end_t - start_t
    cmd = [
        "ffmpeg", "-y", "-i", input_video, "-i", overlay_path,
        "-filter_complex",
        f"[0:v][1:v]overlay=0:0:enable='between(t,{start_t},{end_t})'[v]",
        "-map", "[v]", "-map", "0:a?",
        "-c:a", "copy", output_video,
    ]
    result = subprocess.run(cmd, capture_output=True)
    os.unlink(overlay_path)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg overlay failed: {result.stderr.decode()[-300:]}")


def embed_key_qr(input_video: str, output_video: str, key: str, start_t: float, end_t: float) -> None:
    """Embed key as QR code (key_visibility=5)."""
    try:
        import qrcode
        qr = qrcode.make(key)
        qr = qr.resize((200, 200))
        qr_path = tempfile.mktemp(suffix="_qr.png")
        qr.save(qr_path)
        cmd = [
            "ffmpeg", "-y", "-i", input_video, "-i", qr_path,
            "-filter_complex",
            f"[0:v][1:v]overlay=10:10:enable='between(t,{start_t},{end_t})'[v]",
            "-map", "[v]", "-map", "0:a?", "-c:a", "copy", output_video,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        os.unlink(qr_path)
    except ImportError:
        # fallback to text if qrcode not installed
        embed_key_overlay(input_video, output_video, key, 2, start_t, end_t, random.Random(42))


# ---------------------------------------------------------------------------
# ffmpeg edit chain
# ---------------------------------------------------------------------------

def apply_ops(input_video: str, output_video: str, ops: list[dict]) -> None:
    """Apply a sequence of ops to a video, chaining through temp files."""
    src = input_video
    tmp_files = []

    for i, op in enumerate(ops):
        is_last = (i == len(ops) - 1)
        dst = output_video if is_last else tempfile.mktemp(suffix=f"_step{i}.mp4")
        if not is_last:
            tmp_files.append(dst)

        vf = op["vf"]
        af = op.get("af")

        cmd = ["ffmpeg", "-y", "-i", src]
        if vf:
            cmd += ["-vf", vf]
        if af:
            cmd += ["-af", af]
        elif not af and vf:
            cmd += ["-c:a", "copy"]
        cmd.append(dst)

        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg op '{op['name']}' failed: {result.stderr.decode()[-300:]}")
        src = dst

    for f in tmp_files:
        try:
            os.unlink(f)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Background pool
# ---------------------------------------------------------------------------

class BackgroundPool:
    """
    Manages a collection of background video files for procedural generation.
    Videos are selected deterministically by seed — same seed always picks
    the same background clip and offset.

    Populate with add() before generating samples.
    Recommended minimum: 10+ diverse clips per pool.
    """

    def __init__(self, videos: list[str] = None):
        self._videos: list[str] = []
        for v in (videos or []):
            self.add(v)

    def add(self, path: str) -> None:
        p = os.path.abspath(path)
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        self._videos.append(p)

    def __len__(self):
        return len(self._videos)

    def pick(self, rng: random.Random) -> str:
        if not self._videos:
            raise RuntimeError("BackgroundPool is empty — add videos first")
        return rng.choice(self._videos)

    @classmethod
    def from_dir(cls, directory: str, exts: tuple = (".mp4", ".mkv", ".webm")) -> "BackgroundPool":
        pool = cls()
        for p in sorted(Path(directory).glob("**/*")):
            if p.suffix.lower() in exts:
                pool.add(str(p))
        return pool


# ---------------------------------------------------------------------------
# Sample dataclass
# ---------------------------------------------------------------------------

@dataclass
class CaptchaSample:
    sample_id: str
    level: str
    key: str
    ops: list[str]          # op names in order
    nl_instructions: str
    scrambled_video: str    # path
    background_video: str   # which source clip was used
    key_visibility: int
    op_count: int
    nl_ambiguity: int
    key_start_t: float
    key_end_t: float
    seed: int
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Sample generator
# ---------------------------------------------------------------------------

def generate_sample(
    seed: int,
    level: str,
    pool: BackgroundPool,
    output_dir: str,
    clip_duration: float = 15.0,
    key_length: int = 7,
) -> CaptchaSample:
    """
    Generate one benchmark sample deterministically from seed + level.
    Background video is drawn from pool — same seed always yields same result.
    """
    cfg = LEVELS[level]
    rng = random.Random(seed)

    os.makedirs(output_dir, exist_ok=True)
    sample_id = f"{level}_s{seed:06d}"

    # 1. Select ops
    chosen_ops = rng.sample(CHAIN_OPS, k=min(cfg["op_count"], len(CHAIN_OPS)))

    # 2. Generate key
    key = _random_key(rng, key_length)

    # 3. Pick background video and extract clip
    bg_video = pool.pick(rng)
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", bg_video],
        capture_output=True, text=True,
    )
    try:
        total_dur = float(probe.stdout.strip())
    except Exception:
        total_dur = 60.0

    clip_start = rng.uniform(0, max(0, total_dur - clip_duration - 1))
    clip_path = os.path.join(output_dir, f"{sample_id}_clip.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(clip_start), "-i", bg_video,
         "-t", str(clip_duration), "-c", "copy", clip_path],
        capture_output=True, check=True,
    )

    # 4. Embed key
    key_start = rng.uniform(3.0, clip_duration - 6.0)
    key_end = min(key_start + rng.uniform(2.5, 4.5), clip_duration - 0.5)

    keyed_path = os.path.join(output_dir, f"{sample_id}_keyed.mp4")
    embed_key_overlay(clip_path, keyed_path, key,
                      cfg["key_visibility"], key_start, key_end, rng)

    # 5. Apply scramble ops
    scrambled_path = os.path.join(output_dir, f"{sample_id}_scrambled.mp4")
    apply_ops(keyed_path, scrambled_path, chosen_ops)

    # 6. NL instructions
    instructions = build_instructions(chosen_ops, cfg["nl_ambiguity"], rng)

    # 7. Cleanup intermediates
    for p in [clip_path, keyed_path]:
        try:
            os.unlink(p)
        except Exception:
            pass

    return CaptchaSample(
        sample_id=sample_id,
        level=level,
        key=key,
        ops=[op["name"] for op in chosen_ops],
        nl_instructions=instructions,
        scrambled_video=scrambled_path,
        background_video=bg_video,
        key_visibility=cfg["key_visibility"],
        op_count=cfg["op_count"],
        nl_ambiguity=cfg["nl_ambiguity"],
        key_start_t=key_start,
        key_end_t=key_end,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

def generate_split(
    level: str,
    n_samples: int,
    pool: BackgroundPool,
    output_dir: str,
    seed_offset: int = 0,
) -> list[CaptchaSample]:
    level_hash = int(hashlib.md5(level.encode()).hexdigest()[:8], 16) % 100_000
    samples = []
    for i in range(n_samples):
        seed = seed_offset + level_hash + i
        sample = generate_sample(seed, level, pool,
                                 os.path.join(output_dir, level))
        samples.append(sample)
        bg = os.path.basename(sample.background_video)
        print(f"  [{level}] {i+1}/{n_samples} key={sample.key} ops={sample.ops} bg={bg}")
    return samples


def generate_benchmark(
    pool: BackgroundPool,
    output_dir: str,
    n_per_level: int = 50,
    levels: list[str] = None,
    seed_offset: int = 0,
) -> dict:
    """Generate full benchmark across all levels. Returns manifest dict."""
    if levels is None:
        levels = list(LEVELS.keys())

    manifest = {
        "levels": {},
        "n_per_level": n_per_level,
        "pool_size": len(pool),
    }

    for level in levels:
        print(f"\n=== Generating {level} ({n_per_level} samples) ===")
        samples = generate_split(level, n_per_level, pool, output_dir, seed_offset)
        manifest["levels"][level] = [s.to_dict() for s in samples]

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved: {manifest_path}")
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate CIPHER benchmark")
    parser.add_argument("--videos", nargs="+", required=True,
                        help="Background video files (or dirs) — Kinetics clips recommended")
    parser.add_argument("--output", default="./data", help="Output directory")
    parser.add_argument("--levels", nargs="+", default=list(LEVELS.keys()),
                        choices=list(LEVELS.keys()), help="Difficulty levels to generate")
    parser.add_argument("--n", type=int, default=50, help="Samples per level")
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()

    pool = BackgroundPool()
    for v in args.videos:
        if os.path.isdir(v):
            pool2 = BackgroundPool.from_dir(v)
            for vid in pool2._videos:
                pool.add(vid)
        else:
            pool.add(v)
    print(f"Background pool: {len(pool)} videos")

    generate_benchmark(
        pool=pool,
        output_dir=args.output,
        n_per_level=args.n,
        levels=args.levels,
        seed_offset=args.seed_offset,
    )
