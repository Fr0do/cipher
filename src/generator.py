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
    # distractor: 0=none, 1=1 fake string, 2=2 fakes, 3=2 fakes + color similar to key
    # nl_ambiguity: 1=canonical, 2=paraphrase, 3=reversed desc, 4=compositional
    # NOTE: nl_ambiguity=4 breaks regex S1 parse; use max=3 for scoreable calibration
    "L1": dict(key_visibility=1, op_count=1, nl_ambiguity=1, distractor=0),
    "L2": dict(key_visibility=1, op_count=2, nl_ambiguity=1, distractor=0),
    "L3": dict(key_visibility=2, op_count=2, nl_ambiguity=2, distractor=1),
    "L4": dict(key_visibility=2, op_count=3, nl_ambiguity=2, distractor=2),
    "L5": dict(key_visibility=3, op_count=4, nl_ambiguity=3, distractor=3),
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
        # All phrases parseable by regex: \breverse\b, play.*in reverse, \bbackward\b
        "nl": [
            "reverse the video",
            "play the video in reverse",
            "run the video backward",
            "reverse it",
            "play it backward",
        ],
    },
    {
        "name": "hflip",
        "vf": "hflip", "af": None,
        "inverse_vf": "hflip", "inverse_af": None,
        # All phrases parseable by regex: flip.*horizontally, mirror.*left.to.right, horizontal.*flip
        "nl": [
            "flip the video horizontally",
            "mirror the video left to right",
            "apply a horizontal mirror",
            "flip it left-right",
            "horizontally flip the video",
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
        # Scramble uses inverse (CCW), so NL tells agent to apply CW to restore.
        # All phrases must parse to {op:"rotate", degrees:90} via regex.
        "nl": [
            "rotate the video 90 degrees clockwise",
            "rotate it 90 degrees clockwise",
            "turn the video clockwise by 90 degrees",
            "rotate 90 degrees clockwise",
            "rotate the video 90°",
        ],
    },
    {
        "name": "rotate270",
        "vf": "transpose=2", "af": None,
        "inverse_vf": "transpose=1", "inverse_af": None,
        # Scramble uses inverse (CW), so NL tells agent to apply CCW (270°) to restore.
        # All phrases must parse to {op:"rotate", degrees:270} via regex.
        "nl": [
            "rotate the video 270 degrees clockwise",
            "rotate it 270 degrees clockwise",
            "turn the video clockwise by 270 degrees",
            "rotate 270 degrees clockwise",
            "rotate the video 270°",
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
        # Scramble uses 0.5x (inverse), so NL tells agent to apply 2x (restore).
        # All phrases parseable by regex: speed\s+up.*2x, double.*speed, 2x\s+speed
        "nl": [
            "speed up the video 2x",
            "double the playback speed",
            "speed it up 2x",
            "apply 2x speed",
            "speed up 2x",
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
    n_distractors: int = 0,
) -> None:
    """Burn key text onto video frames during [start_t, end_t].
    Optionally add n_distractors fake alphanumeric strings at different positions/windows.
    Distractors appear at non-overlapping windows with slightly different styling.
    """
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

    # Key rendering params by visibility level.
    # KEY is ALWAYS bright yellow — the unambiguous signal. Only alpha/size/position vary.
    # Distractors use different colors (never yellow) and are always smaller + dimmer.
    vis_params = {
        1: dict(fontsize=int(h * 0.13), color=(255, 255, 0), alpha=255, x_frac=0.5,               y_frac=0.5),
        2: dict(fontsize=int(h * 0.10), color=(255, 255, 0), alpha=240, x_frac=rng.uniform(0.3, 0.7), y_frac=rng.uniform(0.3, 0.7)),
        3: dict(fontsize=int(h * 0.07), color=(255, 255, 0), alpha=200, x_frac=rng.uniform(0.2, 0.8), y_frac=rng.uniform(0.2, 0.8)),
        4: dict(fontsize=int(h * 0.06), color=(255, 255, 0), alpha=170, x_frac=rng.uniform(0.1, 0.9), y_frac=rng.uniform(0.1, 0.9)),
        5: None,  # QR / morse — reserved
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

    # Build distractors.
    # Invariants (so VLM can use heuristics to find the real key):
    #   - Distractors are NEVER yellow (reserved for key)
    #   - Distractor font <= 80% of key font size
    #   - Distractor alpha <= key_alpha - 40
    #   - Distractor window duration < key duration
    #   - L5 only: one distractor is white (close to key at visibility=4) — hardest case
    distractor_overlays = []
    key_duration = end_t - start_t
    clip_dur = end_t + 3.0  # safe upper bound for distractor placement
    # Colors for distractors: cyan, green, magenta — never yellow
    distractor_palette = [(0, 220, 255), (80, 255, 80), (255, 100, 255)]
    d_font_size = max(10, int(params["fontsize"] * 0.70))
    d_alpha_max = max(80, params["alpha"] - 50)

    try:
        d_font = ImageFont.truetype(font._file, d_font_size) if font else None
    except Exception:
        d_font = font

    for d_idx in range(n_distractors):
        # Place distractor in a non-overlapping window, strictly shorter than key
        d_duration = rng.uniform(0.8, min(1.8, key_duration - 0.5))
        if d_idx % 2 == 0:
            # Before key
            d_end = max(d_duration + 0.2, start_t - rng.uniform(0.3, 1.0))
            d_start = d_end - d_duration
        else:
            # After key
            d_start = end_t + rng.uniform(0.3, 1.0)
            d_end = d_start + d_duration
        d_start = max(0.0, d_start)

        fake_key = _random_key(rng, rng.randint(4, 6))  # shorter than real key (7)
        if n_distractors >= 3 and d_idx == 0:
            # L5: first distractor is white (harder to distinguish from key)
            d_color = (220, 220, 220)
        else:
            d_color = distractor_palette[d_idx % len(distractor_palette)]

        d_overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d_draw = ImageDraw.Draw(d_overlay)
        dx = int(rng.uniform(0.05, 0.80) * w)
        dy = int(rng.uniform(0.05, 0.80) * h)
        d_alpha = rng.randint(max(60, d_alpha_max - 40), d_alpha_max)
        d_draw.text((dx + 2, dy + 2), fake_key, fill=(0, 0, 0, d_alpha), font=d_font)
        d_draw.text((dx, dy), fake_key, fill=(*d_color, d_alpha), font=d_font)
        d_path = tempfile.mktemp(suffix=f"_distractor{d_idx}.png")
        d_overlay.save(d_path)
        distractor_overlays.append((d_path, d_start, min(d_end, clip_dur - 0.1)))

    # Build ffmpeg filter chain: chain key + all distractor overlays
    inputs = ["-i", input_video, "-i", overlay_path]
    for d_path, _, _ in distractor_overlays:
        inputs += ["-i", d_path]

    n_inputs = 1 + len(distractor_overlays)  # key + distractors
    # Build overlay filter graph
    filters = []
    prev = "0:v"
    for i in range(n_inputs):
        label_in = f"[{prev}]" if i > 0 else f"[{prev}]"
        img_idx = i + 1  # input index (0 = video)
        if i == 0:
            t_s, t_e = start_t, end_t
        else:
            t_s, t_e = distractor_overlays[i - 1][1], distractor_overlays[i - 1][2]
        out_label = f"[v{i}]" if i < n_inputs - 1 else "[vout]"
        filters.append(
            f"[{prev}][{img_idx}:v]overlay=0:0:enable='between(t,{t_s},{t_e})'{out_label}"
        )
        prev = f"v{i}"

    filter_str = ";".join(filters)
    cmd = ["ffmpeg", "-y"] + inputs + [
        "-filter_complex", filter_str,
        "-map", "[vout]", "-map", "0:a?", "-c:a", "copy", output_video,
    ]
    result = subprocess.run(cmd, capture_output=True)
    os.unlink(overlay_path)
    for d_path, _, _ in distractor_overlays:
        try:
            os.unlink(d_path)
        except Exception:
            pass
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

def apply_ops(input_video: str, output_video: str, ops: list[dict], invert: bool = False) -> None:
    """Apply a sequence of ops to a video, chaining through temp files.

    If invert=True, applies the inverse operations in reversed order.
    Use invert=True for scrambling so that NL instructions describe what the
    agent should apply (forward ops) to restore the original.

    For involutions (reverse, hflip, vflip): forward == inverse, no difference.
    For speed2x: invert applies 0.5x so NL "speed up 2x" correctly restores.
    For rotate90 CW: invert applies CCW so NL "rotate CW" correctly restores.
    """
    src = input_video
    tmp_files = []

    # When inverting, apply ops in reversed order (B⁻¹ then A⁻¹ undoes A then B)
    ops_seq = list(reversed(ops)) if invert else ops

    for i, op in enumerate(ops_seq):
        is_last = (i == len(ops_seq) - 1)
        dst = output_video if is_last else tempfile.mktemp(suffix=f"_step{i}.mp4")
        if not is_last:
            tmp_files.append(dst)

        if invert:
            vf = op.get("inverse_vf") or op["vf"]   # fallback to forward for involutions
            af = op.get("inverse_af") if "inverse_af" in op else op.get("af")
        else:
            vf = op["vf"]
            af = op.get("af")

        cmd = ["ffmpeg", "-y", "-i", src]
        if vf:
            cmd += ["-vf", vf]
        if af:
            cmd += ["-af", af]
        elif vf:
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

    # 4. Embed key + distractors
    key_start = rng.uniform(3.0, clip_duration - 6.0)
    key_end = min(key_start + rng.uniform(2.5, 4.5), clip_duration - 0.5)

    keyed_path = os.path.join(output_dir, f"{sample_id}_keyed.mp4")
    embed_key_overlay(clip_path, keyed_path, key,
                      cfg["key_visibility"], key_start, key_end, rng,
                      n_distractors=cfg["distractor"])

    # 5. Apply scramble ops using INVERSE so NL describes the forward correction.
    # e.g. speed2x scramble → apply 0.5x; agent sees "speed up 2x" and restores.
    scrambled_path = os.path.join(output_dir, f"{sample_id}_scrambled.mp4")
    apply_ops(keyed_path, scrambled_path, chosen_ops, invert=True)

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
