#!/usr/bin/env python3
"""
rl_optimize.py — evolutionary optimization of CIPHER watermark parameters.

Goal: find watermark construction params where frontier VLMs achieve 2-5%
full-pipeline accuracy while humans still solve reliably.

The eval_agent (gemini/claude/codex) and eval_effort (fast/default/detailed)
are part of the genome — they co-evolve with watermark params so the search
finds the hardest (watermark, model) combination.

Architecture:
  Generation (local ffmpeg) → key frames on disk → eval manifest
  Evaluation (pluggable: Swarm agents / direct API) → predictions.json
  Fitness computation → evolutionary selection → next generation

Eval dispatch modes:
  Swarm    — dispatch via `rl_optimize.py dispatch`, spawns gemini/codex/claude
             agents through Swarm MCP. Orchestrator polls with `collect`.
  Hermes   — status messages sent to hermes target (telegram/discord/slack).
  API      — direct VLM API call via `auto` command (needs API key in env).

Orchestrated loop (Claude Code + Swarm + Hermes):
  1. python scripts/rl_optimize.py init --background data/backgrounds/
  2. python scripts/rl_optimize.py step --work-dir runs/rl/
  3. python scripts/rl_optimize.py dispatch --work-dir runs/rl/ [--hermes-target telegram:ID]
     → orchestrator reads dispatch_plan.json, spawns Swarm agents
  4. python scripts/rl_optimize.py collect --work-dir runs/rl/
     → check if all predictions are in
  5. python scripts/rl_optimize.py evolve --work-dir runs/rl/
     → compute fitness, evolve population, repeat from step 2

Autonomous loop (direct API, no agents):
  python scripts/rl_optimize.py auto --background data/backgrounds/ \
      --eval-backend api:haiku --target-acc 0.03 --output runs/rl/
"""
import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from generator import (
    BackgroundPool, CaptchaSample, CHAIN_OPS, OPS,
    _random_key, apply_ops, build_instructions,
)


# ===================================================================
# WatermarkGenome — the tunable parameter space
# ===================================================================

@dataclass
class WatermarkGenome:
    """All tunable parameters for watermark construction."""
    # --- Key text rendering ---
    fontsize_pct: float = 0.07       # % of frame height (0.02 - 0.15)
    alpha: int = 200                 # text opacity (30 - 255)
    key_length: int = 7              # chars (5 - 12)
    font_variant: int = 0            # 0=mono-bold, 1=mono-regular, 2=sans, 3=serif

    # --- Temporal ---
    key_duration_s: float = 3.0      # seconds key is visible (0.08 - 5.0)
    key_flash_hz: float = 0.0        # blink frequency (0=static, 2-8=flashing)
    key_start_range: tuple = (3.0, 9.0)

    # --- Spatial ---
    position_jitter: float = 0.0     # per-frame random offset pixels (0 - 20)
    position_mode: str = "random"    # center / random / corner / edge

    # --- Distractors ---
    n_distractors: int = 3           # 0 - 10
    yellow_distractors: int = 0      # how many are also yellow
    distractor_same_length: bool = False

    # --- Visual perturbation ---
    noise_sigma: float = 0.0         # Gaussian noise (0 - 40)
    blur_radius: float = 0.0         # text blur (0 - 3.0)
    bg_blend_alpha: float = 0.0      # blend with background (0 - 0.5)

    # --- Pipeline difficulty ---
    op_count: int = 3                # 1 - 5
    nl_ambiguity: int = 2            # 1 - 3

    # --- Drop shadow ---
    shadow: bool = True
    shadow_blur: float = 0.0

    # --- Eval model (part of genome — co-evolve with watermark) ---
    eval_agent: str = "gemini"       # gemini / claude / codex
    eval_effort: str = "fast"        # fast / default / detailed
    eval_model_label: str = ""       # auto-filled: "gemini:fast", "codex:detailed" etc.

    def to_dict(self):
        d = asdict(self)
        d["key_start_range"] = list(d["key_start_range"])
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        if "key_start_range" in d:
            d["key_start_range"] = tuple(d["key_start_range"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ===================================================================
# Genome mutation / crossover
# ===================================================================

def mutate(genome: WatermarkGenome, rng: random.Random, strength: float = 0.3) -> WatermarkGenome:
    g = WatermarkGenome.from_dict(genome.to_dict())

    def nf(val, lo, hi, scale=None):
        scale = scale or (hi - lo) * strength
        return max(lo, min(hi, val + rng.gauss(0, scale)))

    def ni(val, lo, hi):
        delta = max(1, int((hi - lo) * strength * 0.5))
        return max(lo, min(hi, val + rng.randint(-delta, delta)))

    if rng.random() < 0.6: g.fontsize_pct = nf(g.fontsize_pct, 0.015, 0.15)
    if rng.random() < 0.6: g.alpha = ni(g.alpha, 25, 255)
    if rng.random() < 0.3: g.key_length = ni(g.key_length, 5, 12)
    if rng.random() < 0.3: g.font_variant = rng.randint(0, 3)
    if rng.random() < 0.7: g.key_duration_s = nf(g.key_duration_s, 0.08, 5.0)
    if rng.random() < 0.4: g.key_flash_hz = nf(g.key_flash_hz, 0.0, 10.0)
    if rng.random() < 0.4: g.position_jitter = nf(g.position_jitter, 0.0, 25.0)
    if rng.random() < 0.3: g.position_mode = rng.choice(["center", "random", "corner", "edge"])
    if rng.random() < 0.5: g.n_distractors = ni(g.n_distractors, 0, 10)
    if rng.random() < 0.5: g.yellow_distractors = ni(g.yellow_distractors, 0, g.n_distractors)
    if rng.random() < 0.3: g.distractor_same_length = rng.random() < 0.5
    if rng.random() < 0.5: g.noise_sigma = nf(g.noise_sigma, 0.0, 40.0)
    if rng.random() < 0.4: g.blur_radius = nf(g.blur_radius, 0.0, 3.0)
    if rng.random() < 0.4: g.bg_blend_alpha = nf(g.bg_blend_alpha, 0.0, 0.5)
    if rng.random() < 0.3: g.op_count = ni(g.op_count, 1, 5)
    if rng.random() < 0.3: g.nl_ambiguity = ni(g.nl_ambiguity, 1, 3)
    if rng.random() < 0.2: g.shadow = rng.random() < 0.5
    if rng.random() < 0.3: g.shadow_blur = nf(g.shadow_blur, 0.0, 5.0)
    if rng.random() < 0.3: g.eval_agent = rng.choice(["gemini", "claude", "codex"])
    if rng.random() < 0.3: g.eval_effort = rng.choice(["fast", "default", "detailed"])
    g.eval_model_label = f"{g.eval_agent}:{g.eval_effort}"
    return g


def crossover(a: WatermarkGenome, b: WatermarkGenome, rng: random.Random) -> WatermarkGenome:
    da, db = a.to_dict(), b.to_dict()
    child = {k: da[k] if rng.random() < 0.5 else db[k] for k in da}
    return WatermarkGenome.from_dict(child)


def initial_population(size: int, rng: random.Random) -> list[WatermarkGenome]:
    pop = []
    for i in range(size):
        t = i / max(size - 1, 1)  # 0 (easy) → 1 (hard)
        pop.append(WatermarkGenome(
            fontsize_pct=0.12 - t * 0.09,
            alpha=int(255 - t * 180),
            key_length=7,
            key_duration_s=4.0 - t * 3.5,
            key_flash_hz=t * 6.0,
            n_distractors=int(t * 8),
            yellow_distractors=int(t * 4),
            distractor_same_length=t > 0.5,
            noise_sigma=t * 25.0,
            blur_radius=t * 2.0,
            bg_blend_alpha=t * 0.3,
            op_count=min(1 + int(t * 4), 5),
            nl_ambiguity=min(1 + int(t * 2), 3),
            position_mode="center" if t < 0.3 else ("random" if t < 0.6 else rng.choice(["corner", "edge"])),
            position_jitter=t * 15.0,
            shadow=t < 0.6,
            shadow_blur=t * 3.0,
            eval_agent=rng.choice(["gemini", "claude", "codex"]),
            eval_effort=["fast", "default", "detailed"][min(int(t * 3), 2)],
        ))
        pop[-1].eval_model_label = f"{pop[-1].eval_agent}:{pop[-1].eval_effort}"
    return pop


# ===================================================================
# Sample generation (local, uses ffmpeg)
# ===================================================================

def _get_font(variant: int, size: int):
    font_paths = {
        0: ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
            "/System/Library/Fonts/Menlo.ttc"],
        1: ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/System/Library/Fonts/Monaco.ttf"],
        2: ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc"],
        3: ["/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "/System/Library/Fonts/Times.ttc"],
    }
    for p in font_paths.get(variant, font_paths[0]):
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def generate_sample(
    seed: int,
    genome: WatermarkGenome,
    pool: BackgroundPool,
    output_dir: str,
    clip_duration: float = 15.0,
) -> dict:
    """Generate one sample. Returns dict with paths and gold data.

    Saves both the scrambled video AND the key frame (corrected, with key visible)
    so that eval agents can test either S3-only or full-pipeline.
    """
    rng = random.Random(seed)
    os.makedirs(output_dir, exist_ok=True)
    sample_id = f"rl_s{seed:06d}"

    # 1. Select ops
    chosen_ops = rng.sample(CHAIN_OPS, k=min(genome.op_count, len(CHAIN_OPS)))

    # 2. Generate key
    key = _random_key(rng, genome.key_length)

    # 3. Background clip
    bg_video = pool.pick(rng)
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", bg_video], capture_output=True, text=True)
    total_dur = float(probe.stdout.strip()) if probe.returncode == 0 else 60.0
    clip_start = rng.uniform(0, max(0, total_dur - clip_duration - 1))
    clip_path = os.path.join(output_dir, f"{sample_id}_clip.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(clip_start), "-i", bg_video,
         "-t", str(clip_duration), "-c", "copy", clip_path],
        capture_output=True, check=True)

    # 4. Probe dimensions
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", clip_path],
        capture_output=True, text=True)
    try:
        w, h = map(int, probe.stdout.strip().split(","))
    except Exception:
        w, h = 640, 360

    # 5. Key timing
    lo, hi = genome.key_start_range
    key_start = rng.uniform(lo, min(hi, clip_duration - genome.key_duration_s - 0.5))
    key_end = key_start + genome.key_duration_s

    # 6. Build key overlay
    fontsize = max(8, int(h * genome.fontsize_pct))
    font = _get_font(genome.font_variant, fontsize)
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        bbox = draw.textbbox((0, 0), key, font=font)
        tw, th_ = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th_ = fontsize * len(key) * 0.6, fontsize

    # Position
    mode = genome.position_mode
    if mode == "center":
        x, y = int(w/2 - tw/2), int(h/2 - th_/2)
    elif mode == "corner":
        cx, cy = rng.choice([(0.05, 0.05), (0.05, 0.85), (0.85, 0.05), (0.85, 0.85)])
        x, y = int(cx * w), int(cy * h)
    elif mode == "edge":
        edge = rng.choice(["top", "bottom", "left", "right"])
        if edge == "top":    x, y = int(rng.uniform(0.1, 0.7) * w), int(0.03 * h)
        elif edge == "bottom": x, y = int(rng.uniform(0.1, 0.7) * w), int(0.90 * h)
        elif edge == "left":  x, y = int(0.03 * w), int(rng.uniform(0.1, 0.7) * h)
        else:                 x, y = int(0.85 * w), int(rng.uniform(0.1, 0.7) * h)
    else:
        x, y = int(rng.uniform(0.05, 0.80) * w), int(rng.uniform(0.05, 0.80) * h)
    x = max(2, min(x, w - int(tw) - 2))
    y = max(2, min(y, h - int(th_) - 2))

    if genome.shadow:
        draw.text((x+2, y+2), key, fill=(0, 0, 0, min(255, genome.alpha)), font=font)
    draw.text((x, y), key, fill=(255, 255, 0, genome.alpha), font=font)

    if genome.blur_radius > 0:
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=genome.blur_radius))
    if genome.noise_sigma > 0:
        arr = np.array(overlay).astype(np.float32)
        noise = np.random.RandomState(seed).normal(0, genome.noise_sigma, arr.shape[:2] + (3,))
        arr[:, :, :3] = np.clip(arr[:, :, :3] + noise, 0, 255)
        overlay = Image.fromarray(arr.astype(np.uint8))

    overlay_path = tempfile.mktemp(suffix="_overlay.png")
    overlay.save(overlay_path)

    # 7. Distractors
    distractor_overlays = []
    palette = [(0, 220, 255), (80, 255, 80), (255, 100, 255), (255, 180, 50), (180, 120, 255)]
    d_font = _get_font(genome.font_variant, max(8, int(fontsize * 0.75)))
    d_alpha_max = max(60, genome.alpha - 40)

    for d_idx in range(genome.n_distractors):
        d_len = genome.key_length if genome.distractor_same_length else rng.randint(4, 6)
        fake_key = _random_key(rng, d_len)
        d_dur = rng.uniform(0.5, min(1.5, max(0.6, genome.key_duration_s - 0.2)))
        if d_idx % 2 == 0:
            d_end = max(d_dur + 0.2, key_start - rng.uniform(0.2, 0.8))
            d_start = d_end - d_dur
        else:
            d_start = key_end + rng.uniform(0.2, 0.8)
            d_end = d_start + d_dur
        d_start = max(0.0, d_start)

        if d_idx < genome.yellow_distractors:
            d_color = (255, 255, 0)
        else:
            d_color = palette[d_idx % len(palette)]
        d_alpha = rng.randint(max(40, d_alpha_max - 40), d_alpha_max)

        d_ov = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d_draw = ImageDraw.Draw(d_ov)
        dx, dy = int(rng.uniform(0.05, 0.80) * w), int(rng.uniform(0.05, 0.80) * h)
        d_draw.text((dx+2, dy+2), fake_key, fill=(0, 0, 0, d_alpha), font=d_font)
        d_draw.text((dx, dy), fake_key, fill=(*d_color, d_alpha), font=d_font)
        d_path = tempfile.mktemp(suffix=f"_d{d_idx}.png")
        d_ov.save(d_path)
        distractor_overlays.append((d_path, d_start, min(d_end, clip_duration - 0.1)))

    # 8. ffmpeg overlay
    if genome.key_flash_hz > 0:
        period = 1.0 / genome.key_flash_hz
        enable_expr = f"between(t,{key_start},{key_end})*lt(mod(t-{key_start},{period}),{period/2})"
    else:
        enable_expr = f"between(t,{key_start},{key_end})"

    inputs = ["-i", clip_path, "-i", overlay_path]
    for dp, _, _ in distractor_overlays:
        inputs += ["-i", dp]

    n_inp = 1 + len(distractor_overlays)
    filters = []
    prev = "0:v"
    for i in range(n_inp):
        img_idx = i + 1
        if i == 0:
            en = f"enable='{enable_expr}'"
        else:
            ds, de = distractor_overlays[i-1][1], distractor_overlays[i-1][2]
            en = f"enable='between(t,{ds},{de})'"
        if genome.position_jitter > 0 and i == 0:
            jit = int(genome.position_jitter)
            opos = f"x=(random(1)*{2*jit}-{jit}):y=(random(2)*{2*jit}-{jit})"
        else:
            opos = "0:0"
        out = f"v{i}" if i < n_inp - 1 else "vout"
        filters.append(f"[{prev}][{img_idx}:v]overlay={opos}:{en}[{out}]")
        prev = f"v{i}"

    keyed_path = os.path.join(output_dir, f"{sample_id}_keyed.mp4")
    cmd = ["ffmpeg", "-y"] + inputs + [
        "-filter_complex", ";".join(filters),
        "-map", "[vout]", "-map", "0:a?", "-c:a", "copy", keyed_path]
    result = subprocess.run(cmd, capture_output=True)
    os.unlink(overlay_path)
    for dp, _, _ in distractor_overlays:
        try: os.unlink(dp)
        except: pass
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg overlay failed: {result.stderr.decode()[-300:]}")

    # 9. Extract key frame (for S3-only eval)
    key_mid = (key_start + key_end) / 2
    key_frame_path = os.path.join(output_dir, f"{sample_id}_keyframe.jpg")
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(key_mid), "-i", keyed_path,
         "-frames:v", "1", "-q:v", "2", key_frame_path],
        capture_output=True)

    # 10. Scramble
    scrambled_path = os.path.join(output_dir, f"{sample_id}_scrambled.mp4")
    apply_ops(keyed_path, scrambled_path, chosen_ops, invert=True)

    # 11. NL instructions
    instructions = build_instructions(chosen_ops, genome.nl_ambiguity, rng)

    # Cleanup intermediates
    for p in [clip_path, keyed_path]:
        try: os.unlink(p)
        except: pass

    return {
        "sample_id": sample_id,
        "seed": seed,
        "gold_key": key,
        "gold_ops": [op["name"] for op in chosen_ops],
        "nl_instructions": instructions,
        "scrambled_video": os.path.abspath(scrambled_path),
        "key_frame": os.path.abspath(key_frame_path),
        "key_start_t": key_start,
        "key_end_t": key_end,
    }


# ===================================================================
# Eval manifest protocol
# ===================================================================
#
# eval_manifest.json:
#   { "config_idx": 0, "genome": {...}, "samples": [
#       {"sample_id": "...", "key_frame": "/abs/path.jpg",
#        "scrambled_video": "/abs/path.mp4", "nl_instructions": "...",
#        "gold_key": "ABC1234"},
#     ...
#   ]}
#
# predictions.json (written by eval agent):
#   { "predictions": [
#       {"sample_id": "...", "pred_key": "ABC1234"},
#     ...
#   ]}
#
# The eval agent prompt (for S3-only):
#   "Read each image listed below. Each contains a short alphanumeric key
#    rendered as yellow text. Return ONLY the key string for each.
#    Output JSON: {predictions: [{sample_id, pred_key}, ...]}"

def write_eval_manifest(config_idx: int, genome: WatermarkGenome,
                        samples: list[dict], out_dir: str) -> str:
    manifest = {
        "config_idx": config_idx,
        "genome": genome.to_dict(),
        "eval_mode": "s3_only",
        "eval_agent": genome.eval_agent,
        "eval_effort": genome.eval_effort,
        "eval_model_label": f"{genome.eval_agent}:{genome.eval_effort}",
        "samples": samples,
    }
    path = os.path.join(out_dir, f"config_{config_idx:02d}_manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


def build_eval_prompt(manifest_path: str, eval_mode: str = "s3_only") -> str:
    """Build the prompt that eval agents receive."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    samples = manifest["samples"]

    if eval_mode == "s3_only":
        frame_list = "\n".join(
            f"  - sample_id={s['sample_id']}, image={s['key_frame']}"
            for s in samples
        )
        return f"""You are evaluating a visual captcha benchmark.
For each image below, look for a short alphanumeric key (5-12 characters) rendered as colored text overlaid on the video frame. The key may be small, faint, blurred, or partially hidden.

Read each image file and extract the key text you see. If you can't find any key, use "NONE".

Images:
{frame_list}

IMPORTANT: Write your predictions as JSON to: {manifest_path.replace('_manifest.json', '_predictions.json')}
Format: {{"predictions": [{{"sample_id": "...", "pred_key": "..."}}, ...]}}

Use the Read tool to view each image, then write the predictions file."""

    else:  # full_pipeline
        sample_list = "\n".join(
            f"  - sample_id={s['sample_id']}\n"
            f"    video={s['scrambled_video']}\n"
            f"    instructions: {s['nl_instructions']}"
            for s in samples
        )
        return f"""You are evaluating a video-editing captcha benchmark.
For each sample, you receive a scrambled video and natural-language editing instructions.
You must:
1. Parse the instructions to identify ffmpeg operations
2. Apply the operations using ffmpeg (you have Bash access)
3. Extract the alphanumeric key from the corrected video

Samples:
{sample_list}

Write predictions as JSON to: {manifest_path.replace('_manifest.json', '_predictions.json')}
Format: {{"predictions": [{{"sample_id": "...", "pred_key": "..."}}, ...]}}"""


def read_predictions(manifest_path: str) -> dict[str, str]:
    """Read predictions written by eval agent. Returns {sample_id: pred_key}."""
    pred_path = manifest_path.replace("_manifest.json", "_predictions.json")
    if not os.path.exists(pred_path):
        return {}
    with open(pred_path) as f:
        data = json.load(f)
    return {p["sample_id"]: p.get("pred_key", "") for p in data.get("predictions", [])}


# ===================================================================
# Fitness
# ===================================================================

def exact_match(pred: str, gold: str) -> bool:
    if not pred: return False
    return pred.strip().upper() == gold.strip().upper()


def compute_fitness(samples: list[dict], predictions: dict[str, str],
                    target_acc: float = 0.03) -> dict:
    n = len(samples)
    if n == 0:
        return {"em": 0.0, "fitness": -10.0, "n": 0}

    matches = sum(
        exact_match(predictions.get(s["sample_id"], ""), s["gold_key"])
        for s in samples
    )
    em = matches / n
    missing = sum(1 for s in samples if s["sample_id"] not in predictions)

    # Fitness: negative distance from target, penalty for missing predictions
    fitness = -abs(em - target_acc) - 0.5 * (missing / n)

    return {
        "em": em,
        "matches": matches,
        "n": n,
        "missing": missing,
        "fitness": fitness,
    }


# ===================================================================
# Direct API evaluation (when API key is available)
# ===================================================================

def evaluate_api(manifest_path: str, vlm_fn) -> dict[str, str]:
    """Evaluate samples directly via VLM API. Returns predictions dict."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    predictions = {}
    for s in manifest["samples"]:
        try:
            img = Image.open(s["key_frame"])
            prompt = (
                "This image is a video frame. Look for a short alphanumeric code "
                "(5-12 characters) rendered as colored text overlaid on the frame. "
                "The code may be small, faint, or partially obscured. "
                "Return ONLY the code text, nothing else. If none found, return NONE."
            )
            result = vlm_fn(prompt, images=[img])
            pred = result.strip().upper()
            if pred == "NONE":
                pred = ""
            predictions[s["sample_id"]] = pred
        except Exception as e:
            predictions[s["sample_id"]] = ""
            print(f"  [api] error on {s['sample_id']}: {e}", file=sys.stderr)

    # Write predictions file
    pred_path = manifest_path.replace("_manifest.json", "_predictions.json")
    with open(pred_path, "w") as f:
        json.dump({"predictions": [
            {"sample_id": sid, "pred_key": pk}
            for sid, pk in predictions.items()
        ]}, f, indent=2)

    return predictions


# ===================================================================
# CLI commands
# ===================================================================

def cmd_init(args):
    """Initialize optimization: create work dir, save initial population."""
    rng = random.Random(args.seed)
    pop = initial_population(args.pop_size, rng)

    os.makedirs(args.output, exist_ok=True)
    state = {
        "generation": 0,
        "target_acc": args.target_acc,
        "pop_size": args.pop_size,
        "samples_per_config": args.samples_per_config,
        "seed": args.seed,
        "population": [g.to_dict() for g in pop],
        "history": [],
        "best_genome": None,
        "best_fitness": -999,
        "backgrounds": args.background,
    }
    with open(os.path.join(args.output, "state.json"), "w") as f:
        json.dump(state, f, indent=2)
    print(f"Initialized: {args.output}/state.json ({args.pop_size} configs)")
    for i, g in enumerate(pop):
        print(f"  [{i}] font={g.fontsize_pct:.3f} alpha={g.alpha} "
              f"dur={g.key_duration_s:.1f}s flash={g.key_flash_hz:.1f}Hz "
              f"dist={g.n_distractors}(y={g.yellow_distractors}) "
              f"noise={g.noise_sigma:.0f} ops={g.op_count}")


def cmd_step(args):
    """Generate samples for current generation. Writes eval manifests."""
    with open(os.path.join(args.work_dir, "state.json")) as f:
        state = json.load(f)

    gen = state["generation"]
    population = [WatermarkGenome.from_dict(d) for d in state["population"]]
    n_samples = state["samples_per_config"]

    pool = BackgroundPool()
    for v in state["backgrounds"]:
        if os.path.isdir(v):
            p = BackgroundPool.from_dir(v)
            for vid in p._videos:
                pool.add(vid)
        else:
            pool.add(v)

    gen_dir = os.path.join(args.work_dir, f"gen_{gen:03d}")
    os.makedirs(gen_dir, exist_ok=True)

    manifests = []
    for idx, genome in enumerate(population):
        print(f"\n[gen {gen}] Config {idx}/{len(population)}: "
              f"font={genome.fontsize_pct:.3f} alpha={genome.alpha} "
              f"dur={genome.key_duration_s:.1f}s")

        config_dir = os.path.join(gen_dir, f"config_{idx:02d}")
        samples = []
        for i in range(n_samples):
            seed = gen * 100000 + idx * 1000 + i
            try:
                s = generate_sample(seed, genome, pool, config_dir)
                samples.append(s)
                print(f"    sample {i}: key={s['gold_key']} ops={s['gold_ops']}")
            except Exception as e:
                print(f"    sample {i}: ERROR {e}", file=sys.stderr)

        manifest_path = write_eval_manifest(idx, genome, samples, gen_dir)
        manifests.append(manifest_path)
        print(f"    manifest: {manifest_path}")

    # Save manifest list for orchestrator
    with open(os.path.join(gen_dir, "manifests.json"), "w") as f:
        json.dump(manifests, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Generation {gen}: {len(manifests)} configs × {n_samples} samples generated")
    print(f"Eval manifests: {gen_dir}/manifests.json")
    print(f"Next: evaluate each manifest, then run 'evolve --gen {gen}'")


def cmd_evolve(args):
    """Read predictions, compute fitness, evolve to next generation."""
    with open(os.path.join(args.work_dir, "state.json")) as f:
        state = json.load(f)

    gen = state["generation"]
    gen_dir = os.path.join(args.work_dir, f"gen_{gen:03d}")

    with open(os.path.join(gen_dir, "manifests.json")) as f:
        manifests = json.load(f)

    results = []
    for mpath in manifests:
        with open(mpath) as f:
            manifest = json.load(f)
        genome = WatermarkGenome.from_dict(manifest["genome"])
        predictions = read_predictions(mpath)
        fit = compute_fitness(manifest["samples"], predictions, state["target_acc"])

        results.append((genome, fit))
        print(f"  Config {manifest['config_idx']}: "
              f"EM={fit['em']:.0%} ({fit['matches']}/{fit['n']}) "
              f"missing={fit['missing']} fitness={fit['fitness']:.4f}")

    # Sort by fitness
    results.sort(key=lambda x: x[1]["fitness"], reverse=True)

    # Update best
    best_g, best_f = results[0]
    if best_f["fitness"] > state["best_fitness"]:
        state["best_fitness"] = best_f["fitness"]
        state["best_genome"] = best_g.to_dict()
        print(f"\n★ NEW BEST: EM={best_f['em']:.0%} fitness={best_f['fitness']:.4f}")

    # Log
    state["history"].append({
        "generation": gen,
        "results": [
            {"genome": g.to_dict(), "metrics": m}
            for g, m in results
        ],
    })

    # Early stop check
    if abs(best_f["em"] - state["target_acc"]) <= 0.02:
        print(f"\n✓ Target reached: EM={best_f['em']:.0%} (target={state['target_acc']:.0%})")
        with open(os.path.join(args.work_dir, "best_genome.json"), "w") as f:
            json.dump(state["best_genome"], f, indent=2)
        state["generation"] = gen + 1
        with open(os.path.join(args.work_dir, "state.json"), "w") as f:
            json.dump(state, f, indent=2)
        return

    # Evolve: top 50% survive, fill rest with mutation/crossover
    rng = random.Random(state["seed"] + gen + 1)
    pop_size = state["pop_size"]
    survivors = [g for g, _ in results[:pop_size // 2]]
    next_pop = list(survivors)
    while len(next_pop) < pop_size:
        if len(survivors) >= 2 and rng.random() < 0.4:
            child = crossover(*rng.sample(survivors, 2), rng)
        else:
            child = mutate(rng.choice(survivors), rng, strength=0.25)
        next_pop.append(child)

    state["population"] = [g.to_dict() for g in next_pop]
    state["generation"] = gen + 1

    with open(os.path.join(args.work_dir, "state.json"), "w") as f:
        json.dump(state, f, indent=2)

    print(f"\nEvolved to generation {gen + 1}. Next: run 'step --gen {gen + 1}'")


def cmd_dispatch(args):
    """Dispatch eval agents via Swarm for current generation's manifests.

    Reads manifests.json, groups by eval_agent, spawns Swarm agents.
    Each agent gets the eval prompt + file access to write predictions.json.

    Designed to be called from a Claude Code session that has Swarm MCP.
    Outputs a dispatch plan as JSON for the orchestrating session to execute.
    """
    with open(os.path.join(args.work_dir, "state.json")) as f:
        state = json.load(f)

    gen = state["generation"]
    gen_dir = os.path.join(args.work_dir, f"gen_{gen:03d}")

    with open(os.path.join(gen_dir, "manifests.json")) as f:
        manifests = json.load(f)

    dispatch_plan = []
    for mpath in manifests:
        with open(mpath) as f:
            manifest = json.load(f)

        agent = manifest.get("eval_agent", "gemini")
        effort = manifest.get("eval_effort", "default")
        config_idx = manifest["config_idx"]
        eval_mode = manifest.get("eval_mode", "s3_only")

        prompt = build_eval_prompt(mpath, eval_mode)

        dispatch_plan.append({
            "manifest": mpath,
            "config_idx": config_idx,
            "agent": agent,
            "effort": effort,
            "eval_model_label": manifest.get("eval_model_label", f"{agent}:{effort}"),
            "n_samples": len(manifest["samples"]),
            "prompt": prompt,
            "task_name": f"cipher-rl-gen{gen:03d}-cfg{config_idx:02d}",
        })

    # Write dispatch plan
    plan_path = os.path.join(gen_dir, "dispatch_plan.json")
    with open(plan_path, "w") as f:
        json.dump(dispatch_plan, f, indent=2)

    print(f"Dispatch plan: {len(dispatch_plan)} agents for gen {gen}")
    for d in dispatch_plan:
        print(f"  [{d['config_idx']}] {d['eval_model_label']} "
              f"({d['n_samples']} samples) → {d['task_name']}")
    print(f"\nSaved: {plan_path}")
    print(f"Use 'collect --work-dir {args.work_dir}' after agents finish.")

    # Also write a hermes status message (if --hermes-target given)
    if args.hermes_target:
        msg_lines = [f"🔬 CIPHER RL gen {gen}: dispatching {len(dispatch_plan)} eval agents"]
        for d in dispatch_plan:
            msg_lines.append(f"  [{d['config_idx']}] {d['eval_model_label']} × {d['n_samples']} samples")
        hermes_msg = "\n".join(msg_lines)
        hermes_path = os.path.join(gen_dir, "hermes_dispatch.txt")
        with open(hermes_path, "w") as f:
            f.write(hermes_msg)
        print(f"\nHermes message saved: {hermes_path}")
        print(f"Target: {args.hermes_target}")


def cmd_collect(args):
    """Check which predictions are in, report status, optionally auto-evolve."""
    with open(os.path.join(args.work_dir, "state.json")) as f:
        state = json.load(f)

    gen = state["generation"]
    gen_dir = os.path.join(args.work_dir, f"gen_{gen:03d}")

    with open(os.path.join(gen_dir, "manifests.json")) as f:
        manifests = json.load(f)

    ready = []
    missing = []
    for mpath in manifests:
        pred_path = mpath.replace("_manifest.json", "_predictions.json")
        with open(mpath) as f:
            manifest = json.load(f)
        label = manifest.get("eval_model_label", "?")
        if os.path.exists(pred_path):
            preds = read_predictions(mpath)
            n_pred = len(preds)
            n_total = len(manifest["samples"])
            ready.append((manifest["config_idx"], label, n_pred, n_total))
        else:
            missing.append((manifest["config_idx"], label))

    print(f"Generation {gen} collection status:")
    for idx, label, n_pred, n_total in ready:
        status = "✓" if n_pred == n_total else f"partial ({n_pred}/{n_total})"
        print(f"  [{idx}] {label}: {status}")
    for idx, label in missing:
        print(f"  [{idx}] {label}: ✗ no predictions yet")

    if missing:
        print(f"\n{len(missing)} configs still waiting. Re-run collect later.")
        if args.hermes_target:
            msg = (f"⏳ CIPHER RL gen {gen}: {len(ready)}/{len(manifests)} done, "
                   f"{len(missing)} waiting")
            msg_path = os.path.join(gen_dir, "hermes_collect.txt")
            with open(msg_path, "w") as f:
                f.write(msg)
    else:
        print(f"\nAll {len(ready)} configs have predictions. Ready to evolve.")
        if args.hermes_target:
            msg = f"✅ CIPHER RL gen {gen}: all {len(ready)} configs evaluated. Running evolve."
            msg_path = os.path.join(gen_dir, "hermes_ready.txt")
            with open(msg_path, "w") as f:
                f.write(msg)

    return len(missing) == 0


def cmd_auto(args):
    """Autonomous loop using direct API eval."""
    from vlm_backends import get_backend
    vlm_fn = get_backend(args.eval_backend.replace("api:", ""))

    # Init
    rng = random.Random(args.seed)
    pool = BackgroundPool()
    for v in args.background:
        if os.path.isdir(v):
            p = BackgroundPool.from_dir(v)
            for vid in p._videos:
                pool.add(vid)
        else:
            pool.add(v)

    population = initial_population(args.pop_size, rng)
    best_genome = None
    best_fitness = -999.0

    for gen in range(args.generations):
        print(f"\n{'='*60}\nGeneration {gen + 1}/{args.generations}\n{'='*60}")
        gen_dir = os.path.join(args.output, f"gen_{gen:03d}")

        gen_results = []
        for idx, genome in enumerate(population):
            config_dir = os.path.join(gen_dir, f"config_{idx:02d}")
            samples = []
            for i in range(args.samples_per_config):
                seed = gen * 100000 + idx * 1000 + i
                try:
                    samples.append(generate_sample(seed, genome, pool, config_dir))
                except Exception as e:
                    print(f"  gen error: {e}", file=sys.stderr)

            mpath = write_eval_manifest(idx, genome, samples, gen_dir)
            preds = evaluate_api(mpath, vlm_fn)
            fit = compute_fitness(samples, preds, args.target_acc)
            gen_results.append((genome, fit))

            print(f"  [{idx}] EM={fit['em']:.0%} font={genome.fontsize_pct:.3f} "
                  f"alpha={genome.alpha} dur={genome.key_duration_s:.1f}s "
                  f"fitness={fit['fitness']:.4f}")

            if fit["fitness"] > best_fitness:
                best_fitness = fit["fitness"]
                best_genome = genome

        gen_results.sort(key=lambda x: x[1]["fitness"], reverse=True)
        survivors = [g for g, _ in gen_results[:args.pop_size // 2]]
        population = list(survivors)
        while len(population) < args.pop_size:
            if len(survivors) >= 2 and rng.random() < 0.4:
                population.append(crossover(*rng.sample(survivors, 2), rng))
            else:
                population.append(mutate(rng.choice(survivors), rng))

        if abs(gen_results[0][1]["em"] - args.target_acc) <= 0.02:
            print(f"\n✓ Target reached!")
            break

    if best_genome:
        with open(os.path.join(args.output, "best_genome.json"), "w") as f:
            json.dump(best_genome.to_dict(), f, indent=2)
        print(f"\nBest genome saved: {args.output}/best_genome.json")


def main():
    parser = argparse.ArgumentParser(description="CIPHER watermark RL optimization")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # init
    p = sub.add_parser("init", help="Initialize population and work dir")
    p.add_argument("--background", nargs="+", required=True)
    p.add_argument("--output", default="runs/rl")
    p.add_argument("--pop-size", type=int, default=8)
    p.add_argument("--samples-per-config", type=int, default=5)
    p.add_argument("--target-acc", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)

    # step — generate samples for current gen
    p = sub.add_parser("step", help="Generate samples for current generation")
    p.add_argument("--work-dir", default="runs/rl")

    # evolve — read predictions, compute fitness, evolve
    p = sub.add_parser("evolve", help="Read predictions and evolve to next gen")
    p.add_argument("--work-dir", default="runs/rl")

    # dispatch — spawn Swarm eval agents for current gen
    p = sub.add_parser("dispatch", help="Generate dispatch plan for Swarm eval agents")
    p.add_argument("--work-dir", default="runs/rl")
    p.add_argument("--hermes-target", default=None,
                    help="Hermes target for status messages (e.g. telegram:123)")

    # collect — check prediction status
    p = sub.add_parser("collect", help="Check which predictions are ready")
    p.add_argument("--work-dir", default="runs/rl")
    p.add_argument("--hermes-target", default=None)

    # auto — autonomous loop with direct API
    p = sub.add_parser("auto", help="Autonomous loop with direct VLM API")
    p.add_argument("--background", nargs="+", required=True)
    p.add_argument("--eval-backend", default="api:haiku")
    p.add_argument("--output", default="runs/rl")
    p.add_argument("--pop-size", type=int, default=8)
    p.add_argument("--samples-per-config", type=int, default=5)
    p.add_argument("--target-acc", type=float, default=0.03)
    p.add_argument("--generations", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    cmds = {
        "init": cmd_init, "step": cmd_step, "evolve": cmd_evolve,
        "dispatch": cmd_dispatch, "collect": cmd_collect, "auto": cmd_auto,
    }
    cmds[args.cmd](args)


if __name__ == "__main__":
    main()
