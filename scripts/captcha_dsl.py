"""
captcha_dsl.py — S-expression DSL for captcha program evolution.

Programs are lisp-like trees that describe how to construct a captcha sample.
GEPA evolves the program text; Nevergrad tunes numeric params within it.

Primitives:
  (seq EXPR ...)              — execute expressions in order
  (overlay-key K V ...)       — place key text; params: alpha, font-pct, dur, position, flash-hz
  (scramble OP ...)           — apply ffmpeg ops to video
  (distractor K V ...)        — add visual distractor; params: color, alpha, same-length, yellow
  (gate CAPTCHA ON-FAIL)      — sub-captcha gate (must solve inner to proceed)
  (text-captcha EXPR)         — text puzzle: "7*8-3=?", "reverse('hello')=?"
  (ocr-captcha K V ...)       — overlaid text that must be typed (classic CAPTCHA)

Ops (used inside scramble):
  reverse, hflip, vflip, rotate90, rotate270, speed2x

Example program:
  (seq
    (overlay-key :alpha 140 :font-pct 0.04 :dur 1.5 :position edge :flash-hz 3.0)
    (scramble reverse hflip speed2x)
    (distractor :n 4 :yellow 2 :same-length true)
    (gate (text-captcha "compute: 7*8-3") :on-fail blur-key))

Numeric params are tagged with $ prefix for Nevergrad:
  (overlay-key :alpha $alpha :font-pct $font_pct :dur $dur)
Nevergrad sees: {"alpha": (30, 255), "font_pct": (0.015, 0.15), "dur": (0.1, 5.0)}
"""
import re
from dataclasses import dataclass, field
from typing import Any


# ===================================================================
# S-expression parser
# ===================================================================

class ParseError(Exception):
    pass


def tokenize(s: str) -> list[str]:
    """Tokenize s-expression string into tokens."""
    # Add spaces around parens, handle quoted strings
    s = s.replace("(", " ( ").replace(")", " ) ")
    tokens = []
    i = 0
    while i < len(s):
        if s[i].isspace():
            i += 1
        elif s[i] == '"':
            # Quoted string
            j = s.index('"', i + 1)
            tokens.append(s[i:j+1])
            i = j + 1
        else:
            j = i
            while j < len(s) and not s[j].isspace():
                j += 1
            tokens.append(s[i:j])
            i = j
    return tokens


def parse(tokens: list[str], pos: int = 0) -> tuple[Any, int]:
    """Parse tokens into nested list structure."""
    if pos >= len(tokens):
        raise ParseError("unexpected end of input")

    tok = tokens[pos]
    if tok == "(":
        lst = []
        pos += 1
        while pos < len(tokens) and tokens[pos] != ")":
            val, pos = parse(tokens, pos)
            lst.append(val)
        if pos >= len(tokens):
            raise ParseError("missing closing )")
        return lst, pos + 1  # skip )
    elif tok == ")":
        raise ParseError("unexpected )")
    else:
        # Atom: try number, then keyword, then symbol
        if tok.startswith('"') and tok.endswith('"'):
            return tok[1:-1], pos + 1
        try:
            return int(tok), pos + 1
        except ValueError:
            pass
        try:
            return float(tok), pos + 1
        except ValueError:
            pass
        # Boolean
        if tok == "true":
            return True, pos + 1
        if tok == "false":
            return False, pos + 1
        return tok, pos + 1


def parse_program(s: str) -> list:
    """Parse a program string into an AST (nested lists)."""
    tokens = tokenize(s)
    if not tokens:
        raise ParseError("empty program")
    ast, pos = parse(tokens, 0)
    return ast


def unparse(ast) -> str:
    """Convert AST back to s-expression string."""
    if isinstance(ast, list):
        return "(" + " ".join(unparse(x) for x in ast) + ")"
    elif isinstance(ast, bool):
        return "true" if ast else "false"
    elif isinstance(ast, str):
        if " " in ast or "(" in ast or ")" in ast:
            return f'"{ast}"'
        return ast
    else:
        return str(ast)


# ===================================================================
# Program → CaptchaSpec compilation
# ===================================================================

@dataclass
class CaptchaSpec:
    """Compiled captcha specification — flat structure for generator."""
    # Key overlay
    alpha: int = 180
    fontsize_pct: float = 0.06
    key_length: int = 7
    key_duration_s: float = 2.0
    position_mode: str = "random"
    key_flash_hz: float = 0.0
    font_variant: int = 0

    # Scramble ops
    ops: list[str] = field(default_factory=lambda: ["reverse"])

    # Distractors
    n_distractors: int = 0
    yellow_distractors: int = 0
    distractor_same_length: bool = False

    # Visual perturbation
    noise_sigma: float = 0.0
    blur_radius: float = 0.0

    # Shadow
    shadow: bool = True

    # Sub-captcha gates
    gates: list[dict] = field(default_factory=list)

    # NL ambiguity (1=exact, 2=paraphrase, 3=reversed)
    nl_ambiguity: int = 1

    # Nevergrad variable references (param_name → (lo, hi))
    ng_params: dict = field(default_factory=dict)


# Param ranges for Nevergrad variables
NG_PARAM_RANGES = {
    "alpha":       (25, 255),
    "font_pct":    (0.015, 0.15),
    "dur":         (0.08, 5.0),
    "flash_hz":    (0.0, 10.0),
    "noise":       (0.0, 40.0),
    "blur":        (0.0, 3.0),
    "jitter":      (0.0, 25.0),
    "n_dist":      (0, 10),
    "n_yellow":    (0, 6),
    "key_len":     (5, 12),
    "nl_ambig":    (1, 3),
}

VALID_OPS = {"reverse", "hflip", "vflip", "rotate90", "rotate270", "speed2x"}
VALID_POSITIONS = {"center", "random", "corner", "edge"}


def _extract_kwargs(args: list) -> dict:
    """Extract :key value pairs from args list."""
    kwargs = {}
    i = 0
    while i < len(args):
        if isinstance(args[i], str) and args[i].startswith(":"):
            key = args[i][1:]  # strip :
            if i + 1 < len(args):
                kwargs[key] = args[i + 1]
                i += 2
            else:
                i += 1
        else:
            i += 1
    return kwargs


def compile_node(node, spec: CaptchaSpec):
    """Compile one AST node into the CaptchaSpec."""
    if not isinstance(node, list) or len(node) == 0:
        return

    head = node[0]

    if head == "seq":
        for child in node[1:]:
            compile_node(child, spec)

    elif head == "overlay-key":
        kw = _extract_kwargs(node[1:])
        if "alpha" in kw:
            v = kw["alpha"]
            if isinstance(v, str) and v.startswith("$"):
                spec.ng_params[v[1:]] = NG_PARAM_RANGES.get(v[1:], (25, 255))
            else:
                spec.alpha = int(v)
        if "font-pct" in kw:
            v = kw["font-pct"]
            if isinstance(v, str) and v.startswith("$"):
                spec.ng_params[v[1:]] = NG_PARAM_RANGES.get(v[1:], (0.015, 0.15))
            else:
                spec.fontsize_pct = float(v)
        if "dur" in kw:
            v = kw["dur"]
            if isinstance(v, str) and v.startswith("$"):
                spec.ng_params[v[1:]] = NG_PARAM_RANGES.get(v[1:], (0.08, 5.0))
            else:
                spec.key_duration_s = float(v)
        if "position" in kw:
            p = kw["position"]
            if p in VALID_POSITIONS:
                spec.position_mode = p
        if "flash-hz" in kw:
            v = kw["flash-hz"]
            if isinstance(v, str) and v.startswith("$"):
                spec.ng_params[v[1:]] = NG_PARAM_RANGES.get(v[1:], (0.0, 10.0))
            else:
                spec.key_flash_hz = float(v)
        if "key-len" in kw:
            v = kw["key-len"]
            if isinstance(v, str) and v.startswith("$"):
                spec.ng_params[v[1:]] = NG_PARAM_RANGES.get(v[1:], (5, 12))
            else:
                spec.key_length = int(v)
        if "font" in kw:
            spec.font_variant = int(kw["font"])

    elif head == "scramble":
        ops = [x for x in node[1:] if isinstance(x, str) and x in VALID_OPS]
        if ops:
            spec.ops = ops

    elif head == "distractor":
        kw = _extract_kwargs(node[1:])
        if "n" in kw:
            v = kw["n"]
            if isinstance(v, str) and v.startswith("$"):
                spec.ng_params[v[1:]] = NG_PARAM_RANGES.get(v[1:], (0, 10))
            else:
                spec.n_distractors = int(v)
        if "yellow" in kw:
            v = kw["yellow"]
            if isinstance(v, str) and v.startswith("$"):
                spec.ng_params[v[1:]] = NG_PARAM_RANGES.get(v[1:], (0, 6))
            else:
                spec.yellow_distractors = int(v)
        if "same-length" in kw:
            spec.distractor_same_length = bool(kw["same-length"])

    elif head == "noise":
        kw = _extract_kwargs(node[1:])
        if "sigma" in kw:
            v = kw["sigma"]
            if isinstance(v, str) and v.startswith("$"):
                spec.ng_params[v[1:]] = NG_PARAM_RANGES.get(v[1:], (0.0, 40.0))
            else:
                spec.noise_sigma = float(v)

    elif head == "blur":
        kw = _extract_kwargs(node[1:])
        if "radius" in kw:
            v = kw["radius"]
            if isinstance(v, str) and v.startswith("$"):
                spec.ng_params[v[1:]] = NG_PARAM_RANGES.get(v[1:], (0.0, 3.0))
            else:
                spec.blur_radius = float(v)

    elif head == "gate":
        gate = {"type": "unknown"}
        if len(node) > 1:
            sub = node[1]
            if isinstance(sub, list) and len(sub) > 0:
                if sub[0] == "text-captcha":
                    expr = sub[1] if len(sub) > 1 else "2+2=?"
                    gate = {"type": "text-captcha", "expr": str(expr)}
                elif sub[0] == "ocr-captcha":
                    kw = _extract_kwargs(sub[1:])
                    gate = {"type": "ocr-captcha", **kw}
        kw = _extract_kwargs(node[1:])
        if "on-fail" in kw:
            gate["on_fail"] = kw["on-fail"]
        spec.gates.append(gate)

    elif head == "nl-ambiguity":
        if len(node) > 1:
            v = node[1]
            if isinstance(v, str) and v.startswith("$"):
                spec.ng_params[v[1:]] = NG_PARAM_RANGES.get(v[1:], (1, 3))
            else:
                spec.nl_ambiguity = int(v)

    elif head == "shadow":
        spec.shadow = True if len(node) < 2 else bool(node[1])


def compile_program(program_str: str) -> CaptchaSpec:
    """Parse and compile a program string into a CaptchaSpec."""
    ast = parse_program(program_str)
    spec = CaptchaSpec()
    compile_node(ast, spec)
    return spec


def spec_to_genome_dict(spec: CaptchaSpec, ng_values: dict | None = None) -> dict:
    """Convert CaptchaSpec to a WatermarkGenome-compatible dict.

    ng_values: resolved Nevergrad values for $-params.
    """
    d = {
        "fontsize_pct": spec.fontsize_pct,
        "alpha": spec.alpha,
        "key_length": spec.key_length,
        "key_duration_s": spec.key_duration_s,
        "position_mode": spec.position_mode,
        "key_flash_hz": spec.key_flash_hz,
        "font_variant": spec.font_variant,
        "n_distractors": spec.n_distractors,
        "yellow_distractors": spec.yellow_distractors,
        "distractor_same_length": spec.distractor_same_length,
        "noise_sigma": spec.noise_sigma,
        "blur_radius": spec.blur_radius,
        "shadow": spec.shadow,
        "nl_ambiguity": spec.nl_ambiguity,
        "op_count": len(spec.ops),
        "ops": spec.ops,
        "gates": spec.gates,
    }

    # Override with Nevergrad-resolved values
    if ng_values:
        param_map = {
            "alpha": ("alpha", int),
            "font_pct": ("fontsize_pct", float),
            "dur": ("key_duration_s", float),
            "flash_hz": ("key_flash_hz", float),
            "noise": ("noise_sigma", float),
            "blur": ("blur_radius", float),
            "n_dist": ("n_distractors", int),
            "n_yellow": ("yellow_distractors", int),
            "key_len": ("key_length", int),
            "nl_ambig": ("nl_ambiguity", int),
        }
        for ng_name, val in ng_values.items():
            if ng_name in param_map:
                field_name, cast = param_map[ng_name]
                d[field_name] = cast(val)

    return d


# ===================================================================
# Seed programs (initial population for GEPA)
# ===================================================================

SEED_PROGRAMS = [
    # Easy baseline
    """(seq
  (overlay-key :alpha 220 :font-pct 0.10 :dur 3.0 :position center)
  (scramble reverse)
  (shadow true))""",

    # Medium — more ops, some distractors
    """(seq
  (overlay-key :alpha 160 :font-pct 0.06 :dur 2.0 :position random :flash-hz 2.0)
  (scramble reverse hflip)
  (distractor :n 3 :yellow 1)
  (noise :sigma 10)
  (shadow true))""",

    # Hard — small text, many distractors, flashing
    """(seq
  (overlay-key :alpha $alpha :font-pct $font_pct :dur $dur :position edge :flash-hz $flash_hz)
  (scramble reverse hflip speed2x)
  (distractor :n $n_dist :yellow $n_yellow :same-length true)
  (noise :sigma $noise)
  (blur :radius $blur)
  (nl-ambiguity 3))""",

    # With text sub-captcha gate
    """(seq
  (overlay-key :alpha 140 :font-pct 0.05 :dur 1.5 :position corner)
  (scramble reverse vflip)
  (gate (text-captcha "compute: 7*8-3") :on-fail blur-key)
  (distractor :n 2 :yellow 1))""",

    # Max difficulty — all knobs
    """(seq
  (overlay-key :alpha $alpha :font-pct $font_pct :dur $dur :position edge :flash-hz $flash_hz :key-len 10)
  (scramble reverse hflip vflip speed2x)
  (distractor :n $n_dist :yellow $n_yellow :same-length true)
  (gate (text-captcha "reverse('cipher')=?") :on-fail add-distractor)
  (noise :sigma $noise)
  (blur :radius $blur)
  (nl-ambiguity $nl_ambig))""",
]


# ===================================================================
# Image-mode generator (no ffmpeg, pure PIL — 50x faster)
# ===================================================================

import random
import string
import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter


def _get_font_path(variant: int) -> str | None:
    paths = {
        0: ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
            "/System/Library/Fonts/Menlo.ttc"],
        1: ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/System/Library/Fonts/Monaco.ttf"],
        2: ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc"],
        3: ["/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "/System/Library/Fonts/Times.ttc"],
    }
    for p in paths.get(variant, paths[0]):
        if Path(p).exists():
            return p
    return None


def _load_font(variant: int, size: int):
    fp = _get_font_path(variant)
    if fp:
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _random_key(rng: random.Random, length: int = 7) -> str:
    chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no ambiguous 0/O/I/1
    return "".join(rng.choices(chars, k=length))


def _pick_position(mode: str, w: int, h: int, tw: int, th: int,
                   rng: random.Random) -> tuple[int, int]:
    if mode == "center":
        x, y = int(w/2 - tw/2), int(h/2 - th/2)
    elif mode == "corner":
        cx, cy = rng.choice([(0.05, 0.05), (0.05, 0.85), (0.85, 0.05), (0.85, 0.85)])
        x, y = int(cx * w), int(cy * h)
    elif mode == "edge":
        edge = rng.choice(["top", "bottom", "left", "right"])
        if edge == "top":    x, y = int(rng.uniform(0.1, 0.7) * w), int(0.03 * h)
        elif edge == "bottom": x, y = int(rng.uniform(0.1, 0.7) * w), int(0.90 * h)
        elif edge == "left":  x, y = int(0.03 * w), int(rng.uniform(0.1, 0.7) * h)
        else:                 x, y = int(0.85 * w), int(rng.uniform(0.1, 0.7) * h)
    else:  # random
        x, y = int(rng.uniform(0.05, 0.80) * w), int(rng.uniform(0.05, 0.80) * h)
    x = max(2, min(x, w - tw - 2))
    y = max(2, min(y, h - th - 2))
    return x, y


def generate_image_sample(
    seed: int,
    genome_dict: dict,
    bg_image: Image.Image | str | None = None,
    output_dir: str | None = None,
    frame_size: tuple[int, int] = (640, 360),
) -> dict:
    """Generate a single image captcha sample. No ffmpeg, pure PIL.

    Args:
        seed: deterministic seed
        genome_dict: from spec_to_genome_dict (or WatermarkGenome-compatible)
        bg_image: background PIL image, path, or None (random noise bg)
        output_dir: where to save the image (or None for in-memory)
        frame_size: (width, height) if no bg_image

    Returns:
        dict with: sample_id, gold_key, image (PIL), image_path (if saved),
                   distractors, gates
    """
    rng = random.Random(seed)
    sample_id = f"img_s{seed:06d}"

    # Background
    if bg_image is None:
        # Generate a noisy background
        w, h = frame_size
        bg_arr = np.random.RandomState(seed).randint(20, 235, (h, w, 3), dtype=np.uint8)
        bg = Image.fromarray(bg_arr, "RGB")
    elif isinstance(bg_image, str):
        bg = Image.open(bg_image).convert("RGB")
        w, h = bg.size
    else:
        bg = bg_image.convert("RGB")
        w, h = bg.size

    # Params
    alpha = int(genome_dict.get("alpha", 180))
    fontsize_pct = float(genome_dict.get("fontsize_pct", 0.06))
    key_length = int(genome_dict.get("key_length", 7))
    font_variant = int(genome_dict.get("font_variant", 0))
    position_mode = genome_dict.get("position_mode", "random")
    n_distractors = int(genome_dict.get("n_distractors", 0))
    yellow_distractors = int(genome_dict.get("yellow_distractors", 0))
    distractor_same_length = bool(genome_dict.get("distractor_same_length", False))
    noise_sigma = float(genome_dict.get("noise_sigma", 0.0))
    blur_radius = float(genome_dict.get("blur_radius", 0.0))
    shadow = bool(genome_dict.get("shadow", True))

    # Generate key
    key = _random_key(rng, key_length)

    # Font
    fontsize = max(8, int(h * fontsize_pct))
    font = _load_font(font_variant, fontsize)

    # Create key overlay
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        bbox = draw.textbbox((0, 0), key, font=font)
        tw, th_ = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th_ = int(fontsize * len(key) * 0.6), fontsize

    x, y = _pick_position(position_mode, w, h, int(tw), int(th_), rng)

    # Shadow
    if shadow:
        draw.text((x+2, y+2), key, fill=(0, 0, 0, min(255, alpha)), font=font)

    # Key text (yellow)
    draw.text((x, y), key, fill=(255, 255, 0, alpha), font=font)

    # Blur overlay
    if blur_radius > 0:
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Noise on overlay
    if noise_sigma > 0:
        arr = np.array(overlay).astype(np.float32)
        noise = np.random.RandomState(seed + 1).normal(0, noise_sigma, arr.shape[:2] + (3,))
        arr[:, :, :3] = np.clip(arr[:, :, :3] + noise, 0, 255)
        overlay = Image.fromarray(arr.astype(np.uint8))

    # Add distractors
    palette = [(0, 220, 255), (80, 255, 80), (255, 100, 255), (255, 180, 50), (180, 120, 255)]
    d_font = _load_font(font_variant, max(8, int(fontsize * 0.75)))
    d_alpha_max = max(60, alpha - 40)
    distractor_info = []

    for d_idx in range(n_distractors):
        d_len = key_length if distractor_same_length else rng.randint(4, 6)
        fake_key = _random_key(rng, d_len)

        if d_idx < yellow_distractors:
            d_color = (255, 255, 0)
        else:
            d_color = palette[d_idx % len(palette)]
        d_alpha = rng.randint(max(40, d_alpha_max - 40), d_alpha_max)

        dx, dy = int(rng.uniform(0.05, 0.80) * w), int(rng.uniform(0.05, 0.80) * h)
        draw_d = ImageDraw.Draw(overlay)
        draw_d.text((dx+2, dy+2), fake_key, fill=(0, 0, 0, d_alpha), font=d_font)
        draw_d.text((dx, dy), fake_key, fill=(*d_color, d_alpha), font=d_font)
        distractor_info.append({"text": fake_key, "color": d_color, "pos": (dx, dy)})

    # Composite: bg + overlay
    bg_rgba = bg.convert("RGBA")
    composite = Image.alpha_composite(bg_rgba, overlay).convert("RGB")

    # Build result
    result = {
        "sample_id": sample_id,
        "seed": seed,
        "gold_key": key,
        "image": composite,
        "key_position": (x, y),
        "frame_size": (w, h),
        "n_distractors": n_distractors,
        "gates": genome_dict.get("gates", []),
    }

    # Save if output_dir given
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        img_path = os.path.join(output_dir, f"{sample_id}.jpg")
        composite.save(img_path, quality=90)
        result["image_path"] = os.path.abspath(img_path)

    return result


def load_bg_images(bg_dir: str, max_images: int = 50) -> list[Image.Image]:
    """Load background images from a directory (or extract frames from videos)."""
    images = []
    if not os.path.isdir(bg_dir):
        return images

    for f in sorted(os.listdir(bg_dir)):
        fp = os.path.join(bg_dir, f)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            try:
                images.append(Image.open(fp).convert("RGB"))
            except Exception:
                pass
        elif f.lower().endswith((".mp4", ".webm", ".mkv")):
            # Extract a single frame from video via ffmpeg
            import subprocess, tempfile
            tmp = tempfile.mktemp(suffix=".jpg")
            subprocess.run(
                ["ffmpeg", "-y", "-ss", "5", "-i", fp, "-frames:v", "1",
                 "-q:v", "2", tmp],
                capture_output=True)
            if os.path.exists(tmp):
                try:
                    images.append(Image.open(tmp).convert("RGB"))
                except Exception:
                    pass
                os.unlink(tmp)

        if len(images) >= max_images:
            break

    return images


# ===================================================================
# CLI self-test
# ===================================================================

if __name__ == "__main__":
    for i, prog in enumerate(SEED_PROGRAMS):
        print(f"\n{'='*60}")
        print(f"Program {i}:")
        print(prog)
        spec = compile_program(prog)
        print(f"\nCompiled spec:")
        print(f"  ops={spec.ops} alpha={spec.alpha} font={spec.fontsize_pct}")
        print(f"  dur={spec.key_duration_s}s flash={spec.key_flash_hz}Hz")
        print(f"  distractors={spec.n_distractors} (yellow={spec.yellow_distractors})")
        print(f"  gates={spec.gates}")
        print(f"  ng_params={spec.ng_params}")
        d = spec_to_genome_dict(spec)
        print(f"  genome_dict keys: {list(d.keys())}")

        # Roundtrip
        ast = parse_program(prog)
        rt = unparse(ast)
        print(f"  roundtrip: {rt[:80]}...")
