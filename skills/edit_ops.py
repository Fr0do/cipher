"""
edit_ops.py — NL instruction parser + ffmpeg chain builder for video-captcha-solver.

Usage:
    from edit_ops import parse_instructions, build_ffmpeg_commands
    ops = parse_instructions("cut seconds 3-7, reverse, flip horizontally", vlm_fn)
    cmds = build_ffmpeg_commands("input.mp4", ops, "output.mp4")
    for cmd in cmds: subprocess.run(cmd, check=True)
"""
import re
import json
import tempfile
import subprocess
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Rule-based NL parser (fast path, no VLM needed for common patterns)
# ---------------------------------------------------------------------------

_OPT_FILLER = r"(?:\s+(?:the\s+)?(?:video|clip|footage|it))?"  # matches " the video", " it", or nothing

_PATTERNS = [
    # trim / cut
    (r"(?:cut|trim|keep|remove)\s+(?:seconds?\s+)?(\d+(?:\.\d+)?)\s*[-–to]+\s*(\d+(?:\.\d+)?)",
     lambda m: {"op": "trim", "start": float(m.group(1)), "end": float(m.group(2))}),
    (r"(?:cut|remove)\s+(?:the\s+)?first\s+(\d+(?:\.\d+)?)\s*s",
     lambda m: {"op": "trim", "start": float(m.group(1)), "end": None}),
    # reverse — "reverse the video", "play it backwards", "play the video in reverse"
    (r"\b(?:reverse|play\s+backwards?|play\s+(?:the\s+)?(?:video\s+)?(?:in\s+)?reverse|backward)\b",
     lambda m: {"op": "reverse"}),
    # hflip — "flip the video horizontally", "mirror left to right", "horizontally mirror"
    (r"\b(?:flip|mirror)" + _OPT_FILLER + r"\s+(?:horizontally|left[\s\-]to[\s\-]right|left.right|h(?:orizontal)?)\b",
     lambda m: {"op": "hflip"}),
    (r"\bmirror" + _OPT_FILLER + r"\s+left[\s\-]to[\s\-]right\b",
     lambda m: {"op": "hflip"}),
    (r"\bhorizontal(?:ly)?\s+(?:flip|mirror)\b",
     lambda m: {"op": "hflip"}),
    (r"\bapply\s+(?:a\s+)?horizontal\s+(?:flip|mirror)\b",
     lambda m: {"op": "hflip"}),
    # vflip — "flip the video vertically", "upside down", "top to bottom"
    (r"\b(?:flip|mirror)" + _OPT_FILLER + r"\s+(?:vertically|up.down|v(?:ertical)?|upside.down|top[\s\-]to[\s\-]bottom)\b",
     lambda m: {"op": "vflip"}),
    (r"\bupside.down\b",
     lambda m: {"op": "vflip"}),
    (r"\bvertical(?:ly)?\s+(?:flip|mirror)\b",
     lambda m: {"op": "vflip"}),
    # rotate — "rotate the video 90 degrees clockwise", "rotate 90°"
    (r"\brotate" + _OPT_FILLER + r"\s+(\d+)\s*(?:degrees?|°)?\b",
     lambda m: {"op": "rotate", "degrees": int(m.group(1))}),
    (r"\bturn" + _OPT_FILLER + r"\s+(?:clockwise|counter.?clockwise)\s+(?:by\s+)?(\d+)\s*(?:degrees?|°)?\b",
     lambda m: {"op": "rotate", "degrees": int(m.group(1))}),
    # grayscale
    (r"\b(?:grayscale|greyscale|black.and.white|b&w|desaturate|convert.*(?:gray|black))\b",
     lambda m: {"op": "grayscale"}),
    # speed — "speed up 2x", "double the playback speed", "2x speed"
    (r"\bspeed\s+up" + _OPT_FILLER + r"\s+(\d+(?:\.\d+)?)x?\b",
     lambda m: {"op": "speed", "factor": float(m.group(1))}),
    (r"\bdouble\s+(?:the\s+)?(?:playback\s+)?speed\b",
     lambda m: {"op": "speed", "factor": 2.0}),
    (r"\bhalve\s+(?:the\s+)?(?:playback\s+)?speed\b",
     lambda m: {"op": "speed", "factor": 0.5}),
    (r"\bslow\s+down" + _OPT_FILLER + r"\s+(\d+(?:\.\d+)?)x?\b",
     lambda m: {"op": "speed", "factor": 1.0 / float(m.group(1))}),
    (r"\b(\d+(?:\.\d+)?)x\s+speed\b",
     lambda m: {"op": "speed", "factor": float(m.group(1))}),
    # crop
    (r"\bcrop\s+(?:to\s+)?(\d+)x(\d+)(?:\s+at\s+(\d+),(\d+))?\b",
     lambda m: {"op": "crop", "w": int(m.group(1)), "h": int(m.group(2)),
                "x": int(m.group(3) or 0), "y": int(m.group(4) or 0)}),
]


def parse_instructions_regex(instructions: str) -> list[dict]:
    # Collect (match_start, op_dict) pairs, then sort by position and dedup by span
    hits = []
    text = instructions.lower()
    for pattern, builder in _PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            hits.append((m.start(), m.end(), builder(m)))

    # Sort by position, drop overlapping matches (keep first)
    hits.sort(key=lambda x: x[0])
    ops = []
    last_end = -1
    for start, end, op in hits:
        if start >= last_end:  # no overlap
            ops.append(op)
            last_end = end
    return ops


def parse_instructions_vlm(instructions: str, vlm_fn) -> list[dict]:
    """VLM fallback for complex/ambiguous instructions."""
    prompt = f"""You are parsing video editing instructions for ffmpeg execution.
Extract ALL edit operations from the following instructions as a JSON list.
Each operation: {{"op": "<type>", "params": {{...}}}}
Op types: trim, reverse, hflip, vflip, rotate, grayscale, speed, crop, blur_region, select_frames

Instructions: {instructions}

Return ONLY valid JSON array, no explanation."""
    response = vlm_fn(prompt)
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # try to extract JSON from response
        m = re.search(r'\[.*\]', response, re.DOTALL)
        if m:
            return json.loads(m.group())
        return []


def parse_instructions(instructions: str, vlm_fn=None) -> list[dict]:
    ops = parse_instructions_regex(instructions)
    if not ops and vlm_fn:
        ops = parse_instructions_vlm(instructions, vlm_fn)
    return ops


# ---------------------------------------------------------------------------
# ffmpeg command builder
# ---------------------------------------------------------------------------

def build_ffmpeg_commands(
    input_path: str,
    ops: list[dict],
    output_path: str,
) -> list[list[str]]:
    """
    Returns a list of ffmpeg commands to run sequentially.
    Uses temp files for intermediate steps.
    """
    cmds = []
    src = input_path
    tmp_files = []

    for i, op in enumerate(ops):
        is_last = (i == len(ops) - 1)
        dst = output_path if is_last else _tmp_path(tmp_files, i)
        cmd = _op_to_cmd(src, op, dst)
        if cmd:
            cmds.append(cmd)
            src = dst

    return cmds, tmp_files


def _tmp_path(tmp_files: list, idx: int) -> str:
    p = tempfile.mktemp(suffix=f"_step{idx}.mp4", prefix="captcha_")
    tmp_files.append(p)
    return p


def _op_to_cmd(src: str, op: dict, dst: str) -> list[str] | None:
    base = ["ffmpeg", "-y", "-i", src]
    t = op["op"]

    if t == "trim":
        start = op.get("start", 0)
        end = op.get("end")
        vf = f"trim=start={start}" + (f":end={end}" if end else "") + ",setpts=PTS-STARTPTS"
        af = f"atrim=start={start}" + (f":end={end}" if end else "") + ",asetpts=PTS-STARTPTS"
        return base + ["-vf", vf, "-af", af, dst]

    elif t == "reverse":
        # For short clips use filter; for long clips extract + reverse + mux
        return base + ["-vf", "reverse", "-af", "areverse", dst]

    elif t == "hflip":
        return base + ["-vf", "hflip", dst]

    elif t == "vflip":
        return base + ["-vf", "vflip", dst]

    elif t == "rotate":
        deg = op.get("degrees", 90)
        transpose_map = {90: "transpose=1", 180: "transpose=2,transpose=2", 270: "transpose=2"}
        vf = transpose_map.get(deg, f"rotate={deg}*PI/180")
        return base + ["-vf", vf, dst]

    elif t == "grayscale":
        return base + ["-vf", "format=gray", dst]

    elif t == "speed":
        factor = op.get("factor", 1.0)
        pts = 1.0 / factor
        vf = f"setpts={pts:.4f}*PTS"
        # atempo only supports 0.5-2.0; chain for extremes
        af = _build_atempo(factor)
        return base + ["-vf", vf, "-af", af, dst]

    elif t == "crop":
        w, h = op.get("w", "iw"), op.get("h", "ih")
        x, y = op.get("x", 0), op.get("y", 0)
        return base + ["-vf", f"crop={w}:{h}:{x}:{y}", dst]

    elif t == "blur_region":
        x1, y1 = op.get("x1", 0), op.get("y1", 0)
        x2, y2 = op.get("x2", 100), op.get("y2", 100)
        w, h = x2 - x1, y2 - y1
        vf = (f"split[a][b];[b]crop={w}:{h}:{x1}:{y1},boxblur=20:20[bb];"
              f"[a][bb]overlay={x1}:{y1}")
        return base + ["-filter_complex", vf, dst]

    elif t == "select_frames":
        f1, f2 = op.get("start_frame", 0), op.get("end_frame", 100)
        vf = f"select='between(n\\,{f1}\\,{f2})',setpts=N/FRAME_RATE/TB"
        return base + ["-vf", vf, "-vsync", "vfr", dst]

    return None  # unknown op, skip


def _build_atempo(factor: float) -> str:
    """Chain atempo filters since each is limited to 0.5-2.0."""
    filters = []
    f = factor
    while f > 2.0:
        filters.append("atempo=2.0")
        f /= 2.0
    while f < 0.5:
        filters.append("atempo=0.5")
        f *= 2.0
    filters.append(f"atempo={f:.4f}")
    return ",".join(filters)
