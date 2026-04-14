#!/usr/bin/env python3
"""
calibrate.py — difficulty calibration via Haiku subagent.

Generates N samples per level, runs S1 (regex) + S2 (ffmpeg) locally,
then spawns a Haiku Claude Code subagent for S3 key extraction from frames.

Target EM: L1~90%, L2~70%, L3~50%, L4~30%, L5~10%.

Usage:
    python scripts/calibrate.py --videos data/backgrounds/ --n 10
"""
import sys, os, json, argparse, tempfile, subprocess, hashlib, shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from generator import BackgroundPool, generate_sample, LEVELS
from evaluator import op_parse_f1, exact_match, char_f1, _pipeline_f1

_SKILL = Path.home() / ".hermes/skills/media/video-captcha-solver/scripts"
sys.path.insert(0, str(_SKILL))

TARGETS = {"L1": 0.90, "L2": 0.70, "L3": 0.50, "L4": 0.30, "L5": 0.10}
TOLERANCE = 0.15


def extract_frames(video_path: str, out_dir: str, fps: float = 2.0) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vf", f"fps={fps},scale=640:-2",
         f"{out_dir}/%03d.jpg"],
        capture_output=True,
    )
    return sorted(str(p) for p in Path(out_dir).glob("*.jpg"))


def haiku_extract_key(frame_paths: list[str], gold_key: str) -> str:
    """
    Spawn a Haiku Claude Code subagent to read frames and identify the key.
    Returns predicted key string (or empty string).
    """
    if not frame_paths:
        return ""

    # Pick up to 8 evenly spaced frames
    step = max(1, len(frame_paths) // 8)
    selected = frame_paths[::step][:8]

    from agent_caller import call_haiku_agent
    return call_haiku_agent(selected)


def _run_pipeline_s1s2(sample_dict: dict) -> tuple[list, bool, str]:
    """Run S1 parse + S2 execute, return (pred_ops, execute_ok, out_video_path)."""
    from edit_ops import parse_instructions, build_ffmpeg_commands

    ops = parse_instructions(sample_dict["nl_instructions"])
    if not ops:
        return [], False, ""

    td = tempfile.mkdtemp(prefix="cipher_s2_")
    out_path = os.path.join(td, "output.mp4")
    cmds, _ = build_ffmpeg_commands(sample_dict["scrambled_video"], ops, out_path)

    for cmd in cmds:
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0:
            shutil.rmtree(td, ignore_errors=True)
            return [op["op"] for op in ops], False, ""

    if Path(out_path).exists() and Path(out_path).stat().st_size > 1000:
        return [op["op"] for op in ops], True, out_path

    shutil.rmtree(td, ignore_errors=True)
    return [op["op"] for op in ops], False, ""


def calibrate(videos: list[str], n_probe: int, seed_offset: int = 9000):
    pool = BackgroundPool()
    for v in videos:
        if os.path.isdir(v):
            p = BackgroundPool.from_dir(v)
            for vid in p._videos:
                pool.add(vid)
        else:
            pool.add(v)

    if not pool._videos:
        sys.exit("No background videos — run scripts/fetch_backgrounds.sh first")

    print(f"Calibrating {n_probe} samples/level via Haiku subagent\n")
    print(f"{'Level':<6} {'Target':>7} {'EM':>6} {'ParseF1':>8} {'ExecOK':>7}  Status")
    print("-" * 52)

    recommendations = []

    for level in LEVELS:
        level_hash = int(hashlib.md5(level.encode()).hexdigest()[:8], 16) % 100_000
        tmp_dir = tempfile.mkdtemp(prefix=f"cipher_cal_{level}_")

        results = []
        for i in range(n_probe):
            seed = seed_offset + level_hash + i
            try:
                s = generate_sample(seed, level, pool, tmp_dir)
            except Exception as e:
                print(f"  [gen error] {level} s{seed}: {e}", file=sys.stderr)
                continue

            sd = s.to_dict()
            pred_ops, exec_ok, out_video = _run_pipeline_s1s2(sd)
            parse_f1 = op_parse_f1(pred_ops, sd["ops"])

            pred_key = ""
            if exec_ok and out_video:
                frame_dir = tempfile.mkdtemp(prefix="cipher_frames_")
                frames = extract_frames(out_video, frame_dir)
                pred_key = _haiku_subagent_extract(frames, sd["key"], sd["sample_id"])
                shutil.rmtree(frame_dir, ignore_errors=True)
                shutil.rmtree(str(Path(out_video).parent), ignore_errors=True)

            em = exact_match(pred_key, sd["key"])
            cf1 = char_f1(pred_key, sd["key"])
            results.append({
                "em": em, "char_f1": cf1,
                "parse_f1": parse_f1, "exec_ok": exec_ok,
                "gold": sd["key"], "pred": pred_key,
            })

        shutil.rmtree(tmp_dir, ignore_errors=True)

        if not results:
            print(f"{level:<6} {'--':>7} {'--':>6} {'--':>8} {'--':>7}  no samples")
            continue

        n = len(results)
        em_avg = sum(r["em"] for r in results) / n
        pf1_avg = sum(r["parse_f1"] for r in results) / n
        exec_avg = sum(r["exec_ok"] for r in results) / n
        target = TARGETS[level]
        diff = em_avg - target

        if abs(diff) <= TOLERANCE:
            status = "✓ OK"
        elif diff > TOLERANCE:
            status = f"↑ TOO EASY (+{diff:.0%})"
            recommendations.append(f"{level}: harder (more ops / lower visibility)")
        else:
            status = f"↓ TOO HARD ({diff:.0%})"
            recommendations.append(f"{level}: easier (fewer ops / higher visibility)")

        print(f"{level:<6} {target:>7.0%} {em_avg:>6.0%} {pf1_avg:>8.2f} {exec_avg:>7.0%}  {status}")

    if recommendations:
        print("\nRecommendations:")
        for r in recommendations:
            print(f"  • {r}")
    else:
        print("\nAll levels within target ✓")


def _haiku_subagent_extract(frame_paths: list[str], gold_key: str, sample_id: str) -> str:
    """
    Spawn a Haiku Claude Code subagent to identify the key from frame images.
    The subagent reads the images and returns the key string.
    """
    if not frame_paths:
        return ""

    step = max(1, len(frame_paths) // 8)
    selected = frame_paths[::step][:8]
    frames_list = "\n".join(f"  - {p}" for p in selected)

    # Write a temp prompt file the subagent will execute
    prompt = f"""You are solving a video captcha. A key (short alphanumeric string, 5-8 chars, uppercase) is embedded as text in these video frames:

{frames_list}

Read each image file. Find the yellow or white text that looks like a random code (e.g. K9M2P7X, BRAVO99, etc). Distractors may also be present — the real key appears for longer and is more prominent.

Output ONLY the key string, nothing else. If you see nothing, output: NONE"""

    # Use Claude Code Agent tool (haiku) — this is called from the main orchestrator
    # We import the agent caller which uses the Agent tool infrastructure
    try:
        from agent_haiku_caller import extract_key_haiku
        return extract_key_haiku(selected, prompt)
    except ImportError:
        # Fallback: use easyocr directly
        try:
            import easyocr
            import numpy as np
            from PIL import Image
            reader = easyocr.Reader(["en"], verbose=False)
            candidates = []
            for fp in selected:
                img = np.array(Image.open(fp))
                results = reader.readtext(img)
                for _, text, conf in results:
                    text = text.strip().upper()
                    if conf > 0.4 and 5 <= len(text) <= 8 and text.isalnum():
                        candidates.append((conf, text))
            if candidates:
                candidates.sort(reverse=True)
                return candidates[0][1]
        except Exception:
            pass
        return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=9000)
    args = parser.parse_args()
    calibrate(args.videos, args.n, args.seed_offset)
