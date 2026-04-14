#!/usr/bin/env python3
"""
calibrate_overnight.py — autonomous overnight calibration for CIPHER.

Generates N samples per level, evaluates with OCR-only MCP skill,
saves manifest + frames for subsequent VLM agent evaluation.

Designed for unattended execution:
- Monitors disk usage, aborts if < 2GB free
- Progressive saves (results after each sample)
- Robust error handling per-sample
- Cleans intermediate videos after frame extraction

Usage:
    nohup python scripts/calibrate_overnight.py \
        --bg /tmp/big_buck_bunny_720p.mp4 \
        --n 20 --out-dir /tmp/cipher_overnight &
"""
import sys, os, json, hashlib, tempfile, subprocess, shutil, time, logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
_SKILL = Path.home() / ".hermes/skills/media/video-captcha-solver/scripts"
sys.path.insert(0, str(_SKILL))

from generator import BackgroundPool, generate_sample, LEVELS
from evaluator import op_parse_f1, exact_match, char_f1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/tmp/cipher_overnight.log"),
    ],
)
log = logging.getLogger("cipher")


def disk_free_gb() -> float:
    st = os.statvfs("/tmp")
    return (st.f_bavail * st.f_frsize) / (1024 ** 3)


def extract_frames(video_path: str, out_dir: str, fps: float = 2.0) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path,
         "-vf", f"fps={fps},scale=640:-2", f"{out_dir}/%03d.jpg"],
        capture_output=True,
    )
    return sorted(str(p) for p in Path(out_dir).glob("*.jpg"))


def run_skill_ocr(scrambled_video: str, nl_instructions: str) -> str:
    """Run the MCP solve_captcha skill (OCR-only, no VLM)."""
    try:
        from solve_captcha import solve
        key = solve(
            video_path=scrambled_video,
            instructions=nl_instructions,
            vlm_fn=None,
            max_retries=2,
        )
        return key or ""
    except Exception as e:
        log.warning(f"skill error: {e}")
        return ""


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bg", nargs="+", required=True, help="Background video(s)")
    parser.add_argument("--n", type=int, default=20, help="Samples per level")
    parser.add_argument("--out-dir", default="/tmp/cipher_overnight")
    parser.add_argument("--seed-offset", type=int, default=42000)
    parser.add_argument("--min-disk-gb", type=float, default=2.0)
    parser.add_argument("--fps", type=float, default=2.0, help="Frame extraction FPS")
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    videos_dir = os.path.join(out_dir, "videos")
    frames_dir = os.path.join(out_dir, "frames")
    results_path = os.path.join(out_dir, "results.json")

    # Load background pool
    pool = BackgroundPool()
    for v in args.bg:
        if os.path.isdir(v):
            p = BackgroundPool.from_dir(v)
            for vid in p._videos:
                pool.add(vid)
        else:
            pool.add(v)

    if not pool._videos:
        log.error("No background videos found")
        sys.exit(1)

    log.info(f"Background pool: {len(pool)} videos")
    log.info(f"Generating {args.n} samples/level, {len(LEVELS)} levels")
    log.info(f"Output: {out_dir}")
    log.info(f"Disk free: {disk_free_gb():.1f} GB")

    # Load existing results (for resumability)
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        log.info(f"Resuming: {sum(len(v) for v in results.values())} samples done")
    else:
        results = {}

    t0 = time.time()

    for level in LEVELS:
        if level not in results:
            results[level] = []

        done_ids = {r["sample_id"] for r in results[level]}
        level_hash = int(hashlib.md5(level.encode()).hexdigest()[:8], 16) % 100_000
        level_vid_dir = os.path.join(videos_dir, level)

        log.info(f"\n=== {level} ({len(done_ids)}/{args.n} done) ===")

        for i in range(args.n):
            seed = args.seed_offset + level_hash + i
            sample_id = f"{level}_s{seed:06d}"

            if sample_id in done_ids:
                continue

            # Disk check
            free = disk_free_gb()
            if free < args.min_disk_gb:
                log.error(f"Disk low: {free:.1f} GB < {args.min_disk_gb} GB — aborting")
                _save(results, results_path)
                sys.exit(2)

            try:
                s = generate_sample(seed, level, pool, level_vid_dir)
            except Exception as e:
                log.warning(f"  gen error {sample_id}: {e}")
                continue

            sd = s.to_dict()

            # S1 parse
            from edit_ops import parse_instructions
            ops = parse_instructions(sd["nl_instructions"])
            pred_ops = [op["op"] for op in ops]
            s1_f1 = op_parse_f1(pred_ops, sd["ops"])

            # MCP skill (OCR-only)
            skill_key = run_skill_ocr(sd["scrambled_video"], sd["nl_instructions"])
            skill_em = exact_match(skill_key, sd["key"])
            skill_cf1 = char_f1(skill_key, sd["key"])

            # Extract frames (for later agent evaluation)
            frame_dir = os.path.join(frames_dir, sample_id)
            frame_paths = extract_frames(sd["scrambled_video"], frame_dir, fps=args.fps)

            # Also extract corrected video frames (after S2)
            # Run S2 ourselves to get the corrected video
            corrected_frames = []
            try:
                from edit_ops import build_ffmpeg_commands
                td = tempfile.mkdtemp(prefix="cipher_s2_")
                out_path = os.path.join(td, "corrected.mp4")
                cmds, _ = build_ffmpeg_commands(sd["scrambled_video"], ops, out_path)
                s2_ok = True
                for cmd in cmds:
                    r = subprocess.run(cmd, capture_output=True)
                    if r.returncode != 0:
                        s2_ok = False
                        break
                if s2_ok and Path(out_path).exists() and Path(out_path).stat().st_size > 1000:
                    corrected_dir = os.path.join(frames_dir, sample_id + "_corrected")
                    corrected_frames = extract_frames(out_path, corrected_dir, fps=args.fps)
                shutil.rmtree(td, ignore_errors=True)
            except Exception as e:
                log.warning(f"  S2 error {sample_id}: {e}")
                s2_ok = False

            # Keep scrambled video path (it's in videos_dir, persistent)
            record = {
                "sample_id": sample_id,
                "level": level,
                "seed": seed,
                "gold_key": sd["key"],
                "gold_ops": sd["ops"],
                "nl_instructions": sd["nl_instructions"],
                "scrambled_video": sd["scrambled_video"],
                "scrambled_frames": frame_paths,
                "corrected_frames": corrected_frames,
                "s1_parse_f1": round(s1_f1, 3),
                "s2_ok": s2_ok,
                "skill_pred_key": skill_key,
                "skill_em": skill_em,
                "skill_char_f1": round(skill_cf1, 3),
            }
            results[level].append(record)

            mark = "✓" if skill_em else ("~" if skill_cf1 > 0.5 else "✗")
            log.info(
                f"  {mark} {sample_id} gold={sd['key']} skill={skill_key!r:10s} "
                f"S1={s1_f1:.2f} S2={'ok' if s2_ok else 'no'} "
                f"frames={len(frame_paths)}+{len(corrected_frames)} "
                f"disk={disk_free_gb():.1f}GB"
            )

            # Progressive save
            _save(results, results_path)

    elapsed = time.time() - t0
    log.info(f"\nDone in {elapsed/60:.1f} min")
    _print_summary(results)
    _save(results, results_path)


def _save(results: dict, path: str):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def _print_summary(results: dict):
    print("\n" + "=" * 65)
    print(f"{'Level':<6} {'N':>3} {'Skill EM':>10} {'Skill CF1':>10} {'S1 F1':>8} {'S2 OK':>6}")
    print("-" * 55)
    for level, recs in results.items():
        if not recs:
            continue
        n = len(recs)
        em = sum(r["skill_em"] for r in recs) / n
        cf1 = sum(r["skill_char_f1"] for r in recs) / n
        s1 = sum(r["s1_parse_f1"] for r in recs) / n
        s2 = sum(r.get("s2_ok", False) for r in recs) / n
        print(f"{level:<6} {n:>3} {em:>10.0%} {cf1:>10.3f} {s1:>8.3f} {s2:>6.0%}")

    total = [r for recs in results.values() for r in recs]
    if total:
        n = len(total)
        print("-" * 55)
        print(f"{'TOTAL':<6} {n:>3} "
              f"{sum(r['skill_em'] for r in total)/n:>10.0%} "
              f"{sum(r['skill_char_f1'] for r in total)/n:>10.3f} "
              f"{sum(r['s1_parse_f1'] for r in total)/n:>8.3f} "
              f"{sum(r.get('s2_ok', False) for r in total)/n:>6.0%}")
    print(f"\nResults: /tmp/cipher_overnight/results.json")
    print(f"Disk free: {disk_free_gb():.1f} GB")


if __name__ == "__main__":
    main()
