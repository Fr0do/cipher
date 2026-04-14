#!/usr/bin/env python3
"""
prepare_calibration.py — generates samples, runs S1+S2, extracts frames.
Output: calibration_state.json with per-sample state for S3 evaluation.

Run this first, then the orchestrator spawns Haiku agents for S3.

Usage:
    python scripts/prepare_calibration.py \
        --videos /tmp/big_buck_bunny_720p.mp4 \
        --n 5 \
        --out calibration_state.json
"""
import sys, os, json, argparse, tempfile, subprocess, hashlib, shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path.home() / ".hermes/skills/media/video-captcha-solver/scripts"))

from generator import BackgroundPool, generate_sample, LEVELS
from edit_ops import parse_instructions, build_ffmpeg_commands
from evaluator import op_parse_f1


def extract_frames(video_path: str, out_dir: str, fps: float = 2.0) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path,
         "-vf", f"fps={fps},scale=640:-2", f"{out_dir}/%03d.jpg"],
        capture_output=True,
    )
    return sorted(str(p) for p in Path(out_dir).glob("*.jpg"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed-offset", type=int, default=9000)
    parser.add_argument("--out", default="calibration_state.json")
    parser.add_argument("--frames-dir", default="/tmp/cipher_cal_frames")
    args = parser.parse_args()

    pool = BackgroundPool()
    for v in args.videos:
        if os.path.isdir(v):
            for vid in BackgroundPool.from_dir(v)._videos:
                pool.add(vid)
        else:
            pool.add(v)

    state = {}  # level -> list of sample records

    for level in LEVELS:
        print(f"\n=== {level} ===")
        level_hash = int(hashlib.md5(level.encode()).hexdigest()[:8], 16) % 100_000
        tmp_dir = tempfile.mkdtemp(prefix=f"cipher_gen_{level}_")
        records = []

        for i in range(args.n):
            seed = args.seed_offset + level_hash + i
            try:
                s = generate_sample(seed, level, pool, tmp_dir)
            except Exception as e:
                print(f"  gen error seed={seed}: {e}")
                continue

            sd = s.to_dict()

            # S1: parse instructions
            ops = parse_instructions(sd["nl_instructions"])
            pred_ops = [op["op"] for op in ops]
            s1_f1 = op_parse_f1(pred_ops, sd["ops"])

            # S2: execute ffmpeg
            exec_ok = False
            frame_paths = []
            if ops:
                out_td = tempfile.mkdtemp(prefix="cipher_s2_")
                out_path = os.path.join(out_td, "output.mp4")
                cmds, _ = build_ffmpeg_commands(sd["scrambled_video"], ops, out_path)
                ok = True
                for cmd in cmds:
                    r = subprocess.run(cmd, capture_output=True)
                    if r.returncode != 0:
                        ok = False
                        break
                if ok and Path(out_path).exists() and Path(out_path).stat().st_size > 1000:
                    exec_ok = True
                    frame_dir = os.path.join(args.frames_dir, sd["sample_id"])
                    frame_paths = extract_frames(out_path, frame_dir)
                shutil.rmtree(out_td, ignore_errors=True)

            record = {
                "sample_id": sd["sample_id"],
                "level": level,
                "gold_key": sd["key"],
                "gold_ops": sd["ops"],
                "nl_instructions": sd["nl_instructions"],
                "pred_ops": pred_ops,
                "s1_parse_f1": round(s1_f1, 3),
                "s2_execute_ok": exec_ok,
                "frame_paths": frame_paths,  # for S3 by Haiku agent
                "pred_key": "",              # filled in by S3 agent
                "s3_em": False,
                "s3_char_f1": 0.0,
            }
            records.append(record)
            mark = "✓" if exec_ok else "✗"
            print(f"  {mark} {sd['sample_id']} key={sd['key']} ops={sd['ops']} "
                  f"parse_f1={s1_f1:.2f} frames={len(frame_paths)}")

        state[level] = records
        shutil.rmtree(tmp_dir, ignore_errors=True)

    with open(args.out, "w") as f:
        json.dump(state, f, indent=2)
    print(f"\nState saved → {args.out}")
    total = sum(len(v) for v in state.values())
    ready = sum(1 for v in state.values() for r in v if r["s2_execute_ok"])
    print(f"Samples: {total} total, {ready} ready for S3")


if __name__ == "__main__":
    main()
