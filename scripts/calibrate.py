#!/usr/bin/env python3
"""
calibrate.py — difficulty calibration via Haiku probe.

Generates N_PROBE samples per level, runs Haiku on them,
reports EM per level. Target: L1~90%, L2~70%, L3~50%, L4~30%, L5~10%.
If a level is too easy/hard, prints recommendations.

Usage:
    python scripts/calibrate.py --videos data/backgrounds/ --n 10
"""
import sys, os, json, time, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from generator import BackgroundPool, generate_sample, LEVELS
from evaluator import _run_one, compute_stats, TrackedVlmFn
from vlm_backends import anthropic_claude

# Target EM per level (for top model = Haiku here as proxy)
TARGETS = {"L1": 0.90, "L2": 0.70, "L3": 0.50, "L4": 0.30, "L5": 0.10}
TOLERANCE = 0.15  # acceptable deviation from target


def calibrate(videos, n_probe, seed_offset=9000):
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

    vlm_fn = anthropic_claude("claude-haiku-4-5-20251001")

    print(f"Calibrating with {n_probe} samples/level, model=haiku\n")
    print(f"{'Level':<6} {'Target':>7} {'Actual':>7} {'Tokens/s':>9} {'Status'}")
    print("-" * 45)

    recommendations = []

    for level in LEVELS:
        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix=f"cipher_cal_{level}_")
        import hashlib
        level_hash = int(hashlib.md5(level.encode()).hexdigest()[:8], 16) % 100_000

        samples = []
        for i in range(n_probe):
            seed = seed_offset + level_hash + i
            try:
                s = generate_sample(seed, level, pool, tmp_dir)
                samples.append(s.to_dict())
            except Exception as e:
                print(f"  [gen error] {level} seed={seed}: {e}")

        results = []
        for s in samples:
            r = _run_one(s, vlm_fn=vlm_fn)
            results.append(r)

        stats = compute_stats(results)
        em = stats.get("EM", 0)
        tok = stats.get("avg_total_tokens", 0)
        target = TARGETS[level]
        diff = em - target

        if abs(diff) <= TOLERANCE:
            status = "✓ OK"
        elif diff > TOLERANCE:
            status = f"↑ TOO EASY (+{diff:.0%})"
            recommendations.append(f"{level}: increase op_count or reduce key_visibility")
        else:
            status = f"↓ TOO HARD ({diff:.0%})"
            recommendations.append(f"{level}: decrease op_count or increase key_visibility")

        print(f"{level:<6} {target:>7.0%} {em:>7.0%} {tok:>9.0f}   {status}")

        # Cleanup generated videos
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if recommendations:
        print("\nRecommendations:")
        for r in recommendations:
            print(f"  • {r}")
    else:
        print("\nAll levels within target range ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=9000)
    args = parser.parse_args()
    calibrate(args.videos, args.n, args.seed_offset)
