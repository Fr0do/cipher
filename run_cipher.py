#!/usr/bin/env python3
"""
run_cipher.py — top-level CIPHER benchmark runner.

Usage:
  # Generate dataset (once):
  python run_cipher.py generate --videos data/backgrounds/ --n 50

  # Benchmark a model:
  python run_cipher.py bench --model haiku --manifest data/manifest.json

  # Benchmark multiple models in sequence:
  python run_cipher.py bench --model haiku gpt-4o-mini ocr-only --manifest data/manifest.json

  # Quick smoke test (2 samples per level, ocr-only):
  python run_cipher.py bench --model ocr-only --manifest data/manifest.json --max-samples 2
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from generator import BackgroundPool, generate_benchmark, LEVELS
from evaluator import run_benchmark
from vlm_backends import get_backend, REGISTRY


def cmd_generate(args):
    pool = BackgroundPool()
    for v in args.videos:
        if os.path.isdir(v):
            p = BackgroundPool.from_dir(v)
            for vid in p._videos:
                pool.add(vid)
        else:
            pool.add(v)

    if len(pool) == 0:
        sys.exit("Error: no background videos found. Run scripts/fetch_backgrounds.sh first.")

    print(f"Background pool: {len(pool)} videos")
    levels = args.levels or list(LEVELS.keys())
    generate_benchmark(pool=pool, output_dir=args.output,
                       n_per_level=args.n, levels=levels,
                       seed_offset=args.seed_offset)


def cmd_bench(args):
    if not os.path.exists(args.manifest):
        sys.exit(f"Manifest not found: {args.manifest}")

    models = args.model
    all_summaries = {}

    for model_name in models:
        print(f"\n{'='*60}\nModel: {model_name}\n{'='*60}")
        vlm_fn = get_backend(model_name)

        os.makedirs(args.runs_dir, exist_ok=True)
        out_path = os.path.join(args.runs_dir,
                                f"{model_name.replace('/', '_')}.json")

        report = run_benchmark(
            manifest_path=args.manifest,
            vlm_fn=vlm_fn,
            levels=args.levels or None,
            max_samples=args.max_samples,
            output_path=out_path,
        )
        all_summaries[model_name] = report["overall"]

    # Cross-model summary table
    if len(models) > 1:
        print(f"\n{'='*60}\nCROSS-MODEL SUMMARY\n{'='*60}")
        print(f"{'Model':<20} {'EM':>6} {'CharF1':>7} {'PipeF1':>7} {'S1':>6} {'S2':>6}")
        print("-" * 55)
        for name, st in all_summaries.items():
            print(f"{name:<20} {st['EM']:>6.1%} {st['char_F1']:>7.3f} "
                  f"{st['pipeline_F1']:>7.3f} {st['S1_parse_F1']:>6.3f} "
                  f"{st['S2_execute_rate']:>6.1%}")


def main():
    parser = argparse.ArgumentParser(description="CIPHER benchmark")
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate", help="Generate benchmark dataset")
    gen.add_argument("--videos", nargs="+", required=True)
    gen.add_argument("--output", default="data")
    gen.add_argument("--levels", nargs="+", choices=list(LEVELS.keys()))
    gen.add_argument("--n", type=int, default=50)
    gen.add_argument("--seed-offset", type=int, default=0)

    bench = sub.add_parser("bench", help="Run model benchmark")
    bench.add_argument("--manifest", default="data/manifest.json")
    bench.add_argument("--model", nargs="+", default=["ocr-only"],
                       choices=list(REGISTRY.keys()))
    bench.add_argument("--levels", nargs="+", choices=list(LEVELS.keys()))
    bench.add_argument("--max-samples", type=int)
    bench.add_argument("--runs-dir", default="runs")

    args = parser.parse_args()
    {"generate": cmd_generate, "bench": cmd_bench}[args.cmd](args)


if __name__ == "__main__":
    main()
