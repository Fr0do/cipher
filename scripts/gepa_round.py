#!/usr/bin/env python3
"""GEPA round orchestrator: generate batches, then score when predictions are ready."""
import argparse
import json
import os
import sys

sys.path.insert(0, "/Users/mkurkin/experiments/cipher/scripts")
from captcha_dsl import generate_image_sample  # type: ignore

EXCLUDE_KEYS = {"shadow", "position_mode", "distractor_same_length"}


def _load_programs(programs_path):
    with open(programs_path) as f:
        programs = json.load(f)
    if not isinstance(programs, list):
        raise ValueError("programs.json must be a list")
    for p in programs:
        if not all(k in p for k in ("program_idx", "desc", "genome")):
            raise ValueError("Each program must have program_idx, desc, genome")
    return programs


def generate_round(workdir, programs_path, n_samples, seed_base):
    programs = _load_programs(programs_path)
    os.makedirs(workdir, exist_ok=True)

    batches = []
    for prog in programs:
        pi = int(prog["program_idx"])
        out_dir = os.path.join(workdir, f"prog_{pi}")
        os.makedirs(out_dir, exist_ok=True)
        samples = []
        for j in range(n_samples):
            seed = seed_base + pi * 1000 + j
            r = generate_image_sample(seed=seed, genome_dict=prog["genome"], output_dir=out_dir)
            samples.append({
                "sample_id": r["sample_id"],
                "image_path": r["image_path"],
                "gold_key": r["gold_key"],
            })
            print(f"  prog_{pi} s{j}: {r['sample_id']} key={r['gold_key']}")
        batches.append({
            "program_idx": pi,
            "desc": prog.get("desc", ""),
            "samples": samples,
            "genome_summary": {k: v for k, v in prog["genome"].items() if k not in EXCLUDE_KEYS},
        })

    with open(os.path.join(workdir, "batches.json"), "w") as f:
        json.dump(batches, f, indent=2)

    total = sum(len(b["samples"]) for b in batches)
    print(f"\n{len(batches)} progs × {n_samples} = {total} images")


def score_round(workdir):
    batches_path = os.path.join(workdir, "batches.json")
    if not os.path.exists(batches_path):
        raise FileNotFoundError(f"Missing batches.json in {workdir}")

    with open(batches_path) as f:
        batches = json.load(f)

    target_acc = 0.03
    results = []

    for b in batches:
        pi = b["program_idx"]
        pred_path = os.path.join(workdir, f"pred_prog{pi}.json")
        gold = {s["sample_id"]: s["gold_key"] for s in b["samples"]}

        if os.path.exists(pred_path):
            with open(pred_path) as f:
                preds = {p["sample_id"]: p.get("pred_key", "")
                         for p in json.load(f).get("predictions", [])}
        else:
            preds = {}

        n = len(gold)
        matches = sum(1 for sid, gk in gold.items()
                      if preds.get(sid, "").strip().upper() == gk.upper())
        missing = sum(1 for sid in gold if sid not in preds)
        em = matches / n if n > 0 else 0

        char_correct = char_total = 0
        for sid, gk in gold.items():
            pk = preds.get(sid, "").strip().upper()
            for i, gc in enumerate(gk.upper()):
                char_total += 1
                if i < len(pk) and pk[i] == gc:
                    char_correct += 1
        char_acc = char_correct / char_total if char_total > 0 else 0
        fitness = -abs(em - target_acc) - 0.5 * (missing / max(n, 1))

        r = {
            "program_idx": pi,
            "em": em,
            "char_acc": char_acc,
            "matches": matches,
            "n": n,
            "missing": missing,
            "fitness": fitness,
            "desc": b.get("desc", ""),
            "params": b.get("genome_summary", {}),
        }
        results.append(r)

        status = "NEAR TARGET" if abs(em - target_acc) <= 0.05 else ("TOO EASY" if em > 0.08 else "TOO HARD")
        print(f"[{pi}] EM={em:.0%} ({matches}/{n}) char_acc={char_acc:.0%} fit={fitness:.3f} | {status}")
        print(f"    {b.get('desc','')}")

    print(f"\n{'='*60}")
    best = max(results, key=lambda r: r["fitness"])
    print(f"Best: prog_{best['program_idx']} fit={best['fitness']:.3f} EM={best['em']:.0%} char_acc={best['char_acc']:.0%}")
    print(f"  {best['desc']}")

    with open(os.path.join(workdir, "scores.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results, best


def main():
    ap = argparse.ArgumentParser(description="GEPA round orchestrator")
    ap.add_argument("--round", type=int, required=True, help="Round number (for logging)")
    ap.add_argument("--workdir", required=True, help="Work directory for round outputs")
    ap.add_argument("--programs", required=True, help="Path to programs.json")
    ap.add_argument("--samples", type=int, default=8, help="Samples per program")
    ap.add_argument("--seed-base", type=int, default=30000, help="Base seed")
    ap.add_argument("--score-only", action="store_true", help="Skip generation and only score")
    args = ap.parse_args()

    if args.score_only:
        score_round(args.workdir)
        return

    print(f"GEPA round {args.round}: generating batches in {args.workdir}")
    generate_round(args.workdir, args.programs, args.samples, args.seed_base)
    print("\nNow run VLM agents to populate pred_prog*.json, then re-run this script with --score-only")


if __name__ == "__main__":
    main()
