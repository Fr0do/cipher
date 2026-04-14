#!/usr/bin/env python3
"""Generate Round 4 GEPA images — interpolating between R3 prog_3 (0%) and prog_0 (25%).
Target: ~3% EM. Need slightly easier than prog_3 but harder than prog_0.
Also increase sample count to 8 for more reliable EM estimates."""
import json, os, sys
sys.path.insert(0, "/Users/mkurkin/experiments/cipher/scripts")
from captcha_dsl import generate_image_sample

work_dir = "/tmp/cipher_gepa_r4"
os.makedirs(work_dir, exist_ok=True)

N_SAMPLES = 8  # more samples for reliable EM

programs = [
    {
        "program_idx": 0,
        "desc": "α=148, font=4.9%, 3dist(2y), noise=3.5, blur=0.22 — midpoint prog3↔prog0",
        "genome": {"alpha": 148, "fontsize_pct": 0.049, "key_length": 8, "n_distractors": 3, "yellow_distractors": 2, "noise_sigma": 3.5, "blur_radius": 0.22, "shadow": True, "position_mode": "edge", "distractor_same_length": True},
    },
    {
        "program_idx": 1,
        "desc": "α=147, font=4.85%, 3dist(2y), noise=3.8, blur=0.23 — lean toward prog3",
        "genome": {"alpha": 147, "fontsize_pct": 0.0485, "key_length": 8, "n_distractors": 3, "yellow_distractors": 2, "noise_sigma": 3.8, "blur_radius": 0.23, "shadow": True, "position_mode": "edge", "distractor_same_length": True},
    },
    {
        "program_idx": 2,
        "desc": "α=146, font=4.9%, 4dist(3y), noise=3, blur=0.2 — more yellow distractors",
        "genome": {"alpha": 146, "fontsize_pct": 0.049, "key_length": 8, "n_distractors": 4, "yellow_distractors": 3, "noise_sigma": 3.0, "blur_radius": 0.2, "shadow": True, "position_mode": "random", "distractor_same_length": True},
    },
    {
        "program_idx": 3,
        "desc": "α=149, font=4.95%, 3dist(2y), noise=3, blur=0.2 — slightly easier than midpoint",
        "genome": {"alpha": 149, "fontsize_pct": 0.0495, "key_length": 8, "n_distractors": 3, "yellow_distractors": 2, "noise_sigma": 3.0, "blur_radius": 0.2, "shadow": True, "position_mode": "corner", "distractor_same_length": True},
    },
    {
        "program_idx": 4,
        "desc": "α=145, font=4.9%, 4dist(4y), noise=2, blur=0.15 — max yellow distractor confusion",
        "genome": {"alpha": 145, "fontsize_pct": 0.049, "key_length": 8, "n_distractors": 4, "yellow_distractors": 4, "noise_sigma": 2.0, "blur_radius": 0.15, "shadow": True, "position_mode": "random", "distractor_same_length": True},
    },
]

batches = []
seed_base = 20000

for prog in programs:
    pi = prog["program_idx"]
    out_dir = os.path.join(work_dir, f"prog_{pi}")
    os.makedirs(out_dir, exist_ok=True)

    samples = []
    for j in range(N_SAMPLES):
        seed = seed_base + pi * 1000 + j
        result = generate_image_sample(
            seed=seed,
            genome_dict=prog["genome"],
            output_dir=out_dir,
        )
        samples.append({
            "sample_id": result["sample_id"],
            "image_path": result["image_path"],
            "gold_key": result["gold_key"],
        })
        print(f"  prog_{pi} sample {j}: {result['sample_id']} key={result['gold_key']}")

    batches.append({
        "program_idx": pi,
        "desc": prog["desc"],
        "samples": samples,
        "genome_summary": {k: v for k, v in prog["genome"].items() if k not in ("shadow", "position_mode", "distractor_same_length")},
    })

with open(os.path.join(work_dir, "batches.json"), "w") as f:
    json.dump(batches, f, indent=2)

print(f"\nGenerated {len(batches)} programs x {N_SAMPLES} samples = {sum(len(b['samples']) for b in batches)} images")
