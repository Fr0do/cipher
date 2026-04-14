#!/usr/bin/env python3
"""R5 — evolution targeted at Claude Sonnet (Gemini API down).
Memory note: Sonnet is weaker than Gemini on captcha; sweep from very-easy to R3-hard."""
import json, os, sys
sys.path.insert(0, "/Users/mkurkin/experiments/cipher/scripts")
from captcha_dsl import generate_image_sample

work_dir = "/tmp/cipher_gepa_r5"
os.makedirs(work_dir, exist_ok=True)
N_SAMPLES = 8

programs = [
    {"program_idx": 0, "desc": "very easy: α=220 font=8% no dist, no noise",
     "genome": {"alpha": 220, "fontsize_pct": 0.08, "key_length": 7,
                "n_distractors": 0, "yellow_distractors": 0,
                "noise_sigma": 0.0, "blur_radius": 0.0, "shadow": True,
                "position_mode": "edge", "distractor_same_length": True}},
    {"program_idx": 1, "desc": "easy: α=200 font=7% 1 dist (0y) σ=1",
     "genome": {"alpha": 200, "fontsize_pct": 0.07, "key_length": 7,
                "n_distractors": 1, "yellow_distractors": 0,
                "noise_sigma": 1.0, "blur_radius": 0.0, "shadow": True,
                "position_mode": "edge", "distractor_same_length": True}},
    {"program_idx": 2, "desc": "medium: α=180 font=6% 2 dist (1y) σ=2 blur=0.1",
     "genome": {"alpha": 180, "fontsize_pct": 0.06, "key_length": 8,
                "n_distractors": 2, "yellow_distractors": 1,
                "noise_sigma": 2.0, "blur_radius": 0.1, "shadow": True,
                "position_mode": "edge", "distractor_same_length": True}},
    {"program_idx": 3, "desc": "hard: α=160 font=5.5% 3 dist (1y) σ=3 blur=0.15",
     "genome": {"alpha": 160, "fontsize_pct": 0.055, "key_length": 8,
                "n_distractors": 3, "yellow_distractors": 1,
                "noise_sigma": 3.0, "blur_radius": 0.15, "shadow": True,
                "position_mode": "edge", "distractor_same_length": True}},
    {"program_idx": 4, "desc": "R3 prog_3 setting: α=145 font=4.8% 3 dist (2y) σ=4 blur=0.25",
     "genome": {"alpha": 145, "fontsize_pct": 0.048, "key_length": 8,
                "n_distractors": 3, "yellow_distractors": 2,
                "noise_sigma": 4.0, "blur_radius": 0.25, "shadow": True,
                "position_mode": "edge", "distractor_same_length": True}},
]

batches = []
seed_base = 30000
for prog in programs:
    pi = prog["program_idx"]
    out_dir = os.path.join(work_dir, f"prog_{pi}")
    os.makedirs(out_dir, exist_ok=True)
    samples = []
    for j in range(N_SAMPLES):
        seed = seed_base + pi * 1000 + j
        r = generate_image_sample(seed=seed, genome_dict=prog["genome"], output_dir=out_dir)
        samples.append({"sample_id": r["sample_id"], "image_path": r["image_path"], "gold_key": r["gold_key"]})
        print(f"  prog_{pi} s{j}: {r['sample_id']} key={r['gold_key']}")
    batches.append({
        "program_idx": pi, "desc": prog["desc"], "samples": samples,
        "genome_summary": {k: v for k, v in prog["genome"].items()
                           if k not in ("shadow", "position_mode", "distractor_same_length")},
    })

with open(os.path.join(work_dir, "batches.json"), "w") as f:
    json.dump(batches, f, indent=2)
print(f"\n{len(batches)} progs × {N_SAMPLES} = {sum(len(b['samples']) for b in batches)} images")
