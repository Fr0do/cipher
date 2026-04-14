#!/usr/bin/env python3
"""Generate Round 3 GEPA images — interpolating between R2 prog_0 (0% near-miss) and prog_1 (100%)."""
import json, os, sys
sys.path.insert(0, "/Users/mkurkin/experiments/cipher/scripts")
from captcha_dsl import generate_image_sample

work_dir = "/tmp/cipher_gepa_r3"
os.makedirs(work_dir, exist_ok=True)

# Round 3 programs: sweet spot search between prog_0 (α=130,font=4.5%,3dist) and prog_1 (α=180,font=6%,4dist,no noise)
# Key insight from R2: prog_0 has near-miss char confusion at 0%. Need slightly easier.
programs = [
    {
        "program_idx": 0,
        "desc": "prog_0 + bump alpha to 150, font to 5%",
        "program": "(seq (overlay-key :alpha 150 :font-pct 0.05 :position edge :key-len 8) (scramble reverse hflip) (distractor :n 3 :yellow 2 :same-length true) (noise :sigma 3) (blur :radius 0.2) (shadow true))",
        "genome": {"alpha": 150, "fontsize_pct": 0.05, "key_length": 8, "n_distractors": 3, "yellow_distractors": 2, "noise_sigma": 3.0, "blur_radius": 0.2, "shadow": True, "position_mode": "edge", "distractor_same_length": True},
    },
    {
        "program_idx": 1,
        "desc": "α=160, font=5.5%, 2 distractors, minimal noise",
        "program": "(seq (overlay-key :alpha 160 :font-pct 0.055 :position corner :key-len 7) (scramble reverse) (distractor :n 2 :yellow 1 :same-length true) (noise :sigma 2) (blur :radius 0.1) (shadow true))",
        "genome": {"alpha": 160, "fontsize_pct": 0.055, "key_length": 7, "n_distractors": 2, "yellow_distractors": 1, "noise_sigma": 2.0, "blur_radius": 0.1, "shadow": True, "position_mode": "corner", "distractor_same_length": True},
    },
    {
        "program_idx": 2,
        "desc": "α=140, font=5%, 4 yellow distractors, no noise — distractor confusion test",
        "program": "(seq (overlay-key :alpha 140 :font-pct 0.05 :position random :key-len 8) (distractor :n 4 :yellow 4 :same-length true) (shadow true))",
        "genome": {"alpha": 140, "fontsize_pct": 0.05, "key_length": 8, "n_distractors": 4, "yellow_distractors": 4, "noise_sigma": 0.0, "blur_radius": 0.0, "shadow": True, "position_mode": "random", "distractor_same_length": True},
    },
    {
        "program_idx": 3,
        "desc": "α=145, font=4.8%, 3 dist (2 yellow), noise=4, blur=0.25 — fine-tuned mid",
        "program": "(seq (overlay-key :alpha 145 :font-pct 0.048 :position edge :key-len 8) (scramble reverse hflip) (distractor :n 3 :yellow 2 :same-length true) (noise :sigma 4) (blur :radius 0.25) (shadow true))",
        "genome": {"alpha": 145, "fontsize_pct": 0.048, "key_length": 8, "n_distractors": 3, "yellow_distractors": 2, "noise_sigma": 4.0, "blur_radius": 0.25, "shadow": True, "position_mode": "edge", "distractor_same_length": True},
    },
    {
        "program_idx": 4,
        "desc": "α=155, font=5.2%, 3 dist (3 yellow), noise=2 — all-yellow distractor stress",
        "program": "(seq (overlay-key :alpha 155 :font-pct 0.052 :position corner :key-len 7) (scramble reverse) (distractor :n 3 :yellow 3 :same-length true) (noise :sigma 2) (blur :radius 0.15) (shadow true))",
        "genome": {"alpha": 155, "fontsize_pct": 0.052, "key_length": 7, "n_distractors": 3, "yellow_distractors": 3, "noise_sigma": 2.0, "blur_radius": 0.15, "shadow": True, "position_mode": "corner", "distractor_same_length": True},
    },
]

batches = []
seed_base = 10000

for prog in programs:
    pi = prog["program_idx"]
    out_dir = os.path.join(work_dir, f"prog_{pi}")
    os.makedirs(out_dir, exist_ok=True)

    samples = []
    for j in range(4):
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
        "program": prog["program"],
        "desc": prog["desc"],
        "samples": samples,
        "genome_summary": {k: v for k, v in prog["genome"].items() if k not in ("shadow", "position_mode", "distractor_same_length")},
    })

with open(os.path.join(work_dir, "batches.json"), "w") as f:
    json.dump(batches, f, indent=2)

print(f"\nGenerated {len(batches)} programs x 4 samples = {sum(len(b['samples']) for b in batches)} images")
print(f"Saved to {work_dir}/batches.json")
