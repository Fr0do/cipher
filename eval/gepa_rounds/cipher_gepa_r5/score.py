#!/usr/bin/env python3
"""Score GEPA R5: Sonnet as reader (Gemini API down)."""
import json, os

work_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(work_dir, "batches.json")) as f:
    batches = json.load(f)

target_acc = 0.03
results = []

for b in batches:
    pi = b["program_idx"]
    pred_path = os.path.join(work_dir, f"pred_prog{pi}.json")
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

    r = {"program_idx": pi, "em": em, "char_acc": char_acc, "matches": matches,
         "n": n, "missing": missing, "fitness": fitness,
         "desc": b.get("desc", ""), "params": b["genome_summary"]}
    results.append(r)

    status = "NEAR TARGET" if abs(em - target_acc) <= 0.05 else ("TOO EASY" if em > 0.08 else "TOO HARD")
    print(f"[{pi}] EM={em:.0%} ({matches}/{n}) char_acc={char_acc:.0%} fit={fitness:.3f} | {status}")
    print(f"    {b.get('desc','')}")
    for sid in sorted(gold):
        g, p = gold[sid], preds.get(sid, "???")
        mark = "v" if p.strip().upper() == g.upper() else "x"
        print(f"     {sid}: gold={g} pred={p} [{mark}]")

print(f"\n{'='*60}")
best = max(results, key=lambda r: r["fitness"])
print(f"Best: prog_{best['program_idx']} fit={best['fitness']:.3f} EM={best['em']:.0%} char_acc={best['char_acc']:.0%}")
print(f"  {best['desc']}")

with open(os.path.join(work_dir, "scores.json"), "w") as f:
    json.dump(results, f, indent=2)
