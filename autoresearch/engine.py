"""Round orchestrator.

Wires together: proposer (DSPy) -> generator (scripts/captcha_dsl) ->
external eval (Swarm MCP or local VLM) -> grader -> knowledge hub.

Dispatch to VLMs is *not* in-process yet: we emit a programs.json for the
round and expect the caller to invoke `scripts/gepa_round.py` + the usual
Swarm agents. Once scored, `ingest_round()` writes the round into the hub
and prepares the history string for the next proposer call.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from autoresearch import grader, knowledge, proposer


def plan_next_round(task_path: str, hub_path: str, out_path: str, k_hist: int = 3) -> None:
    targets = grader.load_task(task_path)
    history = knowledge.read_tail(hub_path, k=k_hist)
    if proposer.dspy is None or not history:
        # Cold start / no DSPy: seed with the current R8 ladder.
        seed = Path(__file__).parent.parent / "eval/gepa_rounds/cipher_gepa_r8/programs.json"
        Path(out_path).write_text(seed.read_text())
        print(f"[plan] cold-start, copied {seed} -> {out_path}")
        return
    pred = proposer.make_proposer()
    out = pred(
        task_spec=Path(task_path).read_text(),
        history=json.dumps(history),
    )
    Path(out_path).write_text(out.proposals)
    print(f"[plan] wrote proposals for round -> {out_path}")


def ingest_round(task_path: str, hub_path: str, round_idx: int,
                 programs_path: str, scores_path: str) -> None:
    targets = grader.load_task(task_path)
    programs = json.loads(Path(programs_path).read_text())
    scores_raw = json.loads(Path(scores_path).read_text())
    obs = [(s["em"], s["char_acc"]) for s in scores_raw]
    scored = grader.grade_round(targets, obs)
    reflection_lines = [f"Round {round_idx}:"]
    for lv, s in zip(targets, scored):
        reflection_lines.append(
            f"  {lv.name}: EM={s.em:.2f} (tgt {lv.target_em:.2f}) "
            f"char_acc={s.char_acc:.2f} fit={s.fitness:.3f} -> {s.verdict}"
        )
    reflection = "\n".join(reflection_lines)
    knowledge.append_round(hub_path, round_idx, programs,
                           [s.__dict__ for s in scored], reflection)
    print(reflection)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    plan = sub.add_parser("plan")
    plan.add_argument("--task", required=True)
    plan.add_argument("--hub", required=True)
    plan.add_argument("--out", required=True)

    ing = sub.add_parser("ingest")
    ing.add_argument("--task", required=True)
    ing.add_argument("--hub", required=True)
    ing.add_argument("--round", type=int, required=True)
    ing.add_argument("--programs", required=True)
    ing.add_argument("--scores", required=True)

    args = p.parse_args()
    if args.cmd == "plan":
        plan_next_round(args.task, args.hub, args.out)
    else:
        ingest_round(args.task, args.hub, args.round, args.programs, args.scores)


if __name__ == "__main__":
    main()
