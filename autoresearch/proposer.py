"""Proposer — DSPy Predict module that emits the next round's genomes.

Signature (typed DSPy): given the last K rounds of (genome, score, verdict,
confusion_pairs), emit N genomes — one per ladder level — that should move
each level's EM closer to its target.

Left as a thin skeleton: the real work is (a) shaping the signature for
BootstrapFewShot optimization, (b) wiring a small reflection LM, (c) handing
the output to engine.py which calls captcha_dsl + Swarm for evaluation.
"""
from __future__ import annotations
import json
from pathlib import Path

try:
    import dspy
except ImportError:  # keep module importable without dspy for now
    dspy = None


if dspy is not None:
    class LadderProposer(dspy.Signature):
        """Propose next-round genomes for a 5-level captcha-difficulty ladder.

        Given the history of (genome, observed_em, observed_char_acc, verdict,
        confusion_pairs) for each level, output the next genome per level that
        should push EM toward its target.
        """

        task_spec: str = dspy.InputField(desc="YAML with per-level EM targets")
        history: str = dspy.InputField(desc="JSON list of recent rounds")
        proposals: str = dspy.OutputField(desc="JSON list of 5 genome dicts")

    def make_proposer() -> "dspy.Predict":
        return dspy.Predict(LadderProposer)
else:
    def make_proposer():  # noqa: D401
        raise ImportError("dspy is required for LadderProposer; `pip install dspy-ai`")


def load_history(hub_path: str | Path, k: int = 3) -> list[dict]:
    """Read last K rounds from the shared knowledge hub (JSONL)."""
    p = Path(hub_path)
    if not p.exists():
        return []
    rounds: list[dict] = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        rounds.append(json.loads(line))
    return rounds[-k:]
