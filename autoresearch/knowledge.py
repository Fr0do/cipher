"""Shared knowledge hub — append-only JSONL of round outcomes.

CORAL-style: every agent/round writes one JSON line, and future rounds read
the tail of the file for context. Keeps evolution state out of git tree
churn (scores.json per round) and into a single streamable log.
"""
from __future__ import annotations
import json
import time
from pathlib import Path


def append_round(hub_path: str | Path, round_idx: int,
                 programs: list[dict], scores: list[dict],
                 reflection: str) -> None:
    entry = {
        "ts": time.time(),
        "round": round_idx,
        "programs": programs,
        "scores": scores,
        "reflection": reflection,
    }
    p = Path(hub_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def read_tail(hub_path: str | Path, k: int = 3) -> list[dict]:
    p = Path(hub_path)
    if not p.exists():
        return []
    lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines[-k:]]
