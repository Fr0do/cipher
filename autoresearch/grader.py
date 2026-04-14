"""TaskGrader — scores a round of programs against the ladder target profile.

Input:  list of (genome, observed_em, observed_char_acc) per program.
Output: per-program fitness + round-level aggregate + human-readable reflection.

Fitness is squared-error vs target_em, with a char_acc floor penalty so a
program cannot be "correctly" calibrated by accident (e.g. 40% EM from random
guessing rather than readable-but-hard).
"""
from __future__ import annotations
import dataclasses
import yaml
from pathlib import Path


@dataclasses.dataclass
class LevelTarget:
    name: str
    target_em: float
    min_char_acc: float


@dataclasses.dataclass
class ProgramScore:
    level: str
    em: float
    char_acc: float
    fitness: float
    verdict: str


def load_task(path: str | Path) -> list[LevelTarget]:
    spec = yaml.safe_load(Path(path).read_text())
    return [LevelTarget(**lv) for lv in spec["levels"]]


def grade_program(level: LevelTarget, em: float, char_acc: float) -> ProgramScore:
    err = (em - level.target_em) ** 2
    floor_penalty = max(0.0, level.min_char_acc - char_acc)
    fitness = -(err + 0.5 * floor_penalty)
    if abs(em - level.target_em) < 0.05:
        verdict = "ON_TARGET"
    elif em > level.target_em + 0.10:
        verdict = "TOO_EASY"
    else:
        verdict = "TOO_HARD"
    return ProgramScore(level.name, em, char_acc, fitness, verdict)


def grade_round(targets: list[LevelTarget],
                observations: list[tuple[float, float]]) -> list[ProgramScore]:
    assert len(targets) == len(observations), "one observation per level"
    return [grade_program(t, em, ca) for t, (em, ca) in zip(targets, observations)]
