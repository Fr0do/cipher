# autoresearch — CIPHER program-evolution engine

A scaffold combining three ideas for automated benchmark calibration:

- **GEPA** (arXiv:2507.19457) — reflective evolution of programs. Each "program"
  is a full genome for the captcha generator (alpha, font size, distractors,
  noise, blur, position policy). Mutation is *reflective*: a proposer LLM reads
  previous scores + confusion traces and writes the next genome.
- **DSPy** — typed LLM pipelines. The proposer, grader-reasoner, and
  knowledge-hub querent are DSPy `Predict` modules so prompts are optimizable
  via `BootstrapFewShot` / `MIPROv2` once we have enough trace data.
- **CORAL** (Human-Agent-Society/CORAL) — multi-agent code evolution with a
  shared `.coral/public/` state and a `TaskGrader`. We reuse the pattern: every
  round publishes `round_NN/{programs.json, batches.json, scores.json,
  reflection.md}` to a shared hub so multiple proposer agents can read prior
  context.

## Files

- `task.yaml` — grader target profile (per-level EM targets + weights).
- `engine.py` — round orchestrator: propose → generate → dispatch evals →
  grade → reflect → persist.
- `grader.py` — `TaskGrader` that scores a batch of programs against the
  ladder target profile.
- `proposer.py` — DSPy `Predict` that takes the last K rounds of
  (genome, scores, confusions) and emits the next round's genomes.
- `knowledge.py` — append-only JSONL knowledge hub under
  `eval/gepa_rounds/hub.jsonl` shared across rounds and across agents.

## Status

Scaffold only. `engine.py` wires the call graph but dispatch is still manual
(via `scripts/gepa_round.py` + Swarm MCP agents). See
`eval/gepa_rounds/cipher_gepa_r8/` for the current best ladder.
