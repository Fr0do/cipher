# CLAUDE.md — CIPHER

**CIPHER**: Captcha Instruction Parsing with Hidden Extraction and Reasoning  
NeurIPS benchmark for video-editing captcha solving by VLMs.

## Repo structure

```
cipher/
  run_cipher.py          # top-level CLI: generate / bench
  src/
    generator.py         # procedural sample generation (BackgroundPool, L1-L5)
    evaluator.py         # 3-stage metrics: S1 parse, S2 execute, S3 extract
    vlm_backends.py      # VLM connectors: haiku, gpt-4o, gemini, vllm, ocr-only
  scripts/
    fetch_backgrounds.sh # download Kinetics background clips via yt-dlp
  data/                  # generated samples (gitignored, large)
  runs/                  # benchmark results JSON (committed)
```

## Difficulty levels

| Level | key_vis | ops | nl_ambiguity | Notes |
|-------|---------|-----|--------------|-------|
| L1 | 1 (large) | 1 | exact | Sanity check |
| L2 | 2 | 2 | exact | Two-op chains |
| L3 | 3 | 2 | paraphrase | OCR harder |
| L4 | 3 | 3 | reversed | Instruction order reversed |
| L5 | 4 (small) | 4 | compositional | Full difficulty |

## Metrics

- **EM** — exact match on key (strict)
- **Char-F1** — character-level F1 (partial credit)
- **Pipeline-F1** — 0.25·S1(parse) + 0.25·S2(execute) + 0.50·S3(char-F1)

## Quickstart

```bash
# 1. Get backgrounds
bash scripts/fetch_backgrounds.sh

# 2. Generate dataset (L1-L5, 50 samples each)
python run_cipher.py generate --videos data/backgrounds/ --n 50

# 3. Benchmark
python run_cipher.py bench --model haiku gpt-4o-mini ocr-only
```

## Delegation policy

Follow CLAUDE.md conventions from ourosss:
- Code changes → codex Swarm agent
- Paper writing → gemini `effort=detailed` (long context)
- Planning / review → Opus (you)
- **Never** commit data/ (large videos). Runs/ results are committed.

## Target models (paper baselines)

| Model | Backend | Notes |
|-------|---------|-------|
| OCR-only | — | No VLM, easyocr |
| Claude Haiku | `haiku` | Fast, cheap |
| GPT-4o-mini | `gpt-4o-mini` | OpenAI |
| GPT-4o | `gpt-4o` | OpenAI |
| Gemini Flash | `gemini-flash` | Google |
| Qwen2-VL-7B | `qwen2-vl-7b` | Local, kurkin-vllm |
| Qwen2-VL-72B | `qwen2-vl-72b` | Local, kurkin-vllm |

## Paper

See `cipher-paper/` repo. Sections: intro, task, benchmark, experiments, analysis.
