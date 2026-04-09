# RALPH.md — CIPHER backlog grinder

Ralph mode tasks for CIPHER. Each item is independent — safe for autonomous execution.

## Active backlog

- [ ] **fetch-backgrounds**: run `bash scripts/fetch_backgrounds.sh`, verify ≥10 videos in `data/backgrounds/`
- [ ] **generate-l1-l3**: `python run_cipher.py generate --videos data/backgrounds/ --n 50 --levels L1 L2 L3`
- [ ] **generate-l4-l5**: `python run_cipher.py generate --videos data/backgrounds/ --n 50 --levels L4 L5`
- [ ] **bench-ocr-only**: `python run_cipher.py bench --model ocr-only --manifest data/manifest.json`
- [ ] **bench-haiku**: `python run_cipher.py bench --model haiku --manifest data/manifest.json`
- [ ] **bench-gpt4o-mini**: `python run_cipher.py bench --model gpt-4o-mini --manifest data/manifest.json`
- [ ] **bench-gemini-flash**: `python run_cipher.py bench --model gemini-flash --manifest data/manifest.json`
- [ ] **commit-runs**: `git add runs/ && git commit -m "[results] Add <model> benchmark results"`

## Stop conditions

- Max iterations: 8
- Empty queue
- Any ffmpeg error on >50% of samples (broken data, stop and report)
- API rate limit 3× in a row (stop, report remaining)

## Guards

- Do NOT delete data/ (large, slow to regenerate)
- Do NOT run training (`accelerate launch`, `torchrun`)
- Do NOT push runs/ with partial results (all levels must complete)
