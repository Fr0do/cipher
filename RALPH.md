# RALPH.md — CIPHER backlog grinder

Ralph mode tasks for CIPHER. Each item is independent — safe for autonomous execution.

## Active backlog

### Overnight results (check first!)
- [ ] **check-overnight**: Check `/tmp/cipher_overnight/results.json` and `/tmp/cipher_overnight_yellow_ocr.log`. The overnight script (PID 97453) generates 20 samples/level and runs OCR baseline. Yellow-OCR (PID 62302) evaluates with color-filtered OCR. Report: how many samples done, EM rates, disk usage.
- [ ] **compile-cal-table**: Compile multi-model calibration table from overnight results + previous Haiku cal3 data (`/tmp/cipher_cal3.json`). Include: OCR-only, Yellow-OCR, Haiku (cal3), Sonnet, target EM. Identify levels needing adjustment.

### S3 agent evaluation (GIVE VIDEO, NOT FRAMES)
- [ ] **eval-haiku-video**: For each sample in `/tmp/cipher_overnight/results.json`, spawn Haiku subagent with: scrambled_video path, nl_instructions, and Bash access. Agent runs ffmpeg to correct video, then extracts frames and finds yellow key. NO gold key or ops in prompt.
- [ ] **eval-sonnet-video**: Same as above but with Sonnet subagent.
- [ ] **eval-codex-video**: Spawn Swarm codex agent (effort=detailed) with same task.
- [ ] **eval-gemini-video**: Spawn Swarm gemini agent with same task.
- [ ] **eval-mcp-skill**: Run `solve_captcha.py --video <scrambled> --instructions <nl>` for each sample. This IS the MCP skill baseline.

### Benchmark generation
- [ ] **fetch-backgrounds**: run `bash scripts/fetch_backgrounds.sh`, verify ≥10 videos in `data/backgrounds/`
- [ ] **generate-full**: `python run_cipher.py generate --videos data/backgrounds/ --n 50 --levels L1 L2 L3 L4 L5`

### Paper
- [ ] **fill-results-table**: Fill sections/04_experiments.tex with calibration results
- [ ] **commit-runs**: `git add runs/ && git commit -m "[results] Add calibration results"`

## Stop conditions

- Max iterations: 8
- Empty queue
- Any ffmpeg error on >50% of samples (broken data, stop and report)
- API rate limit 3× in a row (stop, report remaining)

## Guards

- Do NOT delete data/ (large, slow to regenerate)
- Do NOT run training (`accelerate launch`, `torchrun`)
- Do NOT push runs/ with partial results (all levels must complete)
