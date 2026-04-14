"""
evaluator.py — benchmark runner and metrics for REVEL benchmark.

Pipeline has 3 scoreable stages:
  S1. Instruction parsing  — did the model identify the correct edit ops?
  S2. Edit execution       — did ffmpeg produce a valid non-trivial output?
  S3. Key extraction       — did the model recover the key string?

Final score:
  EM          = exact match on key (strict)
  Char-F1     = character-level F1 on key (partial credit S3)
  Pipeline-F1 = weighted mean of S1 + S2 + S3 scores
                weights: parse=0.25, execute=0.25, extract=0.50
"""
import json
import time
import sys
import os
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

# Locate solve_captcha: prefer CIPHER_SKILL_DIR env (container), fallback to ~/.hermes
_SKILL_SCRIPTS = Path(os.environ.get(
    "CIPHER_SKILL_DIR",
    str(Path(__file__).parent.parent.parent.parent /
        ".hermes" / "skills" / "media" / "video-captcha-solver" / "scripts")
))
sys.path.insert(0, str(_SKILL_SCRIPTS))


# ---------------------------------------------------------------------------
# Stage 1: op-parse F1
# ---------------------------------------------------------------------------

def _normalize_op(name: str) -> str:
    """Normalize op names so parse results match generator names.
    parse_instructions returns 'rotate' (generic); generator stores 'rotate90'/'rotate270'.
    Normalize both to 'rotate' for F1 comparison.
    """
    if name.startswith("rotate"):
        return "rotate"
    return name


def op_parse_f1(pred_ops: list[str], gold_ops: list[str]) -> float:
    """
    Sequence-aware F1: score each predicted op against gold at same position.
    Falls back to set-F1 if lengths differ.
    """
    if not pred_ops and not gold_ops:
        return 1.0
    if not pred_ops or not gold_ops:
        return 0.0

    pred_norm = [_normalize_op(p) for p in pred_ops]
    gold_norm = [_normalize_op(g) for g in gold_ops]

    # Ordered comparison (positional)
    correct = sum(p == g for p, g in zip(pred_norm, gold_norm))
    prec = correct / len(pred_norm)
    rec = correct / len(gold_norm)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


# ---------------------------------------------------------------------------
# Stage 3: key extraction metrics
# ---------------------------------------------------------------------------

def exact_match(pred: Optional[str], gold: str) -> bool:
    if not pred:
        return False
    return pred.strip().upper() == gold.strip().upper()


def char_f1(pred: Optional[str], gold: str) -> float:
    """Character-level F1 (partial credit for near-misses)."""
    if not pred:
        return 0.0
    p, g = list(pred.upper()), list(gold.upper())
    common = sum(min(p.count(c), g.count(c)) for c in set(g))
    if common == 0:
        return 0.0
    prec = common / len(p)
    rec = common / len(g)
    return 2 * prec * rec / (prec + rec)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Token / call tracking wrapper
# ---------------------------------------------------------------------------

class TrackedVlmFn:
    """Wraps a vlm_fn to count calls and tokens consumed."""

    def __init__(self, vlm_fn):
        self._fn = vlm_fn
        self.calls: int = 0
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.__name__ = getattr(vlm_fn, "__name__", "tracked")

    def __call__(self, prompt: str, images=None) -> str:
        self.calls += 1
        # Rough token estimate (4 chars ≈ 1 token) — replaced by SDK counts if available
        self.input_tokens += len(prompt) // 4 + (len(images or []) * 256)
        result = self._fn(prompt, images=images)
        self.output_tokens += len(result) // 4
        return result

    def reset(self):
        self.calls = self.input_tokens = self.output_tokens = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class SampleResult:
    sample_id: str
    level: str
    gold_key: str
    gold_ops: list[str]

    # Stage 1
    pred_ops: list[str] = field(default_factory=list)
    s1_parse_f1: float = 0.0

    # Stage 2
    s2_execute_ok: bool = False
    ffmpeg_calls: int = 0

    # Stage 3
    pred_key: str = ""
    s3_em: bool = False
    s3_char_f1: float = 0.0

    # Aggregate
    pipeline_f1: float = 0.0   # 0.25*S1 + 0.25*S2 + 0.50*S3_char_f1

    # Cost tracking
    vlm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    latency_s: float = 0.0
    error: str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def _pipeline_f1(s1: float, s2: bool, s3: float) -> float:
    return 0.25 * s1 + 0.25 * float(s2) + 0.50 * s3


# ---------------------------------------------------------------------------
# Level / overall stats
# ---------------------------------------------------------------------------

def compute_stats(results: list[SampleResult]) -> dict:
    if not results:
        return {}
    n = len(results)
    return {
        "n": n,
        "EM": round(sum(r.s3_em for r in results) / n, 4),
        "char_F1": round(sum(r.s3_char_f1 for r in results) / n, 4),
        "pipeline_F1": round(sum(r.pipeline_f1 for r in results) / n, 4),
        "S1_parse_F1": round(sum(r.s1_parse_f1 for r in results) / n, 4),
        "S2_execute_rate": round(sum(r.s2_execute_ok for r in results) / n, 4),
        "avg_latency_s": round(sum(r.latency_s for r in results) / n, 2),
        # Cost metrics
        "avg_vlm_calls": round(sum(r.vlm_calls for r in results) / n, 2),
        "avg_total_tokens": round(sum(r.total_tokens for r in results) / n, 0),
        "avg_ffmpeg_calls": round(sum(r.ffmpeg_calls for r in results) / n, 2),
    }


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def _run_one(sample: dict, vlm_fn=None) -> SampleResult:
    """Run the full pipeline on one sample, collecting per-stage scores."""
    from edit_ops import parse_instructions, build_ffmpeg_commands
    from extract_key import extract_key
    import subprocess, tempfile
    from pathlib import Path as P

    # Wrap vlm_fn for token/call tracking
    tracker = TrackedVlmFn(vlm_fn) if vlm_fn else None

    res = SampleResult(
        sample_id=sample["sample_id"],
        level=sample["level"],
        gold_key=sample["key"],
        gold_ops=sample["ops"],
    )
    t0 = time.time()

    try:
        # ---- Stage 1: parse ----
        ops = parse_instructions(sample["nl_instructions"], vlm_fn=tracker)
        res.pred_ops = [op["op"] for op in ops]
        res.s1_parse_f1 = op_parse_f1(res.pred_ops, res.gold_ops)

        if not ops:
            res.error = "parse_failed"
            return res

        # ---- Stage 2: execute ----
        with tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, "output.mp4")
            cmds, tmp_files = build_ffmpeg_commands(
                sample["scrambled_video"], ops, out_path)

            ok = True
            for cmd in cmds:
                res.ffmpeg_calls += 1
                r = subprocess.run(cmd, capture_output=True)
                if r.returncode != 0:
                    ok = False
                    res.error = f"ffmpeg: {r.stderr.decode()[-200:]}"
                    break

            if ok and P(out_path).exists() and P(out_path).stat().st_size > 1000:
                res.s2_execute_ok = True

                # ---- Stage 3: extract ----
                key = extract_key(out_path, vlm_fn=tracker)
                res.pred_key = key or ""
                res.s3_em = exact_match(key, sample["key"])
                res.s3_char_f1 = char_f1(key, sample["key"])
            else:
                res.error = res.error or "execute_failed"

    except Exception as e:
        res.error = str(e)

    res.latency_s = round(time.time() - t0, 2)
    res.pipeline_f1 = _pipeline_f1(res.s1_parse_f1, res.s2_execute_ok, res.s3_char_f1)

    if tracker:
        res.vlm_calls = tracker.calls
        res.input_tokens = tracker.input_tokens
        res.output_tokens = tracker.output_tokens

    return res


def run_benchmark(
    manifest_path: str,
    vlm_fn=None,
    levels: list[str] = None,
    max_samples: int = None,
    output_path: str = None,
) -> dict:
    with open(manifest_path) as f:
        manifest = json.load(f)

    all_results: dict[str, list[SampleResult]] = {}

    for level, samples in manifest["levels"].items():
        if levels and level not in levels:
            continue
        if max_samples:
            samples = samples[:max_samples]

        level_results = []
        print(f"\n=== {level} ({len(samples)} samples) ===")

        for s in samples:
            res = _run_one(s, vlm_fn=vlm_fn)
            level_results.append(res)

            mark = "✓" if res.s3_em else ("~" if res.s3_char_f1 > 0.5 else "✗")
            print(f"  {mark} {res.sample_id}: gold={res.gold_key!r:8s} "
                  f"pred={res.pred_key!r:8s} "
                  f"S1={res.s1_parse_f1:.2f} S2={'ok' if res.s2_execute_ok else 'no'} "
                  f"F1={res.s3_char_f1:.2f} pipe={res.pipeline_f1:.2f} "
                  f"t={res.latency_s:.1f}s"
                  + (f" [{res.error}]" if res.error else ""))

        all_results[level] = level_results

    # Aggregate
    report = {
        "model": getattr(vlm_fn, "__name__", "ocr_only"),
        "per_level": {lv: compute_stats(rs) for lv, rs in all_results.items()},
        "overall": compute_stats([r for rs in all_results.values() for r in rs]),
        "raw": {lv: [asdict(r) for r in rs] for lv, rs in all_results.items()},
    }

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport → {output_path}")

    # Summary table
    print(f"\n{'Level':<6} {'EM':>6} {'CharF1':>7} {'PipeF1':>7} {'S1':>6} {'S2':>6} {'Lat':>6}")
    print("-" * 52)
    for lv, st in report["per_level"].items():
        if not st:
            continue
        print(f"{lv:<6} {st['EM']:>6.1%} {st['char_F1']:>7.3f} "
              f"{st['pipeline_F1']:>7.3f} {st['S1_parse_F1']:>6.3f} "
              f"{st['S2_execute_rate']:>6.1%} {st['avg_latency_s']:>6.1f}s")
    ov = report["overall"]
    print("-" * 52)
    print(f"{'TOTAL':<6} {ov['EM']:>6.1%} {ov['char_F1']:>7.3f} "
          f"{ov['pipeline_F1']:>7.3f} {ov['S1_parse_F1']:>6.3f} "
          f"{ov['S2_execute_rate']:>6.1%} {ov['avg_latency_s']:>6.1f}s")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run REVEL benchmark")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--levels", nargs="+")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output", default="runs/results.json")
    parser.add_argument("--vlm-endpoint", default="http://localhost:8000/v1")
    parser.add_argument("--vlm-model", default="qwen2-vl-7b-instruct")
    parser.add_argument("--no-vlm", action="store_true")
    args = parser.parse_args()

    vlm_fn = None
    if not args.no_vlm:
        from solve_captcha import make_vlm_fn
        vlm_fn = make_vlm_fn(args.vlm_endpoint, args.vlm_model)

    run_benchmark(
        manifest_path=args.manifest,
        vlm_fn=vlm_fn,
        levels=args.levels,
        max_samples=args.max_samples,
        output_path=args.output,
    )
