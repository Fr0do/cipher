#!/usr/bin/env python3
"""
hybrid_optimize.py — GEPA (program evolution) + Nevergrad (param tuning).

Outer loop (GEPA): evolves captcha programs as s-expressions via LLM reflection.
Inner loop (Nevergrad): tunes numeric $-params within each program.

Evaluation: Swarm agents (gemini/codex/claude) or direct API.

Architecture:
  GEPA proposes program_str → compile → CaptchaSpec (with $-params)
  For each program, Nevergrad optimizes $-params:
    candidate values → generate N samples → run VLM → fitness
  Best (program, params) fitness returned to GEPA as metric + feedback.
  GEPA reflects on traces and proposes better programs.

Usage:
  # Image-only mode (fast iteration, no video):
  python scripts/hybrid_optimize.py run --mode image \
      --background data/backgrounds/ --target-acc 0.03

  # Dry run (heuristic fitness, no VLM needed):
  python scripts/hybrid_optimize.py dry-run --ng-budget 30
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import nevergrad as ng

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from captcha_dsl import (
    compile_program, spec_to_genome_dict, unparse, parse_program,
    CaptchaSpec, SEED_PROGRAMS, NG_PARAM_RANGES,
)


# ===================================================================
# Nevergrad inner loop: tune $-params for a fixed program
# ===================================================================

def make_ng_parametrization(spec: CaptchaSpec) -> ng.p.Dict | None:
    """Build Nevergrad parametrization from spec's $-params."""
    if not spec.ng_params:
        return None

    params = {}
    for name, (lo, hi) in spec.ng_params.items():
        if isinstance(lo, int) and isinstance(hi, int):
            params[name] = ng.p.Scalar(lower=lo, upper=hi).set_integer_casting()
        else:
            params[name] = ng.p.Scalar(lower=float(lo), upper=float(hi))

    return ng.p.Dict(**params)


def ng_optimize(
    program_str: str,
    eval_fn,
    budget: int = 20,
    n_samples_per_eval: int = 3,
) -> tuple[dict, float, str]:
    """Run Nevergrad to optimize $-params within a program.

    Args:
        program_str: the captcha program (s-expression)
        eval_fn: callable(genome_dict) -> (fitness: float, feedback: str)
        budget: Nevergrad evaluation budget
        n_samples_per_eval: samples per fitness evaluation

    Returns:
        (best_ng_values, best_fitness, feedback_text)
    """
    spec = compile_program(program_str)
    parametrization = make_ng_parametrization(spec)

    if parametrization is None:
        # No $-params, evaluate once with defaults
        genome_dict = spec_to_genome_dict(spec)
        fitness, feedback = eval_fn(genome_dict)
        return {}, fitness, feedback

    optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=budget)

    best_fitness = -999.0
    best_values = {}
    best_feedback = ""

    for _ in range(budget):
        candidate = optimizer.ask()
        ng_values = candidate.value

        genome_dict = spec_to_genome_dict(spec, ng_values)
        fitness, feedback = eval_fn(genome_dict)

        # Nevergrad minimizes, we maximize fitness -> negate
        optimizer.tell(candidate, -fitness)

        if fitness > best_fitness:
            best_fitness = fitness
            best_values = dict(ng_values)
            best_feedback = feedback

    return best_values, best_fitness, best_feedback


# ===================================================================
# DSPy/GEPA integration
# ===================================================================

def run_gepa(
    eval_fn_factory,
    target_acc: float = 0.03,
    gepa_budget: int = 15,
    ng_budget: int = 20,
    n_samples: int = 3,
    output_dir: str = "runs/hybrid",
    seed: int = 42,
    reflection_model: str = "claude-sonnet-4-20250514",
):
    """Run GEPA outer loop with Nevergrad inner loop."""
    import dspy
    from dspy import GEPA

    os.makedirs(output_dir, exist_ok=True)

    # GEPA evolves the program text
    class CaptchaDesigner(dspy.Signature):
        """Design a captcha program as an s-expression that achieves target
        VLM accuracy (2-5%). The program specifies key overlay parameters,
        scramble operations, distractors, and optional sub-captcha gates.

        Available primitives:
          (seq EXPR ...)              -- sequence of expressions
          (overlay-key K V ...)       -- key text: :alpha (25-255), :font-pct (0.015-0.15),
                                        :dur (0.08-5.0), :position (center/random/corner/edge),
                                        :flash-hz (0-10), :key-len (5-12), :font (0-3)
          (scramble OP ...)           -- ffmpeg ops: reverse, hflip, vflip, rotate90, rotate270, speed2x
          (distractor K V ...)        -- distractors: :n (0-10), :yellow (0-6), :same-length (true/false)
          (gate (text-captcha EXPR))  -- sub-captcha: math/text puzzle
          (noise :sigma N)            -- gaussian noise (0-40)
          (blur :radius N)            -- text blur (0-3)
          (nl-ambiguity N)            -- instruction ambiguity (1=exact, 2=paraphrase, 3=reversed)
          (shadow true/false)         -- drop shadow

        Use $var_name for numeric params that Nevergrad should optimize:
          (overlay-key :alpha $alpha :font-pct $font_pct)

        The goal: VLM accuracy should be near the target (2-5%), not 0% and not 100%.
        """
        task_description: dspy.InputField = dspy.InputField(
            desc="Description of what to optimize")
        previous_feedback: dspy.InputField = dspy.InputField(
            desc="Feedback from previous evaluation attempts")
        captcha_program: dspy.OutputField = dspy.OutputField(
            desc="S-expression captcha program")

    designer = dspy.ChainOfThought(CaptchaDesigner)

    # Metric for GEPA
    def metric(gold, pred, trace=None, **kwargs):
        from dspy.primitives.assertions import ScoreWithFeedback
        program_str = pred.captcha_program

        # Validate parse
        try:
            spec = compile_program(program_str)
        except Exception as e:
            return ScoreWithFeedback(
                score=0.0,
                feedback=f"PARSE ERROR: {e}. Fix the s-expression syntax.")

        # Run Nevergrad inner loop
        best_values, fitness, feedback = ng_optimize(
            program_str=program_str,
            eval_fn=eval_fn_factory,
            budget=ng_budget,
            n_samples_per_eval=n_samples,
        )

        # Normalize fitness to [0, 1] for GEPA (0 = worst, 1 = perfect target match)
        score = max(0.0, 1.0 + fitness)

        full_feedback = (
            f"Program: {program_str[:100]}...\n"
            f"Best NG params: {best_values}\n"
            f"Fitness: {fitness:.4f}\n"
            f"Detail: {feedback}"
        )

        # Log
        log_entry = {
            "program": program_str,
            "best_values": best_values,
            "fitness": fitness,
            "score": score,
            "feedback": feedback,
        }
        log_path = os.path.join(output_dir, "gepa_trace.jsonl")
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry, default=str) + "\n")

        return ScoreWithFeedback(score=score, feedback=full_feedback)

    # Build training examples from seed programs
    train_examples = []
    for i, prog in enumerate(SEED_PROGRAMS):
        train_examples.append(dspy.Example(
            task_description=f"Design captcha program targeting {target_acc:.0%} VLM accuracy",
            previous_feedback=f"Seed program {i} -- no feedback yet",
            captcha_program=prog,
        ).with_inputs("task_description", "previous_feedback"))

    # Configure GEPA
    reflection_lm = dspy.LM(model=reflection_model, temperature=1.0, max_tokens=4096)

    optimizer = GEPA(
        metric=metric,
        auto="light",
        num_threads=1,
        track_stats=True,
        reflection_lm=reflection_lm,
    )

    print(f"Starting GEPA optimization:")
    print(f"  Target accuracy: {target_acc:.0%}")
    print(f"  GEPA budget: {gepa_budget} programs")
    print(f"  Nevergrad budget: {ng_budget} per program")
    print(f"  Samples per fitness: {n_samples}")
    print(f"  Reflection model: {reflection_model}")
    print(f"  Output: {output_dir}")

    optimized = optimizer.compile(
        designer,
        trainset=train_examples,
    )

    # Save results
    results = {
        "target_acc": target_acc,
        "gepa_budget": gepa_budget,
        "ng_budget": ng_budget,
        "reflection_model": reflection_model,
        "seed_programs": SEED_PROGRAMS,
    }
    with open(os.path.join(output_dir, "gepa_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved: {output_dir}/gepa_results.json")
    return optimized


# ===================================================================
# Heuristic eval (no VLM — estimates difficulty from params)
# ===================================================================

def make_heuristic_eval(target_acc: float):
    """Heuristic fitness: estimates VLM accuracy from watermark params.
    Useful for dry-run testing without API calls.
    """
    def heuristic_fn(genome_dict: dict) -> tuple[float, str]:
        alpha = genome_dict.get("alpha", 180)
        font = genome_dict.get("fontsize_pct", 0.06)
        blur = genome_dict.get("blur_radius", 0.0)
        noise = genome_dict.get("noise_sigma", 0.0)
        n_dist = genome_dict.get("n_distractors", 0)
        dur = genome_dict.get("key_duration_s", 2.0)
        flash = genome_dict.get("key_flash_hz", 0.0)

        # Readability model: higher = easier to read
        vis = (alpha / 255) * (font / 0.12) * max(0.05, 1.0 - blur / 3.0)
        vis *= max(0.1, 1.0 - noise / 50.0)
        vis *= min(1.0, dur / 2.0)  # very short = hard
        if flash > 0:
            vis *= max(0.2, 1.0 - flash / 12.0)  # flashing = harder

        # Distractor confusion
        confusion = n_dist * 0.04 + genome_dict.get("yellow_distractors", 0) * 0.06

        estimated_acc = max(0, min(1, vis - confusion))
        fitness = -abs(estimated_acc - target_acc)

        feedback = (f"[heuristic] est_acc={estimated_acc:.2f} "
                   f"vis={vis:.2f} confusion={confusion:.2f} "
                   f"alpha={alpha} font={font:.3f} blur={blur:.1f} "
                   f"noise={noise:.0f} dist={n_dist} dur={dur:.1f}s")

        if estimated_acc > target_acc + 0.05:
            feedback += " | TOO EASY: reduce alpha/font, add blur/noise/distractors"
        elif estimated_acc < max(0, target_acc - 0.02):
            feedback += " | TOO HARD: increase alpha/font, reduce blur/noise"
        else:
            feedback += " | NEAR TARGET"

        return fitness, feedback

    return heuristic_fn


def make_image_eval(backgrounds_dir: str, vlm_backends: list[str],
                    target_acc: float, n_samples: int = 3):
    """Create image-mode multi-model VLM fitness function. No ffmpeg — pure PIL.

    Supports two modes:
      - API backends (haiku, gpt-4o-mini, etc.) — direct call, needs API key
      - Swarm backends (swarm:codex, swarm:gemini, swarm:claude) — no API key needed,
        uses Swarm MCP subscriptions

    Each sample is evaluated round-robin across models.
    """
    from captcha_dsl import generate_image_sample, load_bg_images

    bg_images = load_bg_images(backgrounds_dir) if os.path.isdir(backgrounds_dir) else []
    print(f"Loaded {len(bg_images)} background images from {backgrounds_dir}")

    # Separate swarm vs API backends
    swarm_backends = []
    api_fns = {}
    for b in vlm_backends:
        name = b.replace("api:", "")
        if name.startswith("swarm:"):
            swarm_backends.append(name.replace("swarm:", ""))
            print(f"  Swarm backend: {name}")
        else:
            from vlm_backends import get_backend
            api_fns[name] = get_backend(name)
            print(f"  API backend: {name}")

    all_backends = list(api_fns.keys()) + [f"swarm:{s}" for s in swarm_backends]
    call_count = [0]

    def image_eval_fn(genome_dict: dict) -> tuple[float, str]:
        call_count[0] += 1

        # Generate all samples first (fast — ~34ms each)
        samples = []
        work_dir = f"/tmp/cipher_hybrid/batch_{call_count[0]:04d}"
        for i in range(n_samples):
            seed = call_count[0] * 1000 + i
            bg = bg_images[seed % len(bg_images)] if bg_images else None
            sample = generate_image_sample(seed, genome_dict, bg_image=bg,
                                           output_dir=work_dir)
            samples.append(sample)

        # Write batch manifest for Swarm agents
        manifest = {
            "batch_id": call_count[0],
            "samples": [
                {"sample_id": s["sample_id"],
                 "image_path": s["image_path"],
                 "gold_key": s["gold_key"]}
                for s in samples
            ],
        }
        manifest_path = os.path.join(work_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # Write predictions file path
        pred_path = os.path.join(work_dir, "predictions.json")

        results = []
        for i, sample in enumerate(samples):
            backend = all_backends[i % len(all_backends)]

            if backend.startswith("swarm:"):
                # Swarm eval: write per-sample task file, agent reads image + writes prediction
                agent_type = backend.replace("swarm:", "")
                task_path = os.path.join(work_dir, f"task_{sample['sample_id']}.json")
                with open(task_path, "w") as f:
                    json.dump({
                        "sample_id": sample["sample_id"],
                        "image_path": sample["image_path"],
                        "pred_path": os.path.join(work_dir, f"pred_{sample['sample_id']}.json"),
                        "prompt": (
                            f"Read the image at {sample['image_path']}. "
                            "It contains a short alphanumeric code (5-12 chars) as colored text. "
                            "Write ONLY the code to the pred_path file as JSON: "
                            '{\"pred_key\": \"THE_CODE\"}'
                        ),
                        "agent_type": agent_type,
                    }, f)
                # Mark as pending — orchestrator will spawn Swarm agents
                results.append({
                    "em": False, "pred": "", "gold": sample["gold_key"],
                    "model": backend, "pending": True,
                    "task_path": task_path,
                })
            else:
                # Direct API eval
                try:
                    from PIL import Image
                    img = Image.open(sample["image_path"])
                    prompt = (
                        "This image contains a short alphanumeric code "
                        "(5-12 characters) rendered as colored text overlaid on a background. "
                        "The code may be small, faint, blurred, or partially hidden. "
                        "Return ONLY the code text, nothing else. If none found, return NONE."
                    )
                    pred = api_fns[backend](prompt, images=[img]).strip().upper()
                    if pred == "NONE":
                        pred = ""
                    gold = sample["gold_key"]
                    em = pred == gold.upper()
                    results.append({"em": em, "pred": pred, "gold": gold, "model": backend})
                except Exception as e:
                    results.append({"em": False, "error": str(e), "model": backend})

        # Check for Swarm prediction files that may have been written
        for r in results:
            if r.get("pending"):
                sid = r["gold"]  # gold_key used temporarily
                # Try to read prediction from Swarm agent
                pred_file = os.path.join(work_dir, f"pred_{sid}.json")
                if os.path.exists(pred_file):
                    try:
                        with open(pred_file) as f:
                            p = json.load(f)
                        r["pred"] = p.get("pred_key", "").upper()
                        r["em"] = r["pred"] == r["gold"].upper()
                        r["pending"] = False
                    except Exception:
                        pass

        n = len(results)
        pending = sum(1 for r in results if r.get("pending"))
        em_rate = sum(r["em"] for r in results) / max(n, 1)
        errors = sum(1 for r in results if "error" in r)
        fitness = -abs(em_rate - target_acc) - 0.5 * (errors / max(n, 1))
        if pending:
            fitness -= 0.3 * (pending / n)  # penalty for unevaluated samples

        # Per-model breakdown
        by_model = {}
        for r in results:
            m = r.get("model", "?")
            by_model.setdefault(m, []).append(r)

        parts = [f"EM={em_rate:.0%} ({sum(r['em'] for r in results)}/{n})"]
        for m, mrs in by_model.items():
            m_em = sum(r["em"] for r in mrs) / max(len(mrs), 1)
            parts.append(f"{m}={m_em:.0%}")

        if pending:
            parts.append(f"PENDING={pending}")
        if em_rate > target_acc + 0.05:
            parts.append("TOO EASY: reduce alpha/font, add blur/noise/distractors")
        elif em_rate < max(0, target_acc - 0.02):
            parts.append("TOO HARD: increase alpha/font, reduce blur/noise")
        else:
            parts.append("NEAR TARGET")
        if errors:
            parts.append(f"ERRORS={errors}")
        for r in results[:4]:
            if "error" not in r and not r.get("pending"):
                parts.append(f"  [{r['model']}] pred='{r.get('pred','')}' gold='{r.get('gold','')}'")

        return fitness, "; ".join(parts)

    return image_eval_fn


# ===================================================================
# CLI
# ===================================================================

def cmd_run(args):
    """Run hybrid GEPA+Nevergrad optimization."""
    if args.eval_backend == ["none"] or args.eval_backend == "none":
        eval_fn = make_heuristic_eval(args.target_acc)
    else:
        backends = args.eval_backend if isinstance(args.eval_backend, list) else [args.eval_backend]
        eval_fn = make_image_eval(
            backgrounds_dir=args.background,
            vlm_backends=backends,
            target_acc=args.target_acc,
            n_samples=args.n_samples,
        )

    run_gepa(
        eval_fn_factory=eval_fn,
        target_acc=args.target_acc,
        gepa_budget=args.gepa_budget,
        ng_budget=args.ng_budget,
        n_samples=args.n_samples,
        output_dir=args.output,
        seed=args.seed,
        reflection_model=args.reflection_model,
    )


def cmd_dry_run(args):
    """Dry run: test DSL + Nevergrad without VLM (heuristic fitness)."""
    eval_fn = make_heuristic_eval(args.target_acc)

    print("Dry run: testing Nevergrad inner loop with heuristic fitness\n")
    for i, prog in enumerate(SEED_PROGRAMS):
        print(f"{'='*60}")
        print(f"Program {i}: {prog[:80]}...")
        best_values, fitness, feedback = ng_optimize(
            program_str=prog,
            eval_fn=eval_fn,
            budget=args.ng_budget,
        )
        print(f"  Best fitness: {fitness:.4f}")
        if best_values:
            print(f"  Best params: {json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in best_values.items()})}")
        print(f"  Feedback: {feedback}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid GEPA+Nevergrad captcha optimizer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # run
    p = sub.add_parser("run", help="Run hybrid optimization")
    p.add_argument("--background", default="data/backgrounds/")
    p.add_argument("--eval-backend", nargs="+", default=["haiku"],
                    help="VLM backends (haiku gpt-5.4-mini, etc.)")
    p.add_argument("--target-acc", type=float, default=0.03)
    p.add_argument("--gepa-budget", type=int, default=15)
    p.add_argument("--ng-budget", type=int, default=20)
    p.add_argument("--n-samples", type=int, default=3)
    p.add_argument("--output", default="runs/hybrid")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reflection-model", default="claude-sonnet-4-20250514")

    # dry-run
    p = sub.add_parser("dry-run", help="Test without VLM (heuristic fitness)")
    p.add_argument("--target-acc", type=float, default=0.03)
    p.add_argument("--ng-budget", type=int, default=30)

    args = parser.parse_args()
    {"run": cmd_run, "dry-run": cmd_dry_run}[args.cmd](args)


if __name__ == "__main__":
    main()
