#!/usr/bin/env python3
"""Run prompt optimization on the ManuscriptTranscriber module.

Supports MIPROv2 (text-only proposer) and GEPA (multimodal proposer).

Usage:
    python -m scripts.dspy_pipeline.optimize                              # MIPROv2 light
    python -m scripts.dspy_pipeline.optimize --optimizer gepa --auto medium  # GEPA medium
    python -m scripts.dspy_pipeline.optimize --auto light --limit 5       # quick test
"""

import argparse

import dspy
from dspy.teleprompt import MIPROv2

from .baseline import build_examples
from .config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    MAX_BOOTSTRAPPED_DEMOS,
    MAX_LABELED_DEMOS,
    OPTIMIZATION_DIR,
    PROJECT_ROOT,
)
from .dataset import load_optimization_dataset, split_train_val
from .metric import cer_metric
from .transcriber import ManuscriptTranscriber


def _run_mipro(program, train_examples, val_examples, args):
    """Run MIPROv2 optimization."""
    optimizer = MIPROv2(metric=cer_metric, auto=args.auto)
    optimized = optimizer.compile(
        program,
        trainset=train_examples,
        valset=val_examples,
        max_bootstrapped_demos=args.max_bootstrapped_demos,
        max_labeled_demos=args.max_labeled_demos,
        data_aware_proposer=False,  # proposer cannot summarize image datasets
    )
    return optimized


def _run_gepa(program, train_examples, val_examples, args):
    """Run GEPA optimization with MultiModalInstructionProposer."""
    from dspy.teleprompt.gepa.instruction_proposal import (
        MultiModalInstructionProposer,
    )

    reflection_lm = dspy.LM(
        args.model, temperature=1.0, max_tokens=DEFAULT_MAX_TOKENS * 4
    )
    optimizer = dspy.GEPA(
        metric=cer_metric,
        instruction_proposer=MultiModalInstructionProposer(),
        reflection_lm=reflection_lm,
        auto=args.auto,
        track_stats=True,
    )
    optimized = optimizer.compile(
        program, trainset=train_examples, valset=val_examples
    )
    return optimized


def main():
    parser = argparse.ArgumentParser(description="Run prompt optimization")
    parser.add_argument(
        "--optimizer",
        choices=["mipro", "gepa"],
        default="mipro",
        help="Optimizer: mipro (text-only) or gepa (multimodal)",
    )
    parser.add_argument(
        "--auto",
        choices=["light", "medium", "heavy"],
        default="light",
        help="Search depth",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit val pages (0=all)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LM model string")
    parser.add_argument(
        "--max-bootstrapped-demos",
        type=int,
        default=MAX_BOOTSTRAPPED_DEMOS,
        help="Max bootstrapped few-shot demos (MIPROv2 only)",
    )
    parser.add_argument(
        "--max-labeled-demos",
        type=int,
        default=MAX_LABELED_DEMOS,
        help="Max labeled few-shot demos (MIPROv2 only)",
    )
    args = parser.parse_args()

    # Configure DSPy
    lm = dspy.LM(args.model, max_tokens=DEFAULT_MAX_TOKENS)
    dspy.configure(lm=lm)

    # Load dataset
    entries = load_optimization_dataset()
    train, val = split_train_val(entries)

    train_examples = build_examples(train)
    val_examples = build_examples(val)

    if args.limit:
        val_examples = val_examples[: args.limit]

    print(f"Train: {len(train_examples)} examples")
    print(f"Val:   {len(val_examples)} examples")
    print(f"Optimizer: {args.optimizer} auto={args.auto}")

    # Optimize
    program = ManuscriptTranscriber()

    if args.optimizer == "gepa":
        optimized = _run_gepa(program, train_examples, val_examples, args)
    else:
        optimized = _run_mipro(program, train_examples, val_examples, args)

    # Save optimized program
    programs_dir = OPTIMIZATION_DIR / "programs"
    programs_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"claude_opus_{args.optimizer}_{args.auto}.json"
    output_path = programs_dir / output_name
    optimized.save(str(output_path))
    print(f"\nOptimized program saved to {output_path.relative_to(PROJECT_ROOT)}")

    # Evaluate optimized program on val set
    print("\nEvaluating optimized program on val set...")
    scores = []
    for i, example in enumerate(val_examples):
        try:
            pred = optimized(page_image=example.page_image)
            score = cer_metric(example, pred)
            scores.append(score)
        except Exception as e:
            print(f"  [{i+1}] ERROR: {e}")

    if scores:
        mean_score = sum(scores) / len(scores)
        mean_cer = 1.0 - mean_score
        print(f"\nOptimized results ({len(scores)} pages):")
        print(f"  Mean CER:   {mean_cer:.4f} ({mean_cer*100:.2f}%)")
        print(f"  Mean score: {mean_score:.4f}")


if __name__ == "__main__":
    main()
