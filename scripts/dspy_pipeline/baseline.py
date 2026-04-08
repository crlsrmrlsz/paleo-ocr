#!/usr/bin/env python3
"""Run un-optimized ManuscriptTranscriber on val set to establish CER baseline.

Usage:
    python -m scripts.dspy_pipeline.baseline
    python -m scripts.dspy_pipeline.baseline --limit 3  # quick test with 3 pages
"""

import argparse
import json
import sys
from pathlib import Path

import dspy

from .config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    OPTIMIZATION_DIR,
    PROJECT_ROOT,
)
from .dataset import load_optimization_dataset, save_splits, split_train_val
from .metric import cer_metric
from .transcriber import ContextualManuscriptTranscriber, ManuscriptTranscriber


def _make_bon_program(base_program, n, gt_text):
    """Wrap a program in BestOfN with CER-based reward (for evaluation with GT)."""
    from .metric import _normalize_whitespace

    gt_norm = _normalize_whitespace(gt_text)

    def cer_reward(kwargs, pred):
        import editdistance

        pred_norm = _normalize_whitespace(pred.transcription)
        if not gt_norm:
            return 1.0 if not pred_norm else 0.0
        cer = editdistance.eval(pred_norm, gt_norm) / len(gt_norm)
        return max(0.0, 1.0 - cer)

    return dspy.BestOfN(
        module=base_program, N=n, reward_fn=cer_reward, threshold=0.85
    )


def build_examples(entries: list[dict]) -> list[dspy.Example]:
    """Convert entry dicts to dspy.Examples with Image and GT."""
    examples = []
    for entry in entries:
        image_path = str(PROJECT_ROOT / entry["image_path"])
        gt_path = PROJECT_ROOT / entry["gt_path"]
        gt_text = gt_path.read_text(encoding="utf-8").strip()

        examples.append(
            dspy.Example(
                page_image=dspy.Image(image_path),
                transcription=gt_text,
            ).with_inputs("page_image")
        )
    return examples


def main():
    parser = argparse.ArgumentParser(description="Run DSPy baseline on val set")
    parser.add_argument("--limit", type=int, default=0, help="Limit pages (0=all)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LM model string")
    parser.add_argument(
        "--module", choices=["standard", "contextual"], default="standard",
        help="Module: standard (bare prompt) or contextual (with reading reference)",
    )
    parser.add_argument(
        "--bon", type=int, default=0,
        help="BestOfN: run N times per page, pick best by CER (0=disabled)",
    )
    args = parser.parse_args()

    # Configure DSPy
    lm = dspy.LM(args.model, max_tokens=DEFAULT_MAX_TOKENS)
    dspy.configure(lm=lm)

    # Load and split dataset
    entries = load_optimization_dataset()
    train, val = split_train_val(entries)
    save_splits(train, val)

    if args.limit:
        val = val[: args.limit]

    mode_parts = [args.module]
    if args.bon:
        mode_parts.append(f"BestOf{args.bon}")
    print(f"\nRunning {'+'.join(mode_parts)} on {len(val)} val pages...")
    val_examples = build_examples(val)
    if args.module == "contextual":
        base_program = ContextualManuscriptTranscriber()
    else:
        base_program = ManuscriptTranscriber()

    # Run and score
    results = []
    for i, example in enumerate(val_examples):
        page_id = val[i]["page_id"]
        try:
            if args.bon:
                program = _make_bon_program(
                    base_program, args.bon, example.transcription
                )
            else:
                program = base_program
            pred = program(page_image=example.page_image)
            score = cer_metric(example, pred)
            cer = 1.0 - score
            print(f"  [{i+1}/{len(val_examples)}] {page_id}: CER={cer:.4f}")
            results.append({"page_id": page_id, "cer": cer, "score": score})
        except Exception as e:
            print(f"  [{i+1}/{len(val_examples)}] {page_id}: ERROR — {e}")
            results.append({"page_id": page_id, "cer": None, "error": str(e)})

    # Summary
    valid = [r for r in results if r["cer"] is not None]
    if valid:
        mean_cer = sum(r["cer"] for r in valid) / len(valid)
        mean_score = sum(r["score"] for r in valid) / len(valid)
        print(f"\nBaseline results ({len(valid)}/{len(results)} pages):")
        print(f"  Mean CER:   {mean_cer:.4f} ({mean_cer*100:.2f}%)")
        print(f"  Mean score: {mean_score:.4f}")

    # Save results
    output_path = OPTIMIZATION_DIR / "baseline_results.json"
    with open(output_path, "w") as f:
        json.dump({"model": args.model, "results": results}, f, indent=2)
    print(f"\nSaved to {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
