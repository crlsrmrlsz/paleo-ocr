#!/usr/bin/env python3
"""Run optimized DSPy program on the 42-page CODEA paleographic eval benchmark.

Saves transcriptions to data/results/_experiments/{name}/raw/ in the same
format as existing models, so the evaluation pipeline works unchanged.

Usage:
    python -m scripts.dspy_pipeline.evaluate --program data/optimization/programs/claude_opus_light.json
    python -m scripts.dspy_pipeline.evaluate --program data/optimization/programs/claude_opus_light.json --limit 3
"""

import argparse
import json
import sys
from pathlib import Path

import dspy

from .config import DEFAULT_MAX_TOKENS, DEFAULT_MODEL, PROJECT_ROOT, RESULTS_DIR, SUBSETS_DIR
from .metric import cer_metric
from .transcriber import ManuscriptTranscriber

# Reuse the existing manifest loader
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from pipeline_config import load_manifest  # noqa: E402


def load_eval_entries(subset: str = "codea") -> list[dict]:
    """Load eval benchmark entries and map to image/GT paths."""
    manifest = load_manifest(subset)
    entries = []
    for entry in manifest:
        entries.append({
            "page_id": entry["page_id"],
            "image_path": entry["image_path"],
            "gt_path": entry["gt_paths"]["paleographic"],
        })
    return entries


def main():
    parser = argparse.ArgumentParser(description="Evaluate DSPy program on benchmark")
    parser.add_argument("--program", default=None, help="Path to optimized program JSON (omit for baseline)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LM model string")
    parser.add_argument("--limit", type=int, default=0, help="Limit pages (0=all)")
    parser.add_argument(
        "--output-name",
        default="_experiments/dspy_claude_opus_initial",
        help="Model name for results directory",
    )
    args = parser.parse_args()

    # Configure DSPy
    lm = dspy.LM(args.model, max_tokens=DEFAULT_MAX_TOKENS)
    dspy.configure(lm=lm)

    # Load program
    program = ManuscriptTranscriber()
    if args.program:
        program.load(args.program)
        print(f"Loaded program from {args.program}")
    else:
        print("Running un-optimized baseline program")

    # Load eval benchmark (CODEA paleographic only for v1)
    entries = load_eval_entries("codea")
    if args.limit:
        entries = entries[: args.limit]
    print(f"Evaluating on {len(entries)} pages")

    # Setup output directory
    raw_dir = RESULTS_DIR / args.output_name / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Run
    results = []
    for i, entry in enumerate(entries):
        page_id = entry["page_id"]
        out_path = raw_dir / f"{page_id}_raw.txt"

        # Skip already-processed (resumable)
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"  [{i+1}/{len(entries)}] {page_id} — SKIP")
            continue

        image_path = str(PROJECT_ROOT / entry["image_path"])
        try:
            pred = program(page_image=dspy.Image(image_path))
            text = pred.transcription
            out_path.write_text(text, encoding="utf-8")
            print(f"  [{i+1}/{len(entries)}] {page_id} — {len(text)} chars")
            results.append({"page_id": page_id, "chars": len(text)})
        except Exception as e:
            print(f"  [{i+1}/{len(entries)}] {page_id} — ERROR: {e}")
            results.append({"page_id": page_id, "error": str(e)})

    print(f"\nTranscriptions saved to {raw_dir.relative_to(PROJECT_ROOT)}")
    print(f"Next steps:")
    print(f"  python scripts/evaluation/normalize_output.py --subset codea")
    print(f"  python scripts/evaluation/compute_metrics.py --subset codea --edition paleographic")
    print(f"  python scripts/evaluation/generate_reports.py")


if __name__ == "__main__":
    main()
