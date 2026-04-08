#!/usr/bin/env python3
"""Phase 1 ablation: test each domain knowledge field in isolation.

Usage:
    # R0 baseline (no domain knowledge)
    python -m scripts.dspy_pipeline.evaluate_model_first --limit 5 --output-name model_first_r0_baseline

    # R1a: abbreviation_table only
    python -m scripts.dspy_pipeline.evaluate_model_first --limit 5 --field abbreviation_table --output-name model_first_r1a_abbrev

    # R1b: confusion_pairs only
    python -m scripts.dspy_pipeline.evaluate_model_first --limit 5 --field confusion_pairs --output-name model_first_r1b_confuse

    # R1c: valid_alternations only
    python -m scripts.dspy_pipeline.evaluate_model_first --limit 5 --field valid_alternations --output-name model_first_r1c_altern

    # All fields combined
    python -m scripts.dspy_pipeline.evaluate_model_first --limit 5 --field all --output-name model_first_r2_combined
"""

import argparse
import logging
import sys

import dspy

from .block_pipeline import transcribe_page_blocks
from .block_transcriber import (
    BlockTranscriber,
    EnrichedBlockTranscriber,
    EnrichedStripMerger,
    StripMerger,
)
from .config import DEFAULT_MAX_TOKENS, DEFAULT_MODEL, PROJECT_ROOT, RESULTS_DIR
from .domain_knowledge import load_domain_knowledge
from .layout import load_yolo_model
from .transcriber import ManuscriptTranscriber

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from pipeline_config import load_manifest  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Phase 1 ablation: domain knowledge fields")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=5, help="Pages to test (default 5)")
    parser.add_argument("--output-name", default="model_first_test", help="Results directory name")
    parser.add_argument(
        "--field",
        choices=["abbreviation_table", "confusion_pairs", "valid_alternations", "all"],
        default=None,
        help="Domain knowledge field to test (None = baseline, 'all' = all fields)",
    )
    parser.add_argument(
        "--dataset",
        choices=["optimization", "benchmark"],
        default="optimization",
        help="Which dataset to use",
    )
    parser.add_argument("--rlm", action="store_true", help="Enable RLM per-strip abbreviation lookup")
    parser.add_argument("--critic", action="store_true", help="Enable critic review (different model)")
    parser.add_argument("--critic-model", default="openrouter/openai/gpt-5.4",
                        help="Model for critic (should differ from transcriber)")
    args = parser.parse_args()

    # Configure DSPy
    lm = dspy.LM(args.model, max_tokens=DEFAULT_MAX_TOKENS)
    dspy.configure(lm=lm)

    # Load YOLO
    logger.info("Loading YOLO layout model...")
    yolo_model = load_yolo_model()

    # Choose block program and strip merger based on field selection
    if args.field is None:
        block_program = BlockTranscriber()
        strip_merger = StripMerger()
        domain_knowledge = None
        logger.info("Mode: R0 baseline (no domain knowledge)")
    else:
        block_program = EnrichedBlockTranscriber()
        strip_merger = EnrichedStripMerger()
        if args.field == "all":
            domain_knowledge = load_domain_knowledge()
        else:
            domain_knowledge = load_domain_knowledge(fields=[args.field])
        logger.info("Mode: enriched with fields %s", list(domain_knowledge.keys()))

    # Fallback program for pages where YOLO detects no regions
    fallback_program = ManuscriptTranscriber()

    # RLM per-strip abbreviation lookup
    rlm_corrector = None
    if args.rlm:
        from .rlm_corrector import create_rlm_corrector
        rlm_corrector = create_rlm_corrector()
        logger.info("RLM per-strip abbreviation lookup enabled")

    # Critic (different model)
    critic = None
    if args.critic:
        from .critic import Critic
        critic_lm = dspy.LM(args.critic_model, max_tokens=DEFAULT_MAX_TOKENS)
        critic = Critic(critic_lm=critic_lm)
        logger.info("Critic enabled: %s", args.critic_model)

    # Load pages
    if args.dataset == "optimization":
        from .dataset import load_optimization_dataset, split_train_val
        entries = load_optimization_dataset()
        _, val_entries = split_train_val(entries)
        pages = [
            {"page_id": e["page_id"], "image_path": str(PROJECT_ROOT / e["image_path"])}
            for e in val_entries
        ]
    else:
        manifest = load_manifest("codea")
        pages = [
            {"page_id": e["page_id"], "image_path": str(PROJECT_ROOT / e["image_path"])}
            for e in manifest
        ]

    if args.limit:
        pages = pages[: args.limit]
    logger.info("Testing on %d pages from %s set", len(pages), args.dataset)

    # Setup output
    raw_dir = RESULTS_DIR / args.output_name / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Run
    success = skipped = errors = 0
    for i, entry in enumerate(pages):
        page_id = entry["page_id"]
        out_path = raw_dir / f"{page_id}_raw.txt"

        if out_path.exists() and out_path.stat().st_size > 0:
            logger.info("[%d/%d] %s — SKIP", i + 1, len(pages), page_id)
            skipped += 1
            continue

        logger.info("[%d/%d] %s", i + 1, len(pages), page_id)
        try:
            text = transcribe_page_blocks(
                image_path=entry["image_path"],
                block_program=block_program,
                yolo_model=yolo_model,
                model_name=args.model,
                strip_merger=strip_merger,
                domain_knowledge=domain_knowledge,
                rlm_corrector=rlm_corrector,
                critic=critic,
                fallback_program=fallback_program,
            )
            out_path.write_text(text, encoding="utf-8")
            logger.info("  → %d chars", len(text))
            success += 1
        except Exception as e:
            logger.error("  → ERROR: %s", e)
            errors += 1

    logger.info("\nDone: %d ok, %d skipped, %d errors", success, skipped, errors)
    logger.info("Results: %s", raw_dir.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
