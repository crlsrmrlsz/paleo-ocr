#!/usr/bin/env python3
"""Run DSPy pipeline variants on the CODEA/Toledo eval benchmark.

Uses a strategy registry to dispatch pipeline variants by name.
Each variant combines a transcription function, transcriber, merger,
and optional analyzer/preprocessor.

Usage:
    python -m scripts.dspy_pipeline.evaluate_blocks --pipeline yolo_blocks
    python -m scripts.dspy_pipeline.evaluate_blocks --pipeline yolo_crop_anti_delete
    python -m scripts.dspy_pipeline.evaluate_blocks --list-pipelines
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from typing import Callable

import dspy

from .block_pipeline import (
    transcribe_page_blocks,
    transcribe_page_model_first,
    transcribe_page_preprocess,
    transcribe_page_strips_no_context,
    transcribe_page_strips_page_desc,
    transcribe_page_strips_split_context,
    transcribe_page_tiles_2d,
    transcribe_page_yolo_crop,
    transcribe_page_yolo_crop_anti_halluc,
    transcribe_page_yolo_crop_noise_warn,
    transcribe_page_yolo_crop_strips,
)
from .block_transcriber import (
    AntiDeleteStripMerger,
    AntiHallucStripTranscriber,
    BareStripMerger,
    BlockTranscriber,
    ContextAwareStripTranscriber,
    NoiseAwareStripMerger,
    PageDescStripMerger,
    StripMerger,
    StripTranscriber,
    StructuralStripMerger,
)
from .config import DEFAULT_MAX_TOKENS, DEFAULT_MODEL, PROJECT_ROOT, RESULTS_DIR
from .image_utils import preprocess_clahe, preprocess_denoise, preprocess_sharpen
from .layout import load_yolo_model
from .page_analyzer import NoiseDetector, PageAnalyzer, SplitContextAnalyzer
from .region_decider import RegionDecider
from .transcriber import ManuscriptTranscriber

# Reuse the existing manifest loader
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from pipeline_config import load_manifest  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Configuration for a pipeline variant."""

    description: str
    transcribe_fn: Callable
    transcriber_cls: type
    merger_cls: type
    analyzer_cls: type | None = None
    needs_yolo: bool = False
    needs_region_decider: bool = False
    preprocess_fn: Callable | None = None


PIPELINES: dict[str, PipelineConfig] = {
    "yolo_blocks": PipelineConfig(
        description="YOLO layout + per-block transcription",
        transcribe_fn=transcribe_page_blocks,
        transcriber_cls=BlockTranscriber,
        merger_cls=StripMerger,
        needs_yolo=True,
    ),
    "model_first": PipelineConfig(
        description="Analyze + decide regions + filter + transcribe",
        transcribe_fn=transcribe_page_model_first,
        transcriber_cls=StripTranscriber,
        merger_cls=StripMerger,
        analyzer_cls=PageAnalyzer,
        needs_yolo=True,
        needs_region_decider=True,
    ),
    "strips_page_desc": PipelineConfig(
        description="Auto-crop strips + page description merger",
        transcribe_fn=transcribe_page_strips_page_desc,
        transcriber_cls=StripTranscriber,
        merger_cls=PageDescStripMerger,
        analyzer_cls=PageAnalyzer,
    ),
    "strips_split_context": PipelineConfig(
        description="Split reading/structural context",
        transcribe_fn=transcribe_page_strips_split_context,
        transcriber_cls=ContextAwareStripTranscriber,
        merger_cls=StructuralStripMerger,
        analyzer_cls=SplitContextAnalyzer,
    ),
    "tiles_2d": PipelineConfig(
        description="Full-resolution 2D tiling",
        transcribe_fn=transcribe_page_tiles_2d,
        transcriber_cls=StripTranscriber,
        merger_cls=StripMerger,
    ),
    "strips_no_context": PipelineConfig(
        description="Strips with zero page analysis",
        transcribe_fn=transcribe_page_strips_no_context,
        transcriber_cls=StripTranscriber,
        merger_cls=StripMerger,
    ),
    "yolo_crop_strips": PipelineConfig(
        description="YOLO crop + strips",
        transcribe_fn=transcribe_page_yolo_crop_strips,
        transcriber_cls=StripTranscriber,
        merger_cls=StripMerger,
        needs_yolo=True,
    ),
    "yolo_crop_noise_warn": PipelineConfig(
        description="YOLO crop + noise detection merger",
        transcribe_fn=transcribe_page_yolo_crop_noise_warn,
        transcriber_cls=StripTranscriber,
        merger_cls=NoiseAwareStripMerger,
        analyzer_cls=NoiseDetector,
        needs_yolo=True,
    ),
    "yolo_crop_anti_delete": PipelineConfig(
        description="YOLO crop + anti-deletion merger",
        transcribe_fn=transcribe_page_yolo_crop,
        transcriber_cls=StripTranscriber,
        merger_cls=AntiDeleteStripMerger,
        needs_yolo=True,
    ),
    "yolo_crop_bare_merge": PipelineConfig(
        description="YOLO crop + bare merger",
        transcribe_fn=transcribe_page_yolo_crop,
        transcriber_cls=StripTranscriber,
        merger_cls=BareStripMerger,
        needs_yolo=True,
    ),
    "yolo_crop_anti_halluc": PipelineConfig(
        description="YOLO crop + anti-hallucination transcriber",
        transcribe_fn=transcribe_page_yolo_crop_anti_halluc,
        transcriber_cls=AntiHallucStripTranscriber,
        merger_cls=BareStripMerger,
        needs_yolo=True,
    ),
    "clahe": PipelineConfig(
        description="Bare merge + CLAHE contrast enhancement",
        transcribe_fn=transcribe_page_preprocess,
        transcriber_cls=StripTranscriber,
        merger_cls=AntiDeleteStripMerger,
        needs_yolo=True,
        preprocess_fn=preprocess_clahe,
    ),
    "sharpen": PipelineConfig(
        description="Bare merge + unsharp mask sharpening",
        transcribe_fn=transcribe_page_preprocess,
        transcriber_cls=StripTranscriber,
        merger_cls=AntiDeleteStripMerger,
        needs_yolo=True,
        preprocess_fn=preprocess_sharpen,
    ),
    "denoise": PipelineConfig(
        description="Bare merge + bilateral filter denoising",
        transcribe_fn=transcribe_page_preprocess,
        transcriber_cls=StripTranscriber,
        merger_cls=AntiDeleteStripMerger,
        needs_yolo=True,
        preprocess_fn=preprocess_denoise,
    ),
}


def main():
    parser = argparse.ArgumentParser(
        description="Run DSPy pipeline variants on the eval benchmark"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LM model string")
    parser.add_argument("--limit", type=int, default=0, help="Limit pages (0=all)")
    parser.add_argument(
        "--output-name",
        default=None,
        help="Model name for results directory (default: pipeline/{pipeline_name})",
    )
    parser.add_argument(
        "--pipeline",
        choices=list(PIPELINES.keys()),
        default="yolo_blocks",
        help="Pipeline variant to run",
    )
    parser.add_argument(
        "--list-pipelines",
        action="store_true",
        help="List available pipeline variants and exit",
    )
    args = parser.parse_args()

    if args.list_pipelines:
        print("Available pipeline variants:\n")
        for name, cfg in PIPELINES.items():
            print(f"  {name:30s} {cfg.description}")
        return

    config = PIPELINES[args.pipeline]
    output_name = args.output_name or f"pipeline/{args.pipeline}"

    # Configure DSPy
    lm = dspy.LM(args.model, max_tokens=DEFAULT_MAX_TOKENS)
    dspy.configure(lm=lm)

    # Instantiate components from config
    logger.info("Pipeline: %s — %s", args.pipeline, config.description)
    transcriber = config.transcriber_cls()
    merger = config.merger_cls()
    analyzer = config.analyzer_cls() if config.analyzer_cls else None
    region_decider = RegionDecider() if config.needs_region_decider else None
    fallback = ManuscriptTranscriber()

    yolo_model = None
    if config.needs_yolo:
        logger.info("Loading YOLO layout model...")
        yolo_model = load_yolo_model()

    # Load eval benchmark
    manifest = load_manifest("all")
    entries = []
    for entry in manifest:
        entries.append({
            "page_id": entry["page_id"],
            "image_path": str(PROJECT_ROOT / entry["image_path"]),
        })
    if args.limit:
        entries = entries[: args.limit]
    logger.info("Evaluating on %d pages", len(entries))

    # Setup output directory
    raw_dir = RESULTS_DIR / output_name / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Run
    success = 0
    skipped = 0
    errors = 0

    for i, entry in enumerate(entries):
        page_id = entry["page_id"]
        out_path = raw_dir / f"{page_id}_raw.txt"

        # Resumable: skip already-processed pages
        if out_path.exists() and out_path.stat().st_size > 0:
            logger.info("[%d/%d] %s — SKIP", i + 1, len(entries), page_id)
            skipped += 1
            continue

        logger.info("[%d/%d] %s", i + 1, len(entries), page_id)
        try:
            text = config.transcribe_fn(
                image_path=entry["image_path"],
                strip_transcriber=transcriber,
                strip_merger=merger,
                page_analyzer=analyzer,
                yolo_model=yolo_model,
                region_decider=region_decider,
                model_name=args.model,
                fallback_program=fallback,
                preprocess_fn=config.preprocess_fn,
                block_program=transcriber,
            )
            out_path.write_text(text, encoding="utf-8")
            logger.info("  → %d chars written", len(text))
            success += 1
        except Exception as e:
            logger.error("  → ERROR: %s", e)
            errors += 1

    logger.info(
        "\nDone: %d ok, %d skipped, %d errors",
        success, skipped, errors,
    )
    logger.info("Transcriptions saved to %s", raw_dir.relative_to(PROJECT_ROOT))
    logger.info("\nNext steps:")
    logger.info("  python scripts/evaluation/normalize_output.py")
    logger.info("  python scripts/evaluation/compute_metrics.py --subset codea --edition paleographic")
    logger.info("  python scripts/evaluation/generate_reports.py")


if __name__ == "__main__":
    main()
