#!/usr/bin/env python3
"""Inspect DSPy ChainOfThought reasoning for a single page (yolo_crop pipeline).

Runs yolo_crop_anti_delete on one page and prints the full reasoning trace for each step:
- Per-strip transcription reasoning
- Strip merge reasoning

Saves a markdown report to data/evaluation/review/

Usage:
    python -m scripts.dspy_pipeline.inspect_reasoning
    python -m scripts.dspy_pipeline.inspect_reasoning --page-index 5
    python -m scripts.dspy_pipeline.inspect_reasoning --raw-history
"""

import argparse
import json
import logging
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import dspy

from .block_pipeline import auto_crop_margins, fixed_strip_split
from .block_transcriber import StripTranscriber, VisualReasoningStripTranscriber, AntiDeleteStripMerger
from .config import DEFAULT_MAX_TOKENS, DEFAULT_MODEL, PROJECT_ROOT
from .layout import get_max_px, load_yolo_model

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from pipeline_config import load_manifest  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SEPARATOR = "=" * 70
REVIEW_DIR = PROJECT_ROOT / "data" / "evaluation" / "review"


def print_section(title: str, content: str):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)
    print(content)
    print()


def main():
    parser = argparse.ArgumentParser(description="Inspect yolo_crop CoT reasoning")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--page-index", type=int, default=0,
        help="Index into the manifest (0-based)",
    )
    parser.add_argument(
        "--visual-reasoning", action="store_true",
        help="Use visual reasoning transcriber (forced line-by-line analysis)",
    )
    parser.add_argument(
        "--raw-history", action="store_true",
        help="Also print raw LM prompt/completion history",
    )
    args = parser.parse_args()

    # Configure DSPy
    lm = dspy.LM(args.model, max_tokens=DEFAULT_MAX_TOKENS)
    dspy.configure(lm=lm)

    # Load manifest and pick a page
    manifest = load_manifest("all")
    entry = manifest[args.page_index]
    image_path = str(PROJECT_ROOT / entry["image_path"])
    page_id = entry["page_id"]

    page_info = {
        "page_id": page_id,
        "date": entry.get("date"),
        "letra": entry.get("letra"),
        "title": entry.get("title"),
        "image": f"{entry.get('image_width')}x{entry.get('image_height')}",
    }
    print_section("PAGE INFO", json.dumps(page_info, indent=2, ensure_ascii=False))

    # Collect data for markdown report
    report = {
        "page_info": page_info,
        "strips": [],
        "merge_reasoning": None,
        "merge_output": None,
        "final": None,
    }

    # --- yolo_crop pipeline, step by step ---

    # 1. YOLO crop
    logger.info("Loading YOLO model...")
    yolo_model = load_yolo_model()

    content_bbox = auto_crop_margins(image_path, yolo_model=yolo_model)
    if content_bbox is None:
        print("No ink detected - cannot proceed")
        return

    x1, y1, x2, y2 = content_bbox
    crop_info = f"Content bbox: ({x1},{y1})-({x2},{y2}) = {x2-x1}x{y2-y1}px"
    report["crop_info"] = crop_info
    print_section("STEP 1: YOLO CROP", crop_info)

    # 2. Strip splitting
    max_px = get_max_px(args.model)
    strips = fixed_strip_split(image_path, content_bbox, max_px)
    report["num_strips"] = len(strips)
    report["max_px"] = max_px
    print_section("STEP 2: STRIP SPLIT", (
        f"{len(strips)} strips, max height {max_px}px\n" +
        "\n".join(
            f"  Strip {i+1}: {s.crop_path}"
            for i, s in enumerate(strips)
        )
    ))

    # 3. Transcribe each strip — capture reasoning
    version = "visual_reasoning" if args.visual_reasoning else "yolo_crop"
    transcriber = VisualReasoningStripTranscriber() if args.visual_reasoning else StripTranscriber()
    strip_results = []

    for i, strip in enumerate(strips):
        print_section(
            f"STEP 3.{i+1}: TRANSCRIBE STRIP {i+1}/{len(strips)}",
            f"Image: {strip.crop_path}"
        )

        result = transcriber(block_image=dspy.Image(strip.crop_path))

        reasoning = getattr(result, "reasoning", "(no reasoning field found)")
        transcription = result.transcription

        print("--- REASONING ---")
        print(textwrap.fill(reasoning, width=90))
        print(f"\n--- TRANSCRIPTION ---")
        print(transcription)

        strip_results.append((strip, transcription))
        report["strips"].append({
            "index": i + 1,
            "crop_path": str(strip.crop_path),
            "reasoning": reasoning,
            "transcription": transcription,
        })

    # 4. Merge — capture reasoning
    if len(strip_results) > 1:
        labeled = []
        for i, (_, text) in enumerate(strip_results):
            labeled.append(f"---STRIP {i + 1}---\n{text.strip()}")
        strip_text = "\n\n".join(labeled)

        print_section("STEP 4: MERGE INPUT", strip_text)

        merger = AntiDeleteStripMerger()
        result = merger(strip_transcriptions=strip_text)

        merge_reasoning = getattr(result, "reasoning", "(no reasoning field found)")
        merged_output = result.transcription

        report["merge_reasoning"] = merge_reasoning
        report["merge_output"] = merged_output

        print_section("STEP 4: MERGE REASONING", textwrap.fill(merge_reasoning, width=90))
        print_section("STEP 4: MERGED OUTPUT", merged_output)
    else:
        print_section("STEP 4: MERGE", "Only 1 strip - no merge needed")
        merged_output = strip_results[0][1]
        report["merge_reasoning"] = "Single strip - no merge needed"
        report["merge_output"] = merged_output

    report["final"] = merged_output

    # 5. Optionally print raw LM history
    if args.raw_history:
        print_section("RAW LM HISTORY", "")
        history = lm.history
        for i, hist_entry in enumerate(history):
            print(f"\n{'~' * 60}")
            print(f"  LM Call {i+1}")
            print(f"{'~' * 60}")
            if isinstance(hist_entry, dict):
                for k, v in hist_entry.items():
                    val = str(v)
                    if len(val) > 500:
                        val = val[:500] + "..."
                    print(f"  {k}: {val}")
            else:
                print(f"  {hist_entry}")

    # Final summary
    print_section("FINAL TRANSCRIPTION", merged_output)
    print(f"\nTotal length: {len(merged_output)} chars")

    # --- Generate markdown report ---
    md = _build_markdown(report, args.model, version)
    out_path = REVIEW_DIR / f"cot_reasoning_{version}_{page_id}.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"\nMarkdown report saved to: {out_path.relative_to(PROJECT_ROOT)}")


def _build_markdown(report: dict, model: str, version: str = "yolo_crop") -> str:
    info = report["page_info"]
    transcriber_desc = {
        "yolo_crop": "yolo_crop (YOLO crop + strips + minimal merge, generic CoT)",
        "visual_reasoning": "visual_reasoning (YOLO crop + strips + minimal merge, forced visual reasoning)",
    }
    lines = [
        f"# Chain of Thought Reasoning: {info['page_id']}",
        "",
        f"**Pipeline:** {transcriber_desc.get(version, version)}",
        f"**Model:** `{model}`",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Page Info",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Page ID | {info['page_id']} |",
        f"| Date | {info.get('date', 'N/A')} |",
        f"| Script | {info.get('letra', 'N/A')} |",
        f"| Image | {info.get('image', 'N/A')} |",
        f"| Title | {info.get('title', 'N/A')} |",
        "",
        "## Pipeline Steps",
        "",
        "### 1. YOLO Crop",
        "",
        f"{report.get('crop_info', 'N/A')}",
        "",
        f"### 2. Strip Split",
        "",
        f"{report['num_strips']} strips, max height {report['max_px']}px",
        "",
    ]

    # Per-strip sections
    for strip in report["strips"]:
        i = strip["index"]
        lines.extend([
            f"### 3.{i}. Strip {i}/{report['num_strips']} Transcription",
            "",
            f"**Image:** `{strip['crop_path']}`",
            "",
            f"#### Reasoning",
            "",
            f"{strip['reasoning']}",
            "",
            f"#### Transcription",
            "",
            "```",
            strip["transcription"],
            "```",
            "",
        ])

    # Merge section
    lines.extend([
        "### 4. Strip Merge",
        "",
        "#### Merge Reasoning",
        "",
        f"{report['merge_reasoning']}",
        "",
        "#### Merged Output",
        "",
        "```",
        report["merge_output"],
        "```",
        "",
        "---",
        "",
        "## Final Transcription",
        "",
        "```",
        report["final"],
        "```",
        "",
        f"**Total characters:** {len(report['final'])}",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    main()
