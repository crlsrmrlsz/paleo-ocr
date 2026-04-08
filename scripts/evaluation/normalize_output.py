#!/usr/bin/env python3
"""Normalize raw HTR model outputs for CER comparison.

All raw model outputs pass through identical normalization before evaluation:
  1. Strip model boilerplate (explanations, headers, markdown formatting)
  2. Unicode NFC normalization
  3. Whitespace normalization
  4. Line break standardization

Supports --subset filtering to process only pages from a specific dataset.

Input:  data/results/{model_name}/raw/{page_id}_raw.txt
Output: data/results/{model_name}/normalized/{page_id}.txt
"""

import argparse
import re
import sys
import unicodedata
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(EVAL_DIR))
from pipeline_config import load_manifest  # noqa: E402
from model_preprocessors import preprocess as model_preprocess  # noqa: E402


def normalize_model_output(text: str, model_name: str = None) -> str:
    """Apply standard normalization to model output text."""
    # Model-specific preprocessing (CHURRO header, etc.)
    if model_name:
        text = model_preprocess(text, model_name)

    # Strip common model boilerplate
    text = _strip_boilerplate(text)

    # Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    # Normalize various dash/hyphen characters
    text = text.replace("\u2013", "-")  # en dash
    text = text.replace("\u2014", "-")  # em dash

    # Normalize quotes
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")

    # Remove zero-width characters
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)

    # Normalize marker-like patterns to match GT conventions
    text = _normalize_markers(text)

    # Normalize whitespace (preserve newlines)
    text = re.sub(r"[^\S\n]+", " ", text)

    # Clean up spaces around newlines
    text = re.sub(r" *\n *", "\n", text)

    # Standardize line breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _normalize_markers(text: str) -> str:
    """Normalize marker-like patterns in model output to match GT conventions."""
    # [rubrica] → [rúbrica] (normalize missing accent)
    text = re.sub(r"\[rubrica\]", "[rúbrica]", text, flags=re.IGNORECASE)

    # [firmas: TEXT] → [firma: TEXT]
    text = re.sub(r"\[firmas\s*:", "[firma:", text, flags=re.IGNORECASE)

    # [illegible] or [ilegible] variants
    text = re.sub(r"\[ill?egible\]", "[ilegible]", text, flags=re.IGNORECASE)

    # [...] → [ilegible] (common model output for unclear text)
    text = re.sub(r"\[\.\.\.\]", "[ilegible]", text)

    # [blank] → [blanco]
    text = re.sub(r"\[blank\]", "[blanco]", text, flags=re.IGNORECASE)

    return text


def _strip_boilerplate(text: str) -> str:
    """Remove common model boilerplate patterns."""
    # Strip markdown headers (## Transcription, etc.)
    text = re.sub(r"^#+\s+.*$", "", text, flags=re.MULTILINE)

    # Strip markdown bold/italic markers
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)

    # Strip markdown code fences
    text = re.sub(r"```[^`]*```", "", text, flags=re.DOTALL)

    # Strip common preambles (model explanations before transcription)
    preamble_patterns = [
        r"(?i)^(?:here is|the following is|below is|this is).*?transcription.*?:\s*\n",
        r"(?i)^(?:transcription|transcribed text|output):\s*\n",
        r"(?i)^(?:i can see|the image shows|the manuscript).*?\n(?=\w)",
    ]
    for pattern in preamble_patterns:
        text = re.sub(pattern, "", text, count=1)

    # Strip common postambles (model commentary after transcription)
    postamble_patterns = [
        r"\n(?:note:|notes?:|\*note).*$",
        r"\n(?:the text appears|this appears to be|this document).*$",
        r"\n(?:---+).*$",
    ]
    for pattern in postamble_patterns:
        text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)

    return text


def process_model(model_name: str, manifest: list[dict]) -> dict:
    """Normalize all outputs for a single model. Returns stats dict."""
    model_dir = RESULTS_DIR / model_name
    raw_dir = model_dir / "raw"
    norm_dir = model_dir / "normalized"

    if not raw_dir.exists():
        return {"model": model_name, "status": "no_raw_dir"}

    norm_dir.mkdir(parents=True, exist_ok=True)

    stats = {"model": model_name, "processed": 0, "empty": 0, "missing": 0}

    for entry in manifest:
        page_id = entry["page_id"]
        raw_path = raw_dir / f"{page_id}_raw.txt"
        norm_path = norm_dir / f"{page_id}.txt"

        if not raw_path.exists():
            stats["missing"] += 1
            continue

        raw_text = raw_path.read_text(encoding="utf-8")
        normalized = normalize_model_output(raw_text, model_name=model_name)

        norm_path.write_text(normalized, encoding="utf-8")

        if normalized.strip():
            stats["processed"] += 1
        else:
            stats["empty"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Normalize raw HTR model outputs for CER comparison",
    )
    parser.add_argument(
        "--subset", choices=["codea", "toledo", "all"], default="all",
        help="Which dataset subset to process (default: all)",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.subset)

    if not manifest:
        print(f"No manifest entries found for subset '{args.subset}'.")
        return

    # Find all model directories with raw/ subdirs (supports nested groups)
    model_names = []
    for d in sorted(RESULTS_DIR.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        if (d / "raw").exists():
            model_names.append(d.name)
        else:
            # Check one level deeper (baseline/, pipeline/)
            for sub in sorted(d.iterdir()):
                if sub.is_dir() and (sub / "raw").exists():
                    model_names.append(f"{d.name}/{sub.name}")

    if not model_names:
        print("No model results found in data/results/")
        return

    print(f"Normalizing outputs for {len(model_names)} models ({len(manifest)} pages, subset={args.subset})...\n")

    for model_name in model_names:
        stats = process_model(model_name, manifest)
        status = f"{stats.get('processed', 0)} ok, {stats.get('empty', 0)} empty, {stats.get('missing', 0)} missing"
        print(f"  {model_name}: {status}")

    print("\nDone!")


if __name__ == "__main__":
    main()
