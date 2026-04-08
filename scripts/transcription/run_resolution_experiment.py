#!/usr/bin/env python3
"""Resolution experiment: transcribe selected pages at multiple resolutions.

Tests Claude Opus 4.6 on high-resolution CODEA pages at 5 input resolutions
to quantify the effect of image downscaling on CER. Same model, same prompt —
only image size changes (Lanczos downscale before sending).

Usage:
    python scripts/transcription/run_resolution_experiment.py
    python scripts/transcription/run_resolution_experiment.py --pages codea_CODEA-2025_1r codea_CODEA-2025_2r
    python scripts/transcription/run_resolution_experiment.py --compute-only   # just compute CER from existing outputs
"""

import argparse
import base64
import json
import os
import re
import sys
import time
import unicodedata
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from pipeline_config import EXCLUDED_PAGES

SUBSETS_DIR = PROJECT_ROOT / "data" / "subsets"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
OUTPUT_DIR = RESULTS_DIR / "_experiments" / "resolution_test"

RESOLUTIONS = [3072, 1568, 1024, 768, 512]

MODEL_ID = "anthropic/claude-opus-4.6"

PROMPT = (
    "Transcribe the handwritten text in this manuscript image exactly as written. "
    "Preserve original spelling. Expand abbreviations. Maintain line breaks. "
    "If illegible, write [...]."
)

# 5 best-performing CODEA pages (lowest CER with pipeline/11_yolo_crop_anti_halluc)
DEFAULT_PAGES = [
    "codea_CODEA-2025_1r",   # 11.2% CER — best overall
    "codea_CODEA-2025_2r",   # 14.2% CER
    "codea_CODEA-2025_3v",   # 14.8% CER
    "codea_CODEA-3593_1r",   # 15.2% CER
    "codea_CODEA-3644_1v",   # 18.4% CER
]


# ---------------------------------------------------------------------------
# GT marker stripping (same as evaluation pipeline, content mode)
# ---------------------------------------------------------------------------

_MARKERS_WITH_TEXT = [
    "tachado", "raspado", "sobre raspado", "sobrescrito",
    "interlineado", "firma", "margen", r"mano \d+",
    "encabezamiento", r"título", r"lat\.", "nota",
]
_MARKERS_EMPTY = [
    "rúbrica", "cruz", "signo", "sello", "crismón",
    "blanco", "roto", "doblez", "mancha", "ilegible", "quirógrafo",
]
_KEEP_TEXT = {"interlineado", "sobrescrito", "encabezamiento", "firma", "mano"}


def strip_markers_content(text: str) -> str:
    for name in _MARKERS_WITH_TEXT:
        if any(k in name for k in _KEEP_TEXT):
            text = re.sub(rf"\[{name}\s*:\s*(.*?)\]", r"\1", text, flags=re.I | re.DOTALL)
        else:
            text = re.sub(rf"\[{name}\s*:\s*.*?\]", "", text, flags=re.I | re.DOTALL)
    for name in _MARKERS_EMPTY:
        if name == "blanco":
            text = re.sub(r"\[blanco\]", " ", text, flags=re.I)
        else:
            text = re.sub(rf"\[{name}\]", "", text, flags=re.I)
    text = re.sub(r"\[([a-záéíóúñç]{1,5})\]", r"\1", text)
    return text


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = strip_markers_content(text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Image resizing
# ---------------------------------------------------------------------------

def resize_image(img_path: Path, max_long_side: int) -> bytes:
    """Resize image so longest side <= max_long_side using Lanczos. Returns JPEG bytes."""
    img = Image.open(img_path)
    w, h = img.size
    longest = max(w, h)

    if longest <= max_long_side:
        # Already within limit, return as-is
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return buf.getvalue()

    scale = max_long_side / longest
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    buf = BytesIO()
    img_resized.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def transcribe_at_resolution(img_path: Path, max_px: int) -> str:
    """Transcribe a page at a specific resolution via OpenRouter."""
    from openai import OpenAI

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    img_bytes = resize_image(img_path, max_px)
    b64 = base64.standard_b64encode(img_bytes).decode("utf-8")

    # Check Anthropic's 5MB base64 limit
    if len(b64) > 5_000_000:
        img = Image.open(BytesIO(img_bytes))
        for quality in (85, 75, 65, 50):
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
            if len(b64) <= 5_000_000:
                break

    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
        max_tokens=4096,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# CER computation
# ---------------------------------------------------------------------------

def compute_cer(gt: str, hyp: str) -> float:
    import editdistance
    if not gt:
        return 0.0
    return editdistance.eval(hyp, gt) / len(gt)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(pages: list[str]):
    """Run transcription at all resolutions for given pages."""
    manifest = json.loads((SUBSETS_DIR / "codea" / "manifest.json").read_text())
    entry_map = {e["page_id"]: e for e in manifest}

    total_calls = len(pages) * len(RESOLUTIONS)
    done = 0

    for page_id in pages:
        entry = entry_map.get(page_id)
        if not entry:
            print(f"  SKIP {page_id}: not in manifest")
            continue

        img_path = SUBSETS_DIR / "codea" / "images" / entry["canonical_name"]
        if not img_path.exists():
            print(f"  SKIP {page_id}: image not found at {img_path}")
            continue

        img = Image.open(img_path)
        w, h = img.size
        img.close()
        print(f"\n{page_id} ({w}x{h}, {w*h/1e6:.1f} MP)")

        for max_px in RESOLUTIONS:
            out_dir = OUTPUT_DIR / f"{max_px}px" / "raw"
            out_file = out_dir / f"{page_id}_raw.txt"

            if out_file.exists():
                done += 1
                print(f"  {max_px}px: already exists, skipping")
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"  {max_px}px: transcribing... ", end="", flush=True)

            try:
                text = transcribe_at_resolution(img_path, max_px)
                out_file.write_text(text, encoding="utf-8")
                done += 1
                print(f"OK ({len(text)} chars) [{done}/{total_calls}]")
            except Exception as e:
                print(f"ERROR: {e}")

            time.sleep(1.5)  # rate limit


def compute_results(pages: list[str]):
    """Compute and display CER results from existing outputs."""
    import editdistance  # noqa: F811

    gt_map = {}
    for page_id in pages:
        gt_path = SUBSETS_DIR / "codea" / "ground_truth" / "paleographic" / f"{page_id}.txt"
        if gt_path.exists():
            gt_map[page_id] = normalize_text(gt_path.read_text())

    print(f"\n{'Page':<30} ", end="")
    for r in RESOLUTIONS:
        print(f"{r}px".rjust(8), end="")
    print()
    print("-" * (30 + 8 * len(RESOLUTIONS)))

    all_cers = {r: [] for r in RESOLUTIONS}

    for page_id in pages:
        gt = gt_map.get(page_id)
        if not gt:
            print(f"{page_id:<30}  NO GT")
            continue

        print(f"{page_id:<30} ", end="")
        for max_px in RESOLUTIONS:
            hyp_path = OUTPUT_DIR / f"{max_px}px" / "raw" / f"{page_id}_raw.txt"
            if not hyp_path.exists():
                print(f"{'N/A':>8}", end="")
                continue
            hyp = normalize_text(hyp_path.read_text())
            cer = compute_cer(gt, hyp)
            all_cers[max_px].append(cer)
            print(f"{cer*100:>7.1f}%", end="")
        print()

    print("-" * (30 + 8 * len(RESOLUTIONS)))
    print(f"{'MEAN':<30} ", end="")
    for r in RESOLUTIONS:
        if all_cers[r]:
            mean = sum(all_cers[r]) / len(all_cers[r])
            print(f"{mean*100:>7.1f}%", end="")
    print()

    # Summary for README
    print("\n\nREADME table format:")
    print("| Input Resolution | Mean CER | vs 1568px | What Happens |")
    print("|-----------------|---------|-----------|--------------|")
    baseline = sum(all_cers[1568]) / len(all_cers[1568]) if all_cers[1568] else 0
    for r in RESOLUTIONS:
        if all_cers[r]:
            mean = sum(all_cers[r]) / len(all_cers[r])
            delta = mean - baseline
            if r == 1568:
                print(f"| **{r:,} px** | **{mean*100:.1f}%** | **baseline** | **Claude's native limit — optimal input size** |")
            elif r > 1568:
                print(f"| {r:,} px | {mean*100:.1f}% | {delta*100:+.1f}% | Claude caps internally at 1568 — extra pixels wasted |")
            elif r == 1024:
                print(f"| {r:,} px | {mean*100:.1f}% | {delta*100:+.1f}% | Moderate detail loss — some letterforms ambiguous |")
            elif r == 768:
                print(f"| {r:,} px | {mean*100:.1f}% | {delta*100:+.1f}% | Significant loss — abbreviation marks disappear |")
            elif r == 512:
                print(f"| {r:,} px | {mean*100:.1f}% | {delta*100:+.1f}% | Severe — procesal text effectively illegible |")


def main():
    parser = argparse.ArgumentParser(description="Resolution experiment for HTR")
    parser.add_argument("--pages", nargs="+", default=DEFAULT_PAGES,
                        help="Page IDs to test (default: 5 best-performing)")
    parser.add_argument("--compute-only", action="store_true",
                        help="Only compute CER from existing outputs, no API calls")
    args = parser.parse_args()

    print(f"Pages: {args.pages}")
    print(f"Resolutions: {RESOLUTIONS}")
    print(f"Output: {OUTPUT_DIR}")

    if not args.compute_only:
        if not os.environ.get("OPENROUTER_API_KEY"):
            print("\nERROR: OPENROUTER_API_KEY not set. Add it to .env or export it.")
            sys.exit(1)
        run_experiment(args.pages)

    compute_results(args.pages)


if __name__ == "__main__":
    main()
