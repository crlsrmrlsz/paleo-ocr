"""Shared utilities for HTR model transcription scripts.

Provides:
  - Manifest loading and page iteration (subset-aware)
  - Image loading and base64 encoding
  - Output directory management
  - Progress tracking and error handling
  - Shared prompt template for raw baseline
"""

import base64
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SUBSETS_DIR = DATA_DIR / "subsets"
RESULTS_DIR = DATA_DIR / "results"

# Re-export load_manifest so transcription runners can import from common
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from pipeline_config import load_manifest  # noqa: E402, F401

# Minimal prompt for raw baseline — no domain hints
BASELINE_PROMPT = (
    "Transcribe the handwritten text in this manuscript image exactly as written. "
    "Preserve original spelling. Expand abbreviations. Maintain line breaks. "
    "If illegible, write [...]."
)


def get_image_path(entry: dict) -> Path:
    """Get the canonical image path for a subset entry."""
    dataset = entry["dataset"]
    canonical = entry["canonical_name"]
    return SUBSETS_DIR / dataset / "images" / canonical


def load_image_bytes(entry: dict) -> bytes:
    """Load image file as raw bytes."""
    path = get_image_path(entry)
    if not path.exists():
        # Try resolving symlink target
        target = path.resolve()
        if not target.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        path = target
    return path.read_bytes()


def encode_image_base64(
    entry: dict,
    media_type: str = "image/jpeg",
    max_b64_bytes: int = 0,
) -> str:
    """Encode image as base64 string for API calls.

    Args:
        max_b64_bytes: If >0, recompress JPEG at lower quality until the base64
            output fits. Needed for Anthropic's 5 MB limit.
    """
    img_bytes = load_image_bytes(entry)
    b64 = base64.standard_b64encode(img_bytes).decode("utf-8")

    if max_b64_bytes > 0 and len(b64) > max_b64_bytes:
        from io import BytesIO
        from PIL import Image

        img = Image.open(BytesIO(img_bytes))
        for quality in (85, 75, 65, 50):
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
            if len(b64) <= max_b64_bytes:
                break

    return b64


def setup_output_dir(model_name: str) -> tuple[Path, Path]:
    """Create output directories for a model. Returns (raw_dir, metadata_path)."""
    model_dir = RESULTS_DIR / model_name
    raw_dir = model_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, model_dir / "metadata.json"


def save_raw_output(raw_dir: Path, page_id: str, text: str):
    """Save raw transcription output for a page."""
    out_path = raw_dir / f"{page_id}_raw.txt"
    out_path.write_text(text, encoding="utf-8")


def save_api_response(model_dir: Path, page_id: str, response: dict):
    """Save full API response (for models that provide structured output)."""
    api_dir = model_dir / "api_responses"
    api_dir.mkdir(parents=True, exist_ok=True)
    out_path = api_dir / f"{page_id}.json"
    with open(out_path, "w") as f:
        json.dump(response, f, indent=2, ensure_ascii=False)


def save_metadata(metadata_path: Path, model_name: str, model_id: str, **extra):
    """Save model run metadata."""
    meta = {
        "model_name": model_name,
        "model_id": model_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt": BASELINE_PROMPT,
        **extra,
    }
    with open(metadata_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def is_already_done(raw_dir: Path, page_id: str) -> bool:
    """Check if a page has already been transcribed (for resumable runs)."""
    out_path = raw_dir / f"{page_id}_raw.txt"
    return out_path.exists() and out_path.stat().st_size > 0


def run_model(
    model_name: str,
    model_id: str,
    transcribe_fn,
    rate_limit_delay: float = 1.0,
    subset: str = "all",
    **metadata_extra,
):
    """Generic runner for API-based transcription models.

    Args:
        model_name: Directory name for results (e.g., 'gpt_5_2')
        model_id: API model identifier
        transcribe_fn: Callable(entry, model_id) -> str that performs transcription
        rate_limit_delay: Seconds to wait between API calls
        subset: Which dataset subset to process ("codea", "toledo", or "all")
        **metadata_extra: Additional metadata fields
    """
    manifest = load_manifest(subset)
    raw_dir, metadata_path = setup_output_dir(model_name)

    print(f"Running {model_name} on {len(manifest)} pages (subset={subset})...")
    print(f"Output: {raw_dir.relative_to(PROJECT_ROOT)}")

    save_metadata(
        metadata_path, model_name, model_id,
        subset=subset, **metadata_extra,
    )

    success = 0
    skipped = 0
    errors = 0

    for i, entry in enumerate(manifest):
        page_id = entry["page_id"]

        if is_already_done(raw_dir, page_id):
            skipped += 1
            continue

        print(f"  [{i+1}/{len(manifest)}] {page_id}...", end=" ", flush=True)

        try:
            text = transcribe_fn(entry, model_id)
            save_raw_output(raw_dir, page_id, text)
            print(f"OK ({len(text)} chars)")
            success += 1
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            errors += 1

        if rate_limit_delay > 0:
            time.sleep(rate_limit_delay)

    print(f"\nDone: {success} ok, {skipped} skipped, {errors} errors")
    return success, skipped, errors
