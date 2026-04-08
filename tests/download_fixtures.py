#!/usr/bin/env python3
"""Download test fixtures from external sources.

Run once to populate tests/fixtures/ with test data from:
  - dinglehopper (PAGE XML + ALTO XML for Lorem Ipsum, Fraktur)
  - HIP'21 IMPACT dataset (PAGE XML GT + ALTO XML OCR + dinglehopper CER/WER)

Usage:
    python tests/download_fixtures.py
"""

import os
import sys
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

# ---------------------------------------------------------------------------
# Source URLs
# ---------------------------------------------------------------------------

DINGLEHOPPER_BASE = (
    "https://raw.githubusercontent.com/qurator-spk/dinglehopper/main/src/dinglehopper/tests/data"
)

HIP21_BASE = (
    "https://raw.githubusercontent.com/cneud/hip21_ocrevaluation/main/data"
)

# HIP'21 IMPACT dataset: 5 selected documents, 4 languages
# Source: Neudecker et al. (2021), "A survey of OCR evaluation tools and metrics", HIP'21
HIP21_DOC_IDS = ["00046949", "00046910", "00525436", "00451938", "00539327"]
HIP21_SUFFIXES = [".gt.xml", ".gt4hist.xml", ".gt4hist.dinglehopper.json"]

FIXTURES = {
    # dinglehopper Lorem Ipsum test data
    "dinglehopper/lorem-ipsum-gt.page.xml": f"{DINGLEHOPPER_BASE}/lorem-ipsum-gt.page.xml",
    "dinglehopper/lorem-ipsum-ocr.alto.xml": f"{DINGLEHOPPER_BASE}/lorem-ipsum-ocr.alto.xml",
    # dinglehopper Fraktur test data
    "dinglehopper/fraktur-gt.page.xml": f"{DINGLEHOPPER_BASE}/fraktur-gt.page.xml",
    "dinglehopper/fraktur-ocr.page.xml": f"{DINGLEHOPPER_BASE}/fraktur-ocr.page.xml",
}

# Add HIP'21 fixtures programmatically
for _doc_id in HIP21_DOC_IDS:
    for _suffix in HIP21_SUFFIXES:
        _filename = f"{_doc_id}{_suffix}"
        FIXTURES[f"hip21/{_filename}"] = f"{HIP21_BASE}/{_filename}"


def download_all(force: bool = False):
    """Download all fixtures.

    Args:
        force: If True, re-download even if file exists
    """
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = 0

    for rel_path, url in FIXTURES.items():
        dest = FIXTURES_DIR / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists() and not force:
            print(f"  SKIP (exists): {rel_path}")
            success += 1
            continue

        print(f"  Downloading: {rel_path}")
        try:
            urlretrieve(url, str(dest))
            success += 1
        except (URLError, OSError) as e:
            print(f"  FAILED: {rel_path} — {e}")
            failed += 1

    print(f"\nDone: {success} downloaded, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    force = "--force" in sys.argv
    print(f"Downloading fixtures to {FIXTURES_DIR}")
    ok = download_all(force=force)
    sys.exit(0 if ok else 1)
