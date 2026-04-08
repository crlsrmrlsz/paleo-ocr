#!/usr/bin/env python3
"""Extract facsimile images from Toledo Municipal Archive PDFs.

Reads the raw copia_digital.pdf files from toledo_archive/ and extracts
full-resolution page images into data/corpora/toledo/documents/<dir>/facsimiles/.
Also copies the original PDFs and updates catalog.json with page counts.

Image extraction strategy (PyMuPDF / fitz):
- Single-image pages (vast majority): extract embedded JPEG directly via
  fitz.Pixmap — lossless, preserves original scan resolution.
- Multi-image pages (e.g., softmask overlays): render page at native
  resolution to capture the composited result.
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import fitz  # PyMuPDF
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARCHIVE_DIR = PROJECT_ROOT / "toledo_archive" / "documents"
DATA_DIR = PROJECT_ROOT / "data" / "corpora" / "toledo" / "documents"
CATALOG_PATH = PROJECT_ROOT / "data" / "corpora" / "toledo" / "catalog.json"
LOG_PATH = PROJECT_ROOT / "data" / "corpora" / "toledo" / "extract_facsimiles.log"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Image extraction
# ---------------------------------------------------------------------------


def extract_page_image(doc: fitz.Document, page_idx: int, output_path: Path) -> bool:
    """Extract a single page's image from a PDF.

    For pages with exactly one image (no softmask), extracts the raw embedded
    image data — lossless, preserving original resolution. For pages with
    multiple images or softmask overlays, renders the page at native resolution.

    Returns True on success.
    """
    page = doc[page_idx]
    image_list = page.get_images(full=True)

    # Single image, no softmask → extract raw embedded data
    if len(image_list) == 1:
        xref = image_list[0][0]
        smask = image_list[0][1]  # softmask xref, 0 if none

        if smask == 0:
            pix = fitz.Pixmap(doc, xref)
            # Convert CMYK or other colorspaces to RGB
            if pix.n - pix.alpha > 3:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            pix.save(str(output_path))
            return True

    # Multi-image or softmask: render page at native resolution
    # Determine native DPI from page dimensions vs first image dimensions
    if image_list:
        img_width = image_list[0][2]  # width in pixels
        page_width_pt = page.rect.width  # width in points (1/72 inch)
        native_dpi = img_width / (page_width_pt / 72)
        zoom = native_dpi / 72
    else:
        # No images at all — render at 300 DPI as fallback
        zoom = 300 / 72

    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    pix.save(str(output_path))
    return True


def extract_facsimiles(pdf_path: Path, output_dir: Path) -> int:
    """Extract all pages from a PDF as individual JPEG images.

    Returns the number of pages extracted.
    """
    doc = fitz.open(str(pdf_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    page_count = len(doc)
    for i in range(page_count):
        out_path = output_dir / f"{i + 1:04d}.jpg"
        try:
            extract_page_image(doc, i, out_path)
        except Exception as e:
            log.error("Failed to extract page %d from %s: %s", i + 1, pdf_path.name, e)
            # Fallback: render at 300 DPI
            try:
                mat = fitz.Matrix(300 / 72, 300 / 72)
                pix = doc[i].get_pixmap(matrix=mat)
                pix.save(str(out_path))
            except Exception as e2:
                log.error("Fallback render also failed for page %d: %s", i + 1, e2)

    doc.close()
    return page_count


# ---------------------------------------------------------------------------
# PDF copy
# ---------------------------------------------------------------------------


def copy_pdf(src: Path, dest: Path) -> None:
    """Copy the original copia_digital.pdf into the data directory."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == src.stat().st_size:
        log.debug("PDF already copied: %s", dest)
        return
    shutil.copy2(src, dest)
    log.debug("Copied PDF: %s", dest)


# ---------------------------------------------------------------------------
# Catalog update
# ---------------------------------------------------------------------------


def update_catalog(catalog_path: Path, dir_name: str, page_count: int) -> None:
    """Add facsimile_count to the catalog entry for a given document."""
    if not catalog_path.exists():
        log.warning("Catalog not found: %s", catalog_path)
        return

    data = json.loads(catalog_path.read_text(encoding="utf-8"))
    for entry in data:
        if entry.get("dir_name") == dir_name:
            entry["facsimile_count"] = page_count
            break
    else:
        log.warning("Entry not found in catalog for dir_name=%s", dir_name)
        return

    catalog_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------


def get_document_dirs(start: int, end: int) -> list[Path]:
    """Get sorted list of document directories from the archive, filtered by range."""
    if not ARCHIVE_DIR.exists():
        log.error("Archive directory not found: %s", ARCHIVE_DIR)
        return []

    dirs = sorted(d for d in ARCHIVE_DIR.iterdir() if d.is_dir())

    if end > 0:
        dirs = [d for d in dirs if start <= int(d.name.split("_")[0]) <= end]
    elif start > 1:
        dirs = [d for d in dirs if int(d.name.split("_")[0]) >= start]

    return dirs


def process_documents(
    dirs: list[Path],
    *,
    do_extract: bool = True,
    do_copy_pdf: bool = True,
    dry_run: bool = False,
) -> dict[str, int]:
    """Process a list of document directories.

    Returns a summary dict with counts.
    """
    stats = {"processed": 0, "skipped": 0, "errors": 0, "total_pages": 0}

    for doc_dir in tqdm(dirs, desc="Extracting facsimiles", unit="doc"):
        dir_name = doc_dir.name
        pdf_path = doc_dir / "copia_digital.pdf"
        dest_dir = DATA_DIR / dir_name
        facsimile_dir = dest_dir / "facsimiles"

        if not pdf_path.exists():
            log.warning("No copia_digital.pdf in %s, skipping", dir_name)
            stats["errors"] += 1
            continue

        # Check idempotency: skip if facsimiles already extracted with correct count
        if facsimile_dir.exists() and not dry_run:
            doc = fitz.open(str(pdf_path))
            expected_pages = len(doc)
            doc.close()
            existing = list(facsimile_dir.glob("*.jpg"))
            if len(existing) == expected_pages:
                log.debug("Already extracted %d pages for %s, skipping", expected_pages, dir_name)
                stats["skipped"] += 1
                stats["total_pages"] += expected_pages
                continue

        if dry_run:
            doc = fitz.open(str(pdf_path))
            page_count = len(doc)
            doc.close()
            print(f"  [WOULD EXTRACT] {dir_name}: {page_count} pages → {facsimile_dir.relative_to(PROJECT_ROOT)}")
            if do_copy_pdf:
                dest_pdf = dest_dir / "copia_digital.pdf"
                status = "EXISTS" if dest_pdf.exists() else "WOULD COPY"
                print(f"  [{status}] {dir_name}/copia_digital.pdf")
            stats["total_pages"] += page_count
            stats["processed"] += 1
            continue

        # Extract images
        if do_extract:
            try:
                page_count = extract_facsimiles(pdf_path, facsimile_dir)
                log.info("Extracted %d pages from %s", page_count, dir_name)
                update_catalog(CATALOG_PATH, dir_name, page_count)
                stats["total_pages"] += page_count
            except Exception as e:
                log.error("Failed to process %s: %s", dir_name, e)
                stats["errors"] += 1
                continue

        # Copy PDF
        if do_copy_pdf:
            try:
                copy_pdf(pdf_path, dest_dir / "copia_digital.pdf")
            except Exception as e:
                log.error("Failed to copy PDF for %s: %s", dir_name, e)

        stats["processed"] += 1

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Extract facsimile images from Toledo archive PDFs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--extract", action="store_true", help="Extract page images from PDFs")
    parser.add_argument("--copy-pdf", action="store_true", help="Copy original PDFs to data directory")
    parser.add_argument("--all", action="store_true", help="Extract images and copy PDFs")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing files")
    parser.add_argument("--start", type=int, default=1, help="Start document number (1-based)")
    parser.add_argument("--end", type=int, default=0, help="End document number (0 = all)")
    args = parser.parse_args()

    do_extract = args.extract or args.all
    do_copy_pdf = args.copy_pdf or args.all

    if not any([do_extract, do_copy_pdf]):
        parser.print_help()
        return

    dirs = get_document_dirs(args.start, args.end)
    if not dirs:
        log.error("No document directories found in range %d-%d", args.start, args.end or 999)
        return

    log.info(
        "Processing %d documents (range %d-%d)%s",
        len(dirs), args.start, args.end or len(dirs),
        " [DRY RUN]" if args.dry_run else "",
    )

    stats = process_documents(
        dirs, do_extract=do_extract, do_copy_pdf=do_copy_pdf, dry_run=args.dry_run,
    )

    log.info(
        "Done: %d processed, %d skipped, %d errors, %d total pages",
        stats["processed"], stats["skipped"], stats["errors"], stats["total_pages"],
    )


if __name__ == "__main__":
    main()
