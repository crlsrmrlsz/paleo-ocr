#!/usr/bin/env python3
"""Download and convert Toledo Municipal Archive documents.

Scrapes the Laminario de Documentos collection (~121 historical Spanish
documents, 1289-1940) from the Toledo municipal website, downloads the
PDFs (copia digital, descripcion, transcripcion), and converts the text
PDFs to markdown for downstream OCR/NLP work.

Source: https://www.toledo.es/toledo-siempre/laminario-de-documentos-con-sus-transcripciones/
"""

import argparse
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOURCE_URL = (
    "https://www.toledo.es/toledo-siempre/"
    "laminario-de-documentos-con-sus-transcripciones/"
)
INDEX_PDF_URL = (
    "https://www.toledo.es/wp-content/uploads/2025/02/"
    "00_relacion-completa-de-documentos_laminario.pdf"
)
USER_AGENT = (
    "Mozilla/5.0 (compatible; PaleoOCR-Research/1.0; "
    "+https://github.com/fliperbaker/paleo-ocr)"
)
DOWNLOAD_DELAY = 1.0  # seconds between requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_DIR = PROJECT_ROOT / "data" / "corpora" / "toledo"
CATALOG_PATH = BASE_DIR / "catalog.json"
DOCUMENTS_DIR = BASE_DIR / "documents"
INDEX_PDF_PATH = BASE_DIR / "index.pdf"
INDEX_MD_PATH = BASE_DIR / "index.md"
LOG_PATH = BASE_DIR / "download.log"

ARCHIVE_NAME = "Archivo Municipal de Toledo"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

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
# Data model
# ---------------------------------------------------------------------------


@dataclass
class DocumentEntry:
    """Metadata for a single archive document."""

    number: int  # 1-based sequential index
    slug: str  # e.g. "1289-12-20" — unique key from filename
    title: str  # display title from the page
    url_cd: str  # copia digital PDF
    url_d: str  # descripcion PDF
    url_t: str  # transcripcion PDF
    thumbnail_url: str = ""
    dir_name: str = ""  # e.g. "001_1289-12-20"
    downloaded: bool = False
    converted: bool = False
    errors: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.dir_name:
            self.dir_name = f"{self.number:03d}_{self.slug}"


# ---------------------------------------------------------------------------
# HTTP session
# ---------------------------------------------------------------------------


def make_session() -> requests.Session:
    """Create a requests session with retry logic."""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


# ---------------------------------------------------------------------------
# Phase 1: Scrape catalog
# ---------------------------------------------------------------------------


def scrape_catalog(session: requests.Session, dry_run: bool = False) -> list[DocumentEntry]:
    """Fetch the page and parse document metadata into a catalog."""
    log.info("Fetching source page: %s", SOURCE_URL)
    resp = session.get(SOURCE_URL, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    rows = soup.select("table tr")
    log.info("Found %d table rows", len(rows))

    entries: list[DocumentEntry] = []
    number = 0

    for row in rows:
        link_groups = row.select("div.list-group")
        if not link_groups:
            continue

        links = link_groups[0].select("a.list-group-item")
        if len(links) < 3:
            log.warning("Row with < 3 links, skipping: %s", row.get_text(strip=True)[:60])
            continue

        # Classify links by suffix
        urls = {"cd": "", "d": "", "t": ""}
        for a in links:
            href = a.get("href", "")
            if href.endswith("_cd.pdf"):
                urls["cd"] = href
            elif href.endswith("_d.pdf"):
                urls["d"] = href
            elif href.endswith("_t.pdf"):
                urls["t"] = href

        if not all(urls.values()):
            log.warning("Missing PDF links in row, skipping: %s", urls)
            continue

        # Extract slug from the _cd.pdf URL filename
        cd_filename = Path(urlparse(urls["cd"]).path).stem  # e.g. "1289-12-20_cd"
        slug = cd_filename.removesuffix("_cd")

        # Extract title text from the td (text before the link group)
        td = link_groups[0].parent
        # Get direct text content (the date/title) — before the div.list-group
        title_parts = []
        for child in td.children:
            if hasattr(child, "name") and child.name == "div":
                break
            text = child.get_text(strip=True) if hasattr(child, "get_text") else str(child).strip()
            if text:
                title_parts.append(text)
        title = " ".join(title_parts) if title_parts else slug

        # Thumbnail URL from first td
        tds = row.find_all("td")
        thumb_url = ""
        if tds:
            thumb_link = tds[0].find("a")
            if thumb_link:
                thumb_url = thumb_link.get("href", "")

        number += 1
        entry = DocumentEntry(
            number=number,
            slug=slug,
            title=title,
            url_cd=urls["cd"],
            url_d=urls["d"],
            url_t=urls["t"],
            thumbnail_url=thumb_url,
        )
        entries.append(entry)

    log.info("Parsed %d document entries", len(entries))

    if dry_run:
        for e in entries:
            print(f"  {e.dir_name}: {e.title}")
            print(f"    CD: {e.url_cd}")
            print(f"     D: {e.url_d}")
            print(f"     T: {e.url_t}")
        return entries

    save_catalog(entries)
    return entries


# ---------------------------------------------------------------------------
# Phase 2: Download PDFs
# ---------------------------------------------------------------------------


def download_file(session: requests.Session, url: str, dest: Path) -> bool:
    """Download a single file. Skip if it already exists with non-zero size."""
    if dest.exists() and dest.stat().st_size > 0:
        log.debug("Already exists: %s", dest.name)
        return True
    try:
        resp = session.get(url, timeout=60, stream=True)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        log.debug("Downloaded: %s (%d bytes)", dest.name, dest.stat().st_size)
        return True
    except requests.RequestException as e:
        log.error("Failed to download %s: %s", url, e)
        return False


def download_documents(
    session: requests.Session,
    entries: list[DocumentEntry],
    dry_run: bool = False,
) -> list[DocumentEntry]:
    """Download all PDFs for each document entry."""
    # Download the index PDF first
    if not dry_run:
        log.info("Downloading index PDF...")
        download_file(session, INDEX_PDF_URL, INDEX_PDF_PATH)
        time.sleep(DOWNLOAD_DELAY)

    for i, entry in enumerate(tqdm(entries, desc="Downloading", unit="doc")):
        doc_dir = DOCUMENTS_DIR / entry.dir_name
        files = [
            (entry.url_cd, doc_dir / "copia_digital.pdf"),
            (entry.url_d, doc_dir / "descripcion.pdf"),
            (entry.url_t, doc_dir / "transcripcion.pdf"),
        ]

        if dry_run:
            for url, dest in files:
                status = "EXISTS" if dest.exists() and dest.stat().st_size > 0 else "WOULD DOWNLOAD"
                print(f"  [{status}] {dest.relative_to(BASE_DIR)}")
            continue

        all_ok = True
        for url, dest in files:
            if not download_file(session, url, dest):
                entry.errors.append(f"download_failed: {url}")
                all_ok = False
            time.sleep(DOWNLOAD_DELAY)

        entry.downloaded = all_ok

        # Checkpoint every 10 documents
        if (i + 1) % 10 == 0:
            save_catalog(entries)
            log.info("Checkpoint saved at document %d/%d", i + 1, len(entries))

    if not dry_run:
        save_catalog(entries)
    return entries


# ---------------------------------------------------------------------------
# Phase 3: Convert to Markdown
# ---------------------------------------------------------------------------


def pdf_to_markdown(pdf_path: Path, doc_type: str, entry: DocumentEntry, converter) -> str:
    """Convert a PDF to markdown with YAML front matter.

    Uses docling's DocumentConverter for layout-aware extraction.
    The converter instance should be created once and reused across calls.
    """
    result = converter.convert(str(pdf_path))
    md_body = result.document.export_to_markdown()

    source_url_map = {"descripcion": entry.url_d, "transcripcion": entry.url_t}
    source_url = source_url_map.get(doc_type, "")

    front_matter = (
        "---\n"
        f"document_date: \"{entry.title}\"\n"
        f"slug: \"{entry.slug}\"\n"
        f"type: \"{doc_type}\"\n"
        f"archive: \"{ARCHIVE_NAME}\"\n"
        f"source_url: \"{source_url}\"\n"
        f"source_page: \"{SOURCE_URL}\"\n"
        "---\n\n"
    )
    return front_matter + md_body


def convert_documents(
    entries: list[DocumentEntry],
    dry_run: bool = False,
    force_reconvert: bool = False,
) -> list[DocumentEntry]:
    """Convert description and transcription PDFs to markdown.

    Uses docling for layout-aware PDF extraction. The converter is
    instantiated once (model loading is expensive) and reused.
    """
    from docling.document_converter import DocumentConverter

    converter = None
    if not dry_run:
        log.info("Loading docling models...")
        converter = DocumentConverter()
        log.info("Docling models loaded.")

    # Convert index PDF
    if INDEX_PDF_PATH.exists() and not dry_run:
        if force_reconvert or not INDEX_MD_PATH.exists() or INDEX_MD_PATH.stat().st_size == 0:
            log.info("Converting index PDF to markdown...")
            try:
                result = converter.convert(str(INDEX_PDF_PATH))
                md = result.document.export_to_markdown()
                front_matter = (
                    "---\n"
                    f"type: \"index\"\n"
                    f"archive: \"{ARCHIVE_NAME}\"\n"
                    f"source_url: \"{INDEX_PDF_URL}\"\n"
                    "---\n\n"
                )
                INDEX_MD_PATH.write_text(front_matter + md, encoding="utf-8")
                log.info("Wrote %s", INDEX_MD_PATH.name)
            except Exception as e:
                log.error("Failed to convert index PDF: %s", e)

    # Output goes to ground_truth/ to match downstream expectations
    convert_targets = [
        ("descripcion", "descripcion.pdf", "ground_truth/descripcion.md"),
        ("transcripcion", "transcripcion.pdf", "ground_truth/transcripcion.md"),
    ]

    for entry in tqdm(entries, desc="Converting", unit="doc"):
        doc_dir = DOCUMENTS_DIR / entry.dir_name
        all_ok = True

        for doc_type, pdf_name, md_rel in convert_targets:
            pdf_path = doc_dir / pdf_name
            md_path = doc_dir / md_rel

            if not force_reconvert and md_path.exists() and md_path.stat().st_size > 0:
                log.debug("Already converted: %s", md_path)
                continue

            if not pdf_path.exists():
                log.warning("PDF not found: %s", pdf_path)
                entry.errors.append(f"pdf_missing: {pdf_name}")
                all_ok = False
                continue

            if dry_run:
                print(f"  [WOULD CONVERT] {pdf_path.relative_to(BASE_DIR)} -> {md_rel}")
                continue

            try:
                md_content = pdf_to_markdown(pdf_path, doc_type, entry, converter)
                md_path.parent.mkdir(parents=True, exist_ok=True)
                md_path.write_text(md_content, encoding="utf-8")
                log.debug("Converted: %s", md_path)
            except Exception as e:
                log.error("Failed to convert %s: %s", pdf_path, e)
                entry.errors.append(f"convert_failed: {pdf_name}: {e}")
                all_ok = False

        entry.converted = all_ok

    if not dry_run:
        save_catalog(entries)
    return entries


# ---------------------------------------------------------------------------
# Catalog persistence
# ---------------------------------------------------------------------------


def save_catalog(entries: list[DocumentEntry]) -> None:
    """Save catalog to JSON."""
    data = [asdict(e) for e in entries]
    CATALOG_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Saved catalog with %d entries to %s", len(entries), CATALOG_PATH.name)


def load_catalog() -> list[DocumentEntry]:
    """Load catalog from JSON.

    Filters to DocumentEntry fields only, since normalize_datasets.py
    enriches the catalog with extra keys (facsimile_count, doc_id, etc.).
    """
    if not CATALOG_PATH.exists():
        return []
    valid_fields = {f.name for f in DocumentEntry.__dataclass_fields__.values()}
    data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    return [DocumentEntry(**{k: v for k, v in d.items() if k in valid_fields}) for d in data]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Download Toledo Municipal Archive document collection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scrape", action="store_true", help="Phase 1: Scrape catalog from web page")
    parser.add_argument("--download", action="store_true", help="Phase 2: Download PDFs")
    parser.add_argument("--convert", action="store_true", help="Phase 3: Convert PDFs to markdown")
    parser.add_argument("--all", action="store_true", help="Run all phases")
    parser.add_argument("--dry-run", action="store_true", help="Preview without downloading/converting")
    parser.add_argument(
        "--force-reconvert", action="store_true",
        help="Re-convert even if markdown already exists (use when switching converters)",
    )
    parser.add_argument("--start", type=int, default=1, help="Start document number (1-based)")
    parser.add_argument("--end", type=int, default=0, help="End document number (0 = all)")
    args = parser.parse_args()

    if not any([args.scrape, args.download, args.convert, args.all]):
        parser.print_help()
        return

    run_scrape = args.scrape or args.all
    run_download = args.download or args.all
    run_convert = args.convert or args.all

    session = make_session()

    # Phase 1: Scrape
    if run_scrape:
        entries = scrape_catalog(session, dry_run=args.dry_run)
    else:
        entries = load_catalog()
        if not entries:
            log.error("No catalog found. Run with --scrape first.")
            return

    # Apply start/end filter
    if args.end > 0:
        entries = [e for e in entries if args.start <= e.number <= args.end]
    elif args.start > 1:
        entries = [e for e in entries if e.number >= args.start]

    log.info("Working with %d documents (range %d-%d)", len(entries), args.start, args.end or len(entries))

    # Phase 2: Download
    if run_download:
        entries = download_documents(session, entries, dry_run=args.dry_run)

    # Phase 3: Convert
    if run_convert:
        entries = convert_documents(
            entries, dry_run=args.dry_run, force_reconvert=args.force_reconvert,
        )

    # Summary
    if not args.dry_run:
        total = len(entries)
        downloaded = sum(1 for e in entries if e.downloaded)
        converted = sum(1 for e in entries if e.converted)
        with_errors = sum(1 for e in entries if e.errors)
        log.info(
            "Done: %d total, %d downloaded, %d converted, %d with errors",
            total, downloaded, converted, with_errors,
        )
        if with_errors:
            for e in entries:
                if e.errors:
                    log.warning("  %s: %s", e.dir_name, "; ".join(e.errors))


if __name__ == "__main__":
    main()
