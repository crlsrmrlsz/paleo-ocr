#!/usr/bin/env python3
"""Download CODEA+ 2022 procesal escritura documents.

Queries the CODEA+ corpus (corpuscodea.es) for documents whose letra field
contains "procesal" — including compound types like "procesal encadenada",
"bastarde/procesal", etc. Downloads facsimile images and transcriptions
(paleographic + critical) for benchmarking OCR models on XVI century
Spanish handwriting.

Source: https://corpuscodea.es
"""

import argparse
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://corpuscodea.es"
INVENTORY_URL = f"{BASE_URL}/corpus/inventario2.php"
DOCUMENT_URL = f"{BASE_URL}/corpus/documento.php"
FACSIMILE_BASE = f"{BASE_URL}/corpus/facsimiles"

USER_AGENT = (
    "Mozilla/5.0 (compatible; PaleoOCR-Research/1.0; "
    "+https://github.com/fliperbaker/paleo-ocr)"
)
DOWNLOAD_DELAY = 1.0  # seconds between requests
HEAD_TIMEOUT = 10  # seconds for HEAD probes
MAX_PAGES = 20  # max folio pages to probe per document

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_DIR = PROJECT_ROOT / "data" / "corpora" / "codea"
CATALOG_PATH = BASE_DIR / "catalog.json"
DOCUMENTS_DIR = BASE_DIR / "documents"
LOG_PATH = BASE_DIR / "download.log"

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
    """Metadata for a single CODEA+ document."""

    doc_id: str  # "CODEA-0123"
    regesto: str  # Document abstract
    fecha: str  # Date
    siglo: str  # Century
    provincia: str  # Province
    poblacion: str  # City
    tipologia: str  # Document type
    archivo: str  # Archive name
    letra: str  # Script type (procesal, procesal encadenada, etc.)
    ambito: str  # Scope
    soporte: str  # Physical support (paper/parchment)
    copista: str  # Scribe name
    signatura: str  # Shelf mark
    palsclave: str  # Subject keywords
    transcriptor: str  # Transcriber name
    facsimile_pages: list[str] = field(default_factory=list)
    downloaded: bool = False
    errors: list[str] = field(default_factory=list)


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
# Catalog persistence
# ---------------------------------------------------------------------------


def save_catalog(entries: list[DocumentEntry]) -> None:
    """Save catalog to JSON."""
    data = [asdict(e) for e in entries]
    CATALOG_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Saved catalog with %d entries to %s", len(entries), CATALOG_PATH.name)


def load_catalog() -> list[DocumentEntry]:
    """Load catalog from JSON."""
    if not CATALOG_PATH.exists():
        return []
    data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    return [DocumentEntry(**d) for d in data]


# ---------------------------------------------------------------------------
# Phase 1: Query catalog
# ---------------------------------------------------------------------------


def _parse_doc_id(raw: str) -> str:
    """Extract document ID from HTML anchor tag or plain text.

    The API returns docID as HTML like:
      <a class="inventdoc" title="CODEA-0001" ...>CODEA-0001</a>
    """
    m = re.search(r"CODEA-\d+", raw)
    return m.group() if m else raw.strip()


def query_catalog(
    session: requests.Session, dry_run: bool = False
) -> list[DocumentEntry]:
    """Query CODEA+ API for all documents, filter for procesal escritura."""
    log.info("Querying CODEA+ inventory for procesal documents...")

    # The API ignores search_letra, so fetch all and filter client-side
    params = {"pagenum": "0", "pagesize": "5000"}
    resp = session.get(INVENTORY_URL, params=params, timeout=30)
    resp.raise_for_status()

    raw_docs = resp.json()
    log.info("API returned %d total documents", len(raw_docs))

    # Client-side filter: letra must contain "procesal"
    entries: list[DocumentEntry] = []
    letra_values: set[str] = set()
    filtered_count = 0

    for doc in raw_docs:
        letra = doc.get("letra", "")
        if "procesal" not in letra.lower():
            filtered_count += 1
            continue

        letra_values.add(letra)

        doc_id = _parse_doc_id(doc.get("docID", ""))
        entry = DocumentEntry(
            doc_id=doc_id,
            regesto=doc.get("regesto", "").strip(),
            fecha=doc.get("fecha", "").strip(),
            siglo=doc.get("siglo", "").strip(),
            provincia=doc.get("provincia", "").strip(),
            poblacion=doc.get("poblacion", "").strip(),
            tipologia=doc.get("tipologia", "").strip(),
            archivo=doc.get("archivo", "").strip(),
            letra=letra.strip(),
            ambito=doc.get("ambito", "").strip(),
            soporte=doc.get("soporte", "").strip(),
            copista=doc.get("copista", "").strip(),
            signatura=doc.get("signatura", "").strip(),
            palsclave=doc.get("palsclave", "").strip(),
            transcriptor=doc.get("transcriptor", "").strip(),
        )
        entries.append(entry)

    log.info("Filtered out %d non-procesal documents", filtered_count)

    log.info("Validated %d procesal documents", len(entries))
    log.info("Distinct letra values: %s", sorted(letra_values))

    if dry_run:
        print(f"\n{'='*70}")
        print(f"CODEA+ Procesal Documents: {len(entries)} found")
        print(f"Distinct letra types: {sorted(letra_values)}")
        print(f"{'='*70}\n")
        for i, e in enumerate(entries, 1):
            print(f"  {i:3d}. {e.doc_id} | {e.fecha} | {e.letra}")
            print(f"       {e.regesto[:80]}...")
        return entries

    save_catalog(entries)
    return entries


# ---------------------------------------------------------------------------
# Phase 2: Discover facsimiles
# ---------------------------------------------------------------------------


def _facsimile_url(doc_id: str, page: str) -> str:
    """Build facsimile URL for a given document and page label."""
    return f"{FACSIMILE_BASE}/{doc_id}_{page}.jpg"


def discover_facsimiles(
    session: requests.Session,
    entries: list[DocumentEntry],
    dry_run: bool = False,
) -> list[DocumentEntry]:
    """Probe facsimile URLs to discover available pages per document."""
    log.info("Discovering facsimile pages for %d documents...", len(entries))

    page_labels = []
    for n in range(1, MAX_PAGES + 1):
        page_labels.append(f"{n}r")
        page_labels.append(f"{n}v")

    for entry in tqdm(entries, desc="Discovering facsimiles", unit="doc"):
        if entry.facsimile_pages:
            log.debug("Already discovered: %s (%d pages)", entry.doc_id, len(entry.facsimile_pages))
            continue

        found_pages: list[str] = []
        consecutive_misses = 0

        for page in page_labels:
            url = _facsimile_url(entry.doc_id, page)
            try:
                head = session.head(url, timeout=HEAD_TIMEOUT, allow_redirects=True)
                if head.status_code == 200:
                    content_type = head.headers.get("Content-Type", "")
                    if "image" in content_type or content_type == "":
                        found_pages.append(page)
                        consecutive_misses = 0
                        continue
                consecutive_misses += 1
            except requests.RequestException:
                consecutive_misses += 1

            # Stop probing after 4 consecutive misses (likely no more pages)
            if consecutive_misses >= 4:
                break

        entry.facsimile_pages = found_pages
        if not found_pages:
            log.warning("No facsimiles found for %s", entry.doc_id)

    # Summary
    with_facs = sum(1 for e in entries if e.facsimile_pages)
    total_pages = sum(len(e.facsimile_pages) for e in entries)
    log.info(
        "Discovery complete: %d/%d documents have facsimiles (%d total pages)",
        with_facs, len(entries), total_pages,
    )

    if dry_run:
        print(f"\n{'='*70}")
        print(f"Facsimile Discovery: {with_facs}/{len(entries)} docs have images")
        print(f"Total pages: {total_pages}")
        print(f"{'='*70}\n")
        for e in entries:
            status = f"{len(e.facsimile_pages)} pages" if e.facsimile_pages else "NO IMAGES"
            print(f"  {e.doc_id}: {status}")
            if e.facsimile_pages:
                print(f"    Pages: {', '.join(e.facsimile_pages)}")
        return entries

    save_catalog(entries)
    return entries


# ---------------------------------------------------------------------------
# Phase 3: Download
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


def extract_transcriptions(
    session: requests.Session, doc_id: str
) -> list[dict[str, str]]:
    """Fetch document page and extract per-page paleographic and critical transcriptions.

    Returns a list of dicts, one per page:
        [{"page": "1r", "paleographic": "...", "critical": "..."}, ...]
    """
    url = f"{DOCUMENT_URL}?documento={doc_id}"
    pages: list[dict[str, str]] = []

    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Each page is a <tr class="tr-texto"> with td.textopaleo + td.textocritico
        rows = soup.find_all("tr", class_="tr-texto")

        for row in rows:
            paleo_td = row.find("td", class_="textopaleo")
            critic_td = row.find("td", class_="textocritico")

            # Extract page label from the facsimile <img> in the paleographic cell
            page_label = ""
            if paleo_td:
                img = paleo_td.find("img")
                if img and img.get("src"):
                    m = re.search(r"CODEA-\d+_(.+)\.jpg", img["src"])
                    if m:
                        page_label = m.group(1)

            paleo_text = paleo_td.get_text(separator="\n").strip() if paleo_td else ""
            critic_text = critic_td.get_text(separator="\n").strip() if critic_td else ""

            if page_label and (paleo_text or critic_text):
                pages.append({
                    "page": page_label,
                    "paleographic": paleo_text,
                    "critical": critic_text,
                })

        # Fallback: if no tr-texto rows found, try flat class search (single-page docs)
        if not pages:
            paleo_el = soup.find(class_="textopaleo")
            critic_el = soup.find(class_="textocritico")
            paleo_text = paleo_el.get_text(separator="\n").strip() if paleo_el else ""
            critic_text = critic_el.get_text(separator="\n").strip() if critic_el else ""
            if paleo_text or critic_text:
                pages.append({
                    "page": "1r",
                    "paleographic": paleo_text,
                    "critical": critic_text,
                })

    except requests.RequestException as e:
        log.error("Failed to fetch transcription page for %s: %s", doc_id, e)

    return pages


def download_documents(
    session: requests.Session,
    entries: list[DocumentEntry],
    dry_run: bool = False,
) -> list[DocumentEntry]:
    """Download facsimile images and transcriptions for each document."""
    # Filter to documents that have facsimiles
    downloadable = [e for e in entries if e.facsimile_pages]
    log.info(
        "Downloading %d documents (%d skipped — no facsimiles)",
        len(downloadable),
        len(entries) - len(downloadable),
    )

    for i, entry in enumerate(tqdm(downloadable, desc="Downloading", unit="doc")):
        doc_dir = DOCUMENTS_DIR / entry.doc_id
        facs_dir = doc_dir / "facsimiles"
        trans_dir = doc_dir / "transcriptions"

        if dry_run:
            print(f"\n  {entry.doc_id} ({len(entry.facsimile_pages)} pages)")
            for page in entry.facsimile_pages:
                dest = facs_dir / f"{page}.jpg"
                status = "EXISTS" if dest.exists() and dest.stat().st_size > 0 else "WOULD DOWNLOAD"
                print(f"    [{status}] facsimiles/{page}.jpg")
            print(f"    [WOULD FETCH] transcriptions/paleographic_*.txt (per page)")
            print(f"    [WOULD FETCH] transcriptions/critical_*.txt (per page)")
            print(f"    [WOULD WRITE] metadata.json")
            continue

        all_ok = True

        # Download facsimile images
        for page in entry.facsimile_pages:
            url = _facsimile_url(entry.doc_id, page)
            dest = facs_dir / f"{page}.jpg"
            if not download_file(session, url, dest):
                entry.errors.append(f"download_failed: {page}.jpg")
                all_ok = False
            time.sleep(DOWNLOAD_DELAY)

        # Extract per-page transcriptions
        page_transcriptions = extract_transcriptions(session, entry.doc_id)
        time.sleep(DOWNLOAD_DELAY)

        if not page_transcriptions:
            log.warning("No transcriptions found for %s", entry.doc_id)
        else:
            trans_dir.mkdir(parents=True, exist_ok=True)
            for page_entry in page_transcriptions:
                page = page_entry["page"]

                paleo_path = trans_dir / f"paleographic_{page}.txt"
                if page_entry["paleographic"] and not (paleo_path.exists() and paleo_path.stat().st_size > 0):
                    paleo_path.write_text(page_entry["paleographic"], encoding="utf-8")
                    log.debug("Wrote paleographic transcription for %s page %s", entry.doc_id, page)

                critic_path = trans_dir / f"critical_{page}.txt"
                if page_entry["critical"] and not (critic_path.exists() and critic_path.stat().st_size > 0):
                    critic_path.write_text(page_entry["critical"], encoding="utf-8")
                    log.debug("Wrote critical transcription for %s page %s", entry.doc_id, page)

        # Save per-document metadata
        meta_path = doc_dir / "metadata.json"
        if not (meta_path.exists() and meta_path.stat().st_size > 0):
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta = asdict(entry)
            meta.pop("downloaded", None)
            meta.pop("errors", None)
            meta_path.write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        entry.downloaded = all_ok

        # Checkpoint every 10 documents
        if (i + 1) % 10 == 0:
            save_catalog(entries)
            log.info("Checkpoint saved at document %d/%d", i + 1, len(downloadable))

    if not dry_run:
        save_catalog(entries)
    return entries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Download CODEA+ 2022 procesal escritura documents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s --query --dry-run          List procesal documents
  %(prog)s --query --discover         Query + discover facsimiles
  %(prog)s --all --end 3              Download first 3 documents
  %(prog)s --download --start 10      Resume download from 10th entry
""",
    )
    parser.add_argument(
        "--query", action="store_true", help="Phase 1: Query CODEA+ API for procesal documents"
    )
    parser.add_argument(
        "--discover", action="store_true", help="Phase 2: Discover available facsimile pages"
    )
    parser.add_argument(
        "--download", action="store_true", help="Phase 3: Download images and transcriptions"
    )
    parser.add_argument("--all", action="store_true", help="Run all phases")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview without downloading"
    )
    parser.add_argument(
        "--start", type=int, default=1, help="Start from Nth document in catalog (1-based)"
    )
    parser.add_argument(
        "--end", type=int, default=0, help="Stop at Nth document (0 = all)"
    )
    args = parser.parse_args()

    if not any([args.query, args.discover, args.download, args.all]):
        parser.print_help()
        return

    run_query = args.query or args.all
    run_discover = args.discover or args.all
    run_download = args.download or args.all

    session = make_session()

    # Phase 1: Query
    if run_query:
        entries = query_catalog(session, dry_run=args.dry_run)
    else:
        entries = load_catalog()
        if not entries:
            log.error("No catalog found. Run with --query first.")
            return

    # Apply start/end filter (1-based indexing into catalog order)
    total_before_filter = len(entries)
    start_idx = args.start - 1  # convert to 0-based
    end_idx = args.end if args.end > 0 else total_before_filter
    entries = entries[start_idx:end_idx]

    log.info(
        "Working with %d documents (range %d-%d of %d)",
        len(entries), args.start, end_idx, total_before_filter,
    )

    # Phase 2: Discover facsimiles
    if run_discover:
        entries = discover_facsimiles(session, entries, dry_run=args.dry_run)

    # Phase 3: Download
    if run_download:
        entries = download_documents(session, entries, dry_run=args.dry_run)

    # Summary
    if not args.dry_run:
        total = len(entries)
        downloaded = sum(1 for e in entries if e.downloaded)
        with_facs = sum(1 for e in entries if e.facsimile_pages)
        with_errors = sum(1 for e in entries if e.errors)
        log.info(
            "Done: %d total, %d with facsimiles, %d downloaded, %d with errors",
            total, with_facs, downloaded, with_errors,
        )
        if with_errors:
            for e in entries:
                if e.errors:
                    log.warning("  %s: %s", e.doc_id, "; ".join(e.errors))


if __name__ == "__main__":
    main()
