#!/usr/bin/env python3
"""Download CHARTA corpus vocabulary via the TEI:TOK API.

Fetches all TEI XML documents from the CHARTA corpus
(https://corpora.uah.es/charta/) and extracts unique word forms
across multiple text layers (expanded, normalized, paleographic).

The vocabulary is saved as a JSON file compatible with the coherence
checker's expected format (same structure as impact_es_lexicon.json).

Source: CHARTA en TEITOK — Red Internacional CHARTA
API: https://corpora.uah.es/charta/index.php?action=api
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path
from xml.etree import ElementTree as ET

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_BASE = "https://corpora.uah.es/charta/index.php"
USER_AGENT = (
    "Mozilla/5.0 (compatible; PaleoOCR-Research/1.0; "
    "+https://github.com/fliperbaker/paleo-ocr)"
)
DOWNLOAD_DELAY = 0.5  # seconds between API requests

VOCAB_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "vocabularies"
OUTPUT_FILE = VOCAB_DIR / "charta_vocab.json"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTTP session
# ---------------------------------------------------------------------------

def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def list_documents(session: requests.Session) -> list[str]:
    """Return all document filenames from the CHARTA TEITOK API."""
    resp = session.get(API_BASE, params={"action": "api", "act": "list"})
    resp.raise_for_status()
    data = resp.json()
    return data["files"]


def download_xml(session: requests.Session, doc_id: str) -> str:
    """Download a single document as TEI XML."""
    resp = session.get(
        API_BASE,
        params={"action": "api", "act": "download", "cid": doc_id, "format": "teitok"},
    )
    resp.raise_for_status()
    return resp.text


# ---------------------------------------------------------------------------
# XML parsing — extract word forms
# ---------------------------------------------------------------------------

def extract_forms_from_xml(xml_text: str) -> dict[str, set[str]]:
    """Parse TEI XML and extract word forms from <tok> elements.

    Returns dict with keys 'paleographic', 'expanded', 'normalized',
    each mapping to a set of word forms found in that layer.

    Token structure:
      <tok form="Aug" fform="Augustin" nform="Augustín">Aug<ex>ustin</ex></tok>
      - text content (with <ex> expansions): paleographic as-written
      - form attr: abbreviated paleographic form (when different from text)
      - fform attr: expanded form (abbreviations resolved)
      - nform attr: normalized/critical edition form
    """
    paleo = set()
    expanded = set()
    normalized = set()

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.warning("XML parse error: %s", e)
        return {"paleographic": paleo, "expanded": expanded, "normalized": normalized}

    for tok in root.iter("tok"):
        # Paleographic: the full text content of the element
        text_content = "".join(tok.itertext()).strip()
        if text_content:
            paleo.add(text_content)

        # form attr: abbreviated paleographic form
        form = tok.get("form", "").strip()
        if form and form != "--":
            paleo.add(form)

        # fform: expanded form (abbreviations resolved)
        fform = tok.get("fform", "").strip()
        if fform:
            expanded.add(fform)

        # nform: normalized/critical form
        nform = tok.get("nform", "").strip()
        if nform and nform not in (",", ".", "--", ";", ":"):
            # Skip nforms that contain XML-like supplied tags
            if "<supplied>" not in nform and "<" not in nform:
                normalized.add(nform)

    return {"paleographic": paleo, "expanded": expanded, "normalized": normalized}


# ---------------------------------------------------------------------------
# Vocabulary building
# ---------------------------------------------------------------------------

def _clean_form(word: str) -> str | None:
    """Normalize a word form for vocabulary inclusion.

    Returns None if the form should be excluded (punctuation, numbers,
    multi-word strings, etc).
    """
    word = word.strip().lower()
    # Remove line-break markers (e.g., "ami=go" → "amigo")
    word = word.replace("=", "")
    # Remove HTML entities (e.g., &#9; tab)
    word = re.sub(r'&#\d+;', '', word).strip()
    # Reject multi-word forms (spaces indicate parsing artifacts)
    if " " in word:
        return None
    # Remove forms that are pure punctuation or whitespace
    if not word or all(c in ".,;:!?¿¡-–—()[]{}«»\"'/\\|_=+$&@#" for c in word):
        return None
    # Remove forms containing quotes, $, or stray HTML entities
    if any(c in word for c in '"$'):
        return None
    # Normalize &amp; artifacts from XML
    word = word.replace("&amp;", "&")
    # Remove numbers and numeric-like forms (prices, ordinals, dates)
    if re.match(r'^[\d.,/:;ªº]+$', word):
        return None
    # Remove forms starting with digit (e.g. "13de") or containing digits
    if word and word[0].isdigit():
        return None
    if re.search(r'[\d\u00b9\u00b2\u00b3\u2070-\u2079]', word):
        return None
    # Remove very short forms (single char except common words)
    if len(word) == 1 and word not in {"a", "e", "o", "y", "i", "u"}:
        return None
    # Remove forms shorter than 2 chars that aren't common words
    if len(word) == 2 and word not in {
        "al", "de", "el", "en", "es", "la", "le", "lo", "me", "mi",
        "no", "ni", "os", "se", "si", "su", "te", "tu", "un", "ya",
        "yo", "do", "ha", "he", "ir",
    }:
        return None
    return word


def build_vocabulary(
    all_forms: dict[str, set[str]],
    layers: tuple[str, ...] = ("expanded", "normalized"),
) -> list[str]:
    """Build a sorted, deduplicated vocabulary from selected layers.

    Args:
        all_forms: dict mapping layer names to sets of raw word forms
        layers: which layers to include in the vocabulary
    """
    vocab = set()
    for layer in layers:
        for form in all_forms.get(layer, set()):
            cleaned = _clean_form(form)
            if cleaned:
                vocab.add(cleaned)
    return sorted(vocab)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layers",
        nargs="+",
        default=["paleographic", "expanded", "normalized"],
        choices=["paleographic", "expanded", "normalized"],
        help="Which text layers to include (default: all three)",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_FILE,
        help=f"Output JSON path (default: {OUTPUT_FILE})",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit number of documents to download (0 = all)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    session = _make_session()

    # 1. List all documents
    logger.info("Fetching document list from CHARTA API...")
    doc_ids = list_documents(session)
    logger.info("Found %d documents", len(doc_ids))

    if args.limit:
        doc_ids = doc_ids[: args.limit]
        logger.info("Limiting to %d documents", len(doc_ids))

    # 2. Download and parse each document
    all_forms: dict[str, set[str]] = {
        "paleographic": set(),
        "expanded": set(),
        "normalized": set(),
    }
    errors = 0

    for doc_id in tqdm(doc_ids, desc="Downloading"):
        try:
            xml_text = download_xml(session, doc_id)
            forms = extract_forms_from_xml(xml_text)
            for layer in all_forms:
                all_forms[layer].update(forms[layer])
        except Exception as e:
            logger.warning("Failed to process %s: %s", doc_id, e)
            errors += 1

        time.sleep(DOWNLOAD_DELAY)

    # 3. Build vocabulary
    layers = tuple(args.layers)
    vocab = build_vocabulary(all_forms, layers=layers)

    logger.info("Layer stats:")
    for layer in ("paleographic", "expanded", "normalized"):
        logger.info("  %s: %d raw forms", layer, len(all_forms[layer]))
    logger.info("Final vocabulary (%s): %d unique word forms", "+".join(layers), len(vocab))
    if errors:
        logger.warning("%d documents failed to download", errors)

    # 4. Save
    output = {
        "metadata": {
            "source": "CHARTA — Red Internacional CHARTA (Corpus Hispánico y Americano en la Red: Textos Antiguos)",
            "source_url": "https://corpora.uah.es/charta/",
            "license": "CC BY-NC-ND (educational and research use)",
            "citation": "Red Internacional CHARTA. Corpus CHARTA en TEITOK. Universidad de Alcalá / GITHE.",
            "period": "XII-XIX centuries",
            "description": (
                f"Vocabulary extracted from {len(doc_ids) - errors} TEI XML documents "
                f"({len(doc_ids)} attempted, {errors} errors) in the CHARTA corpus. "
                f"Layers included: {', '.join(layers)}. "
                f"Documents span historical Spanish texts from the Iberian Peninsula "
                f"and Latin America (legislative, judicial, commercial, private)."
            ),
            "date_downloaded": time.strftime("%Y-%m-%d"),
            "layers_included": list(layers),
            "documents_processed": len(doc_ids) - errors,
        },
        "stats": {
            "unique_word_forms": len(vocab),
            "paleographic_raw": len(all_forms["paleographic"]),
            "expanded_raw": len(all_forms["expanded"]),
            "normalized_raw": len(all_forms["normalized"]),
        },
        "word_forms": vocab,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()
