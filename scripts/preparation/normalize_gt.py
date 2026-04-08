#!/usr/bin/env python3
"""Normalize ground truth transcriptions for CER evaluation.

Reads per-subset manifests and produces clean .txt files per page per edition:
  - data/subsets/codea/ground_truth/paleographic/{page_id}.txt
  - data/subsets/codea/ground_truth/critical/{page_id}.txt
  - data/subsets/toledo/ground_truth/editorial/{page_id}.txt

Supports CLI filtering by --subset and --edition.

Two format-specific parsers:
  - CODEA: paleographic/critical edition with multi-line abbreviation encoding
  - Toledo: single-document markdown with editorial annotations

Output format: one manuscript line per text line, abbreviations expanded
inline, no structural markup, original orthography preserved.
"""

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SUBSETS_DIR = DATA_DIR / "subsets"

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "preparation"))
from pipeline_config import VALID_COMBOS  # noqa: E402
from toledo_common import (  # noqa: E402
    COVER_PATTERN,
    FOLIO_PATTERN,
    TOLEDO_DOC_IMAGE_OFFSET,
    TOLEDO_NEWLINE_DOCS,
    estimate_page_offset,
    extract_toledo_page,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Normalize ground truth transcriptions for CER evaluation",
    )
    parser.add_argument(
        "--subset", choices=["codea", "toledo", "all"], default="all",
        help="Which dataset subset to process (default: all)",
    )
    parser.add_argument(
        "--edition", choices=["paleographic", "critical", "editorial", "all"],
        default="all",
        help="Which GT edition to normalize (default: all)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# CODEA edition parser
# ---------------------------------------------------------------------------

def parse_codea_edition(text: str) -> str:
    """Parse CODEA transcription format (works for both paleographic and critical).

    Both editions use the same structural encoding — multi-line abbreviation
    expansion format with {h. Nr} headers and bare line numbers. Only the text
    content differs (paleographic preserves original spelling, critical modernizes).

    Steps:
      1. Remove page headers
      2. Identify line numbers and reconstruct manuscript lines by
         concatenating all text between consecutive line numbers
      3. Strip bracket annotations
      4. Handle illegible markers, uncertain readings
      5. Normalize whitespace and Unicode
    """
    lines = text.split("\n")
    manuscript_lines = []
    current_chunks = []
    in_header = True

    for line in lines:
        stripped = line.rstrip()

        # Skip page header {h. Nr}
        if re.match(r"\{h\.\s*\w+\}", stripped):
            in_header = False
            continue

        # Skip empty lines at the very start
        if in_header and stripped == "":
            continue
        in_header = False

        # Check if this line is a bare line number
        if re.match(r"^\d+$", stripped.strip()):
            # Save previous manuscript line
            if current_chunks:
                joined = "".join(current_chunks)
                manuscript_lines.append(joined)
                current_chunks = []
            continue

        # Accumulate text for current manuscript line
        current_chunks.append(stripped)

    # Don't forget the last manuscript line
    if current_chunks:
        joined = "".join(current_chunks)
        manuscript_lines.append(joined)

    # Process each manuscript line
    cleaned_lines = []
    for ml in manuscript_lines:
        cleaned = _clean_codea_line(ml)
        if cleaned.strip():
            cleaned_lines.append(cleaned.strip())

    return "\n".join(cleaned_lines)



def _clean_codea_line(text: str) -> str:
    """Clean a single reconstructed CODEA manuscript line.

    Preserves unified marker annotations that represent visual features
    in the manuscript image. Strips only editorial/structural markup.
    """
    # --- Strip editorial wrappers, keep base annotations ---

    # [margen mano N: TEXT] → [margen: TEXT] (modifier after marker name)
    # Must come BEFORE the [mano N:] stripping below
    text = re.sub(
        r"\s+mano\s+\d+\s*:\s*", ": ", text, flags=re.IGNORECASE
    )

    # [mano N: TEXT] → TEXT (strip wrapper entirely when mano is the
    # outermost annotation — the [ is removed so no orphaned bracket)
    text = re.sub(
        r"\[mano\s+\d+\s*:\s*", "", text, flags=re.IGNORECASE
    )

    # --- Normalize variant forms to unified markers ---

    # [firmas: TEXT] → [firma: TEXT] (normalize plural)
    text = re.sub(r"\[firmas\s*:", "[firma:", text, flags=re.IGNORECASE)

    # [rubrica] → [rúbrica] (normalize missing accent)
    text = re.sub(r"\[rubrica\]", "[rúbrica]", text, flags=re.IGNORECASE)

    # (tachado TEXT) → [tachado: TEXT] (normalize CODEA variant without brackets)
    text = re.sub(
        r"\(tachado\s+(.*?)\)", r"[tachado: \1]", text, flags=re.IGNORECASE
    )

    # --- Markers to PRESERVE (no-ops, listed for clarity) ---
    # [firma: TEXT]        — signature (readable text)
    # [rúbrica]            — decorative flourish
    # [cruz]               — cross symbol
    # [signo]              — notarial symbol
    # [tachado: TEXT]      — crossed-out text
    # [margen: TEXT]       — marginal notes
    # [interlineado: TEXT] — interlinear additions
    # [sobrescrito: TEXT]  — overwritten text
    # [encabezamiento: T]  — document header
    # [roto]               — torn/damaged

    # --- Markers to TRANSFORM ---

    # [lat.: TEXT] → keep TEXT only (Latin is readable text in image)
    text = re.sub(r"\[lat\.\s*:(.*?)\]", r"\1", text, flags=re.IGNORECASE)

    # [blanco ...] with extra description → [blanco]
    text = re.sub(r"\[blanco\s+[^\]]+\]", "[blanco]", text, flags=re.IGNORECASE)
    # [blanco] → kept as-is (already correct form)

    # [* * *] or [***] → [ilegible]
    text = re.sub(r"\[\*[\s\*]*\*\]", "[ilegible]", text)

    # **** (inline illegible, 3+ asterisks) → [ilegible]
    text = re.sub(r"\*{3,}", "[ilegible]", text)

    # --- Strip editorial markers (not visible in image) ---

    # Remove ≤ (uncertain reading marker)
    text = text.replace("≤", "")

    # Remove | (mid-word break marker)
    text = text.replace("|", "")

    # Remove / at end of line (original line break marker)
    text = re.sub(r"\s*/\s*$", "", text)

    # Remove orphaned ] from [mano N:] wrappers that span multiple lines
    if text.count("]") > text.count("["):
        text = re.sub(r"\](?=\s*$)", "", text)

    # Collapse multiple spaces
    text = re.sub(r"  +", " ", text)

    return text


# ---------------------------------------------------------------------------
# Toledo parser
# ---------------------------------------------------------------------------


def _extract_all_footnotes(text: str) -> tuple[list[tuple[int, str, int]], str]:
    """Extract footnotes from the full document text.

    Returns:
        (footnotes, cleaned_text) where footnotes is a list of
        (ref_num, content, char_position) tuples.
    """
    lines = text.split("\n")
    content_lines = []
    footnotes = []
    in_footnote = False
    # Track char position of each footnote in the original text
    char_pos = 0
    line_positions = []
    for line in lines:
        line_positions.append(char_pos)
        char_pos += len(line) + 1  # +1 for \n

    for i, line in enumerate(lines):
        stripped = line.strip()
        m = re.match(r"^(\d+)\s+([A-ZÁÉÍÓÚ].*)", stripped)
        if m and len(stripped) > 30:
            ref_num = int(m.group(1))
            content = m.group(2)
            footnotes.append((ref_num, content, line_positions[i]))
            in_footnote = True
        elif in_footnote and stripped and stripped[0].islower():
            footnotes[-1] = (footnotes[-1][0],
                             footnotes[-1][1] + " " + stripped,
                             footnotes[-1][2])
            if stripped.endswith((".", "。")):
                in_footnote = False
        else:
            in_footnote = False
            content_lines.append(line)

    cleaned_text = "\n".join(content_lines)
    return footnotes, cleaned_text


def _assign_footnotes_to_pages(
    text: str,
    footnotes: list[tuple[int, str, int]],
) -> dict[int, list[tuple[int, str]]]:
    """Assign footnotes to pages based on where their reference numbers appear.

    Searches for each footnote's reference number (/ N, wordN) in the text
    before the footnote's position, then determines which page section
    contains that reference using folio marker ranges.
    """
    # Build page ranges from folio markers
    markers = []
    for m in FOLIO_PATTERN.finditer(text):
        folio_num = int(m.group(1))
        side = m.group(2).lower()
        is_recto = side in ("recto", "r")
        seq = (folio_num - 1) * 2 + (1 if is_recto else 2)
        markers.append((m.start(), seq))
    markers.sort(key=lambda x: x[0])

    def _page_at_pos(pos: int) -> int:
        """Return sequential page number for a character position."""
        page = 1
        for marker_pos, seq in markers:
            if pos < marker_pos:
                break
            page = seq
        return page

    # Build set of folio marker ranges to exclude from reference search
    folio_ranges = set()
    for m in FOLIO_PATTERN.finditer(text):
        for i in range(m.start(), m.end()):
            folio_ranges.add(i)

    result: dict[int, list[tuple[int, str]]] = {}
    for ref_num, content, fn_pos in footnotes:
        # Search for the reference in main text before the footnote.
        # Use a safety margin to avoid matching the footnote's own line.
        search_end = fn_pos - 5
        if search_end < 0:
            search_end = 0
        search_region = text[:search_end]
        # Try patterns: "/ N", "wordN\s", standalone " N " or "> N"
        ref_patterns = [
            rf"/\s*{ref_num}\s",
            rf"\w{ref_num}\s",
            rf"[>;]\s*{ref_num}[\s,]",
            rf"\s{ref_num}[\s,]",
        ]
        last_match_pos = -1
        for pat in ref_patterns:
            for m in re.finditer(pat, search_region):
                # Skip matches inside folio markers
                if m.start() in folio_ranges:
                    continue
                if m.start() > last_match_pos:
                    last_match_pos = m.start()

        if last_match_pos >= 0:
            page = _page_at_pos(last_match_pos)
        else:
            # Fallback: assign to the page just before the footnote
            page = _page_at_pos(fn_pos)

        result.setdefault(page, []).append((ref_num, content))

    return result


def parse_toledo_transcription(text: str, target_page: str, doc_id: str = "") -> str:
    """Parse Toledo transcription markdown for a specific page.

    Steps:
      1. Strip YAML frontmatter
      2. Extract footnotes from the full document and assign to pages
      3. Split by folio markers to isolate the target page
      4. Strip editorial annotations
      5. Resolve abbreviation brackets
      6. Handle line breaks and word continuations
      7. Normalize whitespace and Unicode
    """
    # Strip YAML frontmatter
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            text = text[end + 3:].strip()

    # Determine which folio section to extract, accounting for cover offset
    page_num = int(target_page.lstrip("0") or "1")

    # Detect cover page offset
    page_offset = estimate_page_offset(text, doc_id)
    adjusted_page = page_num - page_offset

    if adjusted_page < 1:
        # This is the cover page — return empty
        return ""

    # Extract all footnotes from the full document and assign to pages
    all_footnotes, text_no_fn = _extract_all_footnotes(text)
    if all_footnotes:
        page_footnotes = _assign_footnotes_to_pages(text, all_footnotes)
        # Use text with footnote lines removed for page extraction
        page_text = extract_toledo_page(text_no_fn, adjusted_page)
        assigned = page_footnotes.get(adjusted_page, [])
    else:
        page_text = extract_toledo_page(text, adjusted_page)
        assigned = []

    # Clean the extracted page text with pre-assigned footnotes
    return _clean_toledo_text(page_text, doc_id=doc_id, footnotes=assigned)


def _clean_toledo_text(
    text: str,
    doc_id: str = "",
    footnotes: list[tuple[int, str]] | None = None,
) -> str:
    """Clean Toledo transcription text, preserving manuscript markers.

    The `/` character marks actual manuscript line breaks. Newlines in the
    markdown are extraction artifacts (column-wrap from pymupdf4llm, or
    paragraph breaks from docling). We must:
      1. Normalize format (strip headings, tables, rejoin broken markers)
      2. Remove footnotes (or use pre-assigned footnotes)
      3. Convert editorial annotations to unified marker format
      4. Join all extraction-wrapped lines into continuous text
      5. Handle word continuations across line breaks
      6. Split on `/` to recover manuscript lines
    """
    # --- Step 0: Normalize docling format artifacts ---

    # Fix spurious markdown list markers from docling (consumes first char)
    text = re.sub(r"\n\n- ([a-záéíóúñçü])", r"\n\1", text)

    # Strip markdown heading markers (docling adds ## to some lines)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Rejoin broken folio markers split across lines (e.g. "[\nFol. 2 recto\n]")
    text = re.sub(
        r"\[\s*\n\s*((?:Fol\.?|Folio)\s+\d+\s*,?\s*(?:recto|vuelto|vuelta|r|v))\s*\n\s*\]",
        r"[ \1 ]",
        text,
        flags=re.IGNORECASE,
    )

    # Extract text from markdown tables (docling renders two-column layouts as tables)
    # | Name | Description | → "Name Description"
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip table separator rows (|---|---|)
        if re.match(r"^\|[-|\s:]+\|$", stripped):
            continue
        # Extract cell contents from table rows
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            joined = " ".join(c for c in cells if c)
            cleaned_lines.append(joined)
        else:
            cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)

    # --- Step 1: Handle footnotes ---
    if footnotes is not None:
        # Pre-assigned footnotes from whole-document extraction.
        # Just strip reference numbers from the page text.
        for ref_num, _ in footnotes:
            text = re.sub(rf"(/\s*){ref_num}\s*$", r"\1", text, flags=re.MULTILINE)
            text = re.sub(rf"(/\s*){ref_num}(\s)", r"\1\2", text)
            text = re.sub(rf"(\w){ref_num}(\s)", r"\1\2", text)
    else:
        # Legacy path: extract footnotes from this page's text
        lines = text.split("\n")
        content_lines = []
        footnotes = []
        in_footnote = False
        for line in lines:
            stripped = line.strip()
            m = re.match(r"^(\d+)\s+([A-ZÁÉÍÓÚ].*)", stripped)
            if m and len(stripped) > 30:
                ref_num = int(m.group(1))
                content = m.group(2)
                footnotes.append((ref_num, content))
                in_footnote = True
            elif in_footnote and stripped and stripped[0].islower():
                footnotes[-1] = (footnotes[-1][0],
                                 footnotes[-1][1] + " " + stripped)
                if stripped.endswith((".", "。")):
                    in_footnote = False
            else:
                in_footnote = False
                content_lines.append(line)
        text = "\n".join(content_lines)

        for ref_num, _ in footnotes:
            text = re.sub(rf"(/\s*){ref_num}\s*$", r"\1", text, flags=re.MULTILINE)
            text = re.sub(rf"(/\s*){ref_num}(\s)", r"\1\2", text)
            text = re.sub(rf"(\w){ref_num}(\s)", r"\1\2", text)

    # --- Step 2: Convert editorial annotations to unified markers ---
    # All patterns use _? to handle both pymupdf4llm (_italic_) and docling (plain).

    # ( _Cruz_ ) or ( Cruz ) → [cruz]
    text = re.sub(r"\(\s*_?Cruz_?\s*\)", "[cruz]", text, flags=re.IGNORECASE)

    # ( _rúbrica_ ) or ( rúbrica ) → [rúbrica]
    text = re.sub(
        r"\(\s*_?[Rr]úbrica_?\s*\)", "[rúbrica]", text
    )

    # < TEXT > or < TEXT > N → [interlineado: TEXT]
    # "Paréntesis triangulares" mark interlinear additions in Toledo transcriptions.
    # Optional trailing number is a footnote reference consumed by this rule.
    text = re.sub(
        r"<\s*(.*?)\s*>\s*(?:\d+\s*)?",
        lambda m: _make_marker("interlineado", m.group(1)) + " " if m.group(1).strip() else "",
        text,
    )

    # [ _Al margen_ ] or [ Al margen ] TEXT → [margen: TEXT] (text to end of line)
    text = re.sub(
        r"\[\s*_?Al margen_?\s*\]\s*([^\n]*)",
        lambda m: _make_marker("margen", m.group(1)),
        text,
        flags=re.IGNORECASE,
    )

    # Remove folio markers (any that remain after page splitting)
    text = FOLIO_PATTERN.sub("", text)

    # [ _tachado_ : TEXT] or [ tachado: TEXT] → [tachado: TEXT]
    text = re.sub(
        r"\[\s*_?[Tt]achado:?_?\s*[:,]?\s*([^\]]*?)\s*\]",
        lambda m: _make_marker("tachado", m.group(1)),
        text,
    )

    # [ _sic:_ word] or [ sic : word] → keep the corrected word
    text = re.sub(r"\[\s*_?sic:?_?\s*:?\s*([^\]]+)\]", r"\1", text)

    # Bare [sic] (no corrected word) → [sic] marker preserved
    text = re.sub(r"\[\s*_?sic_?\s*\]", "[sic]", text)

    # [ _Brevete_ ] or [ Brevete ] → keep text content only
    # [ _he_ ] or [ he ] etc → keep letter/word content only
    text = re.sub(r"\[\s*_([^_]+)_\s*\]", r"\1", text)

    # Normalize whitespace inside editorial brackets: [ e ]llo → [e]llo
    # Only for short letter additions (1-5 lowercase Spanish letters), not named markers.
    text = re.sub(
        r"\[\s*(?!(?:cruz|rúbrica|signo|ilegible|blanco|roto|nota|sic|interlineado|tachado|margen|firma|sobrescrito|encabezamiento)\s*[:\]])"
        r"([a-záéíóúñç]{1,5})\s*\]",
        r"[\1]",
        text,
    )

    # Remove footnote references [N] where N is a number
    text = re.sub(r"\[(\d+)\]", "", text)

    # Handle _/_ (italic slash used as line break in some docs)
    text = text.replace("_/_", "/")

    # --- Alternate path for newline-preserving docs (e.g. 046 horse registry) ---
    if doc_id in TOLEDO_NEWLINE_DOCS:
        return _clean_toledo_newline_doc(text, footnotes)

    # --- Step 3: Join extraction-wrapped lines into continuous text ---
    text = re.sub(r"\n\n+", " ¶ ", text)
    text = re.sub(r"\n", " ", text)

    # --- Step 4: Remove transcriber continuation hyphens, keep / line breaks ---
    # The transcriber adds "-" before "/" to show a word continues on the next
    # manuscript line, but the hyphen is NOT in the original document.
    # Remove the hyphen; the "/" is preserved for Step 5 to split into lines.
    # Also handles cross-paragraph cases: rre- ¶ /ceviré (¶ from double newlines)
    text = re.sub(r"-(\s*(?:¶\s*)?/)", r"\1", text)
    text = re.sub(r"(/\s*)-", r"\1", text)

    # --- Step 5: Split on / to get manuscript lines ---
    text = _protect_marker_slashes(text)
    parts = text.split("/")
    manuscript_lines = []
    for part in parts:
        line = part.strip()
        if not line or line == "¶":
            continue
        line = line.replace("¶", "").strip()
        line = line.replace("\x00", "/")
        if line:
            manuscript_lines.append(line)

    # Append footnote markers at end
    for _, content in footnotes:
        marker = _make_marker("nota", content)
        if marker:
            manuscript_lines.append(marker)

    return "\n".join(manuscript_lines)


def _clean_toledo_newline_doc(
    text: str,
    footnotes: list[tuple[int, str]],
) -> str:
    """Clean Toledo docs where source newlines are meaningful (e.g. doc 046).

    These docs have single \\n between all lines with no / between entries.
    We preserve source lines and insert blank lines between registry entries.
    """
    # Strip trailing / from lines (page boundary markers)
    lines = [line.rstrip().rstrip("/").rstrip() for line in text.split("\n")]

    # Remove folio markers and (Cruz) lines
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if FOLIO_PATTERN.match(stripped):
            continue
        if re.match(r"^\(?\s*Cruz\s*\)?$", stripped, re.IGNORECASE):
            continue
        cleaned.append(stripped)

    # Remove leading/trailing blank lines
    while cleaned and not cleaned[0]:
        cleaned.pop(0)
    while cleaned and not cleaned[-1]:
        cleaned.pop()

    # Detect entry boundaries and insert blank lines.
    # Each entry pattern: optional name-only lines → description starting with
    # "En [date]" or "Este dicho día"
    date_start_pattern = re.compile(
        r"^(En\s+[IVXLCDM]+\s+de\s+|Este\s+dicho\s+d[ií]a)", re.IGNORECASE
    )
    result_lines = []
    for i, line in enumerate(cleaned):
        if not line:
            continue
        # Insert blank line before an entry if:
        # - This line matches a date pattern AND
        # - It's not the very first content line
        if i > 0 and date_start_pattern.match(line) and result_lines:
            # Check if previous non-blank line was a name (short, no date pattern)
            # If so, the blank line goes before the name, not the date
            # Look back to find where the name lines start
            j = len(result_lines) - 1
            while j >= 0 and result_lines[j] and not date_start_pattern.match(result_lines[j]):
                prev = result_lines[j]
                # Name lines are typically short and don't start with date patterns
                # or known continuation words
                if len(prev) > 60 or prev.endswith((".","menos.", "cerrado.")):
                    break
                j -= 1
            # Insert blank line before the name group
            if j + 1 < len(result_lines) and result_lines[j + 1]:
                result_lines.insert(j + 1, "")
            elif result_lines[-1]:
                result_lines.append("")

        result_lines.append(line)

    # Append footnote markers
    for _, content in footnotes:
        marker = _make_marker("nota", content)
        if marker:
            result_lines.append(marker)

    return "\n".join(result_lines)


def _make_marker(name: str, text: str) -> str:
    """Create a unified marker, stripping / line breaks from content."""
    text = text.replace("/", " ").strip()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return ""
    return f"[{name}: {text}]"


def _protect_marker_slashes(text: str) -> str:
    """Replace / inside matched [...] brackets with a null byte placeholder."""
    # Find matched bracket pairs using a stack
    stack = []
    ranges = []
    for i, ch in enumerate(text):
        if ch == "[":
            stack.append(i)
        elif ch == "]" and stack:
            ranges.append((stack.pop(), i))

    # Replace / only inside matched bracket pairs
    result = list(text)
    for start, end in ranges:
        for i in range(start + 1, end):
            if result[i] == "/":
                result[i] = "\x00"
    return "".join(result)


# ---------------------------------------------------------------------------
# Unicode normalization (shared)
# ---------------------------------------------------------------------------

def normalize_unicode(text: str) -> str:
    """Apply Unicode NFC normalization and clean up whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u2013", "-")  # en dash
    text = text.replace("\u2014", "-")  # em dash
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_entry(entry: dict, edition: str) -> tuple[str, str]:
    """Process a single evaluation subset entry for a specific edition.

    Returns (page_id, normalized_text).
    """
    dataset = entry["dataset"]
    page_id = entry["page_id"]
    gt_paths = entry.get("gt_paths", {})

    if edition not in gt_paths:
        return page_id, ""

    gt_path = PROJECT_ROOT / gt_paths[edition]

    if not gt_path.exists():
        print(f"  WARNING: GT file not found: {gt_path}", file=sys.stderr)
        return page_id, ""

    raw_text = gt_path.read_text(encoding="utf-8")

    if dataset == "codea":
        text = parse_codea_edition(raw_text)
    elif dataset == "toledo":
        text = parse_toledo_transcription(raw_text, entry["page"], entry.get("doc_id", ""))
    else:
        print(f"  WARNING: Unknown dataset: {dataset}", file=sys.stderr)
        return page_id, ""

    text = normalize_unicode(text)
    return page_id, text


def process_subset(subset: str, edition: str, manifest: list[dict]):
    """Process all entries for a given subset/edition combo."""
    gt_output_dir = SUBSETS_DIR / subset / "ground_truth" / edition
    gt_output_dir.mkdir(parents=True, exist_ok=True)

    entries = [e for e in manifest if e["dataset"] == subset]
    print(f"\n  {subset}/{edition}: {len(entries)} entries → {gt_output_dir.relative_to(PROJECT_ROOT)}")

    success = 0
    empty = 0
    errors = 0

    for entry in entries:
        page_id = entry["page_id"]
        try:
            _, text = process_entry(entry, edition)
            if not text.strip():
                print(f"    EMPTY: {page_id}")
                empty += 1
            else:
                success += 1

            out_path = gt_output_dir / f"{page_id}.txt"
            out_path.write_text(text, encoding="utf-8")

        except Exception as e:
            print(f"    ERROR: {page_id}: {e}", file=sys.stderr)
            errors += 1

    print(f"    Done: {success} ok, {empty} empty, {errors} errors")
    return success, empty, errors


def main():
    args = parse_args()

    # Determine which subsets to process
    if args.subset == "all":
        subsets = list(VALID_COMBOS.keys())
    else:
        subsets = [args.subset]

    # Load manifests
    manifests = {}
    for subset in subsets:
        manifest_path = SUBSETS_DIR / subset / "manifest.json"
        if not manifest_path.exists():
            print(f"WARNING: Manifest not found: {manifest_path}", file=sys.stderr)
            print("  Run select_subset.py first to generate per-subset manifests.", file=sys.stderr)
            continue
        with open(manifest_path) as f:
            manifests[subset] = json.load(f)

    if not manifests:
        print("No manifests found. Run select_subset.py first.")
        sys.exit(1)

    total_entries = sum(len(m) for m in manifests.values())
    print(f"Normalizing ground truth for {total_entries} pages...")

    total_success = 0
    total_empty = 0
    total_errors = 0

    for subset, manifest in manifests.items():
        valid_editions = VALID_COMBOS[subset]
        if args.edition == "all":
            editions = valid_editions
        elif args.edition in valid_editions:
            editions = [args.edition]
        else:
            print(f"  Skipping {subset}: edition '{args.edition}' not valid (valid: {valid_editions})")
            continue

        for edition in editions:
            s, e, err = process_subset(subset, edition, manifest)
            total_success += s
            total_empty += e
            total_errors += err

    print(f"\nTotal: {total_success} ok, {total_empty} empty, {total_errors} errors")

    # Print sample for spot-checking
    if total_success > 0:
        for subset in subsets:
            if subset not in manifests:
                continue
            for edition in VALID_COMBOS[subset]:
                gt_dir = SUBSETS_DIR / subset / "ground_truth" / edition
                if not gt_dir.exists():
                    continue
                sample_files = sorted(gt_dir.glob("*.txt"))[:2]
                for sf in sample_files:
                    content = sf.read_text()[:200]
                    print(f"\n--- {subset}/{edition}/{sf.name} (first 200 chars) ---")
                    print(content)


if __name__ == "__main__":
    main()
