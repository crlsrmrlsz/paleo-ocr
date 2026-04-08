"""Shared Toledo manuscript utilities.

Constants and helpers used by both normalize_gt.py and select_subset.py
for parsing Toledo transcription markdown with folio markers and
handling cover-page offset detection.
"""

import re

# Regex for folio markers with various formats.
# Underscores are optional to support both pymupdf4llm (_italic_) and docling (plain) output.
FOLIO_PATTERN = re.compile(
    r"\[\s*_?(?:Fol\.?|Folio)\s+(\d+)\s*,?\s*(recto|vuelto|vuelta|r|v)_?\s*\]",
    re.IGNORECASE,
)

# Cover page pattern (underscores optional)
COVER_PATTERN = re.compile(r"\[\s*_?Cubierta_?\s*\]", re.IGNORECASE)

# Known image offsets for Toledo docs where auto-detection fails.
# Value = number of pre-content images before Fol. 1 recto.
TOLEDO_DOC_IMAGE_OFFSET = {
    "046_1535-09-24-1539-05-28": 4,   # images 001-004 = cover + blanks
    "073_1617-01-16-1617-08-26": 2,   # images 001-002 = cover + blank
}

# Docs where source newlines are meaningful and should be preserved
# (not collapsed into continuous text split on /)
TOLEDO_NEWLINE_DOCS = {
    "046_1535-09-24-1539-05-28",  # horse registry, two-column layout
}


def estimate_page_offset(gt_text: str, doc_id: str = "") -> int:
    """Detect how many pre-content images precede Fol. 1 recto.

    Returns 0 when page 0001 = Fol. 1 recto, or N for N cover/blank images.

    Key fix: if the first folio marker is Fol. 1 vuelta/vuelto, the text
    before it is Fol. 1 recto content (not a cover page), so offset = 0.
    The old heuristic wrongly returned 1 for docs 082/083 whose text
    starts at Fol. 1 recto without an explicit recto marker.
    """
    # Use known offset if available (overrides heuristic)
    if doc_id and doc_id in TOLEDO_DOC_IMAGE_OFFSET:
        return TOLEDO_DOC_IMAGE_OFFSET[doc_id]

    cover_match = COVER_PATTERN.search(gt_text)
    first_folio = FOLIO_PATTERN.search(gt_text)

    # Explicit cover marker before first folio -> offset 1
    if cover_match and first_folio and cover_match.start() < first_folio.start():
        return 1

    if first_folio:
        folio_num = int(first_folio.group(1))
        side = first_folio.group(2).lower()
        is_recto = side in ("recto", "r")

        # If first marker is Fol 1 vuelta, text before it = Fol 1 recto (no cover)
        if folio_num == 1 and not is_recto:
            return 0

        # If there's substantial text before the first folio marker,
        # it's likely a cover/title page
        if first_folio.start() > 100:
            text_before = gt_text[:first_folio.start()].strip()
            if len(text_before) > 50:
                return 1

    return 0


def extract_toledo_page(text: str, page_num: int) -> str:
    """Extract text for a specific page number from Toledo transcription.

    Page numbering: page 1 = first folio recto (content before first
    vuelto/vuelta marker). Folio markers split the document.

    Mapping: page 1 = folio 1 recto, page 2 = folio 1 vuelto,
    page 3 = folio 2 recto, etc.
    """
    # Find all folio markers with their positions
    markers = []
    for m in FOLIO_PATTERN.finditer(text):
        folio_num = int(m.group(1))
        side = m.group(2).lower()
        is_recto = side in ("recto", "r")
        # Convert to sequential page number:
        # folio 1 recto = 1, folio 1 vuelto = 2, folio 2 recto = 3, etc.
        seq = (folio_num - 1) * 2 + (1 if is_recto else 2)
        markers.append((m.start(), m.end(), seq))

    if not markers:
        # No folio markers: entire text is page 1
        if page_num == 1:
            return text
        return ""

    # Sort by position
    markers.sort(key=lambda x: x[0])

    # Page 1 is everything before the first marker
    if page_num == 1:
        return text[:markers[0][0]]

    # Find the marker that starts this page
    for i, (start, end, seq) in enumerate(markers):
        if seq == page_num:
            # Content from after this marker to the next marker (or end)
            content_start = end
            content_end = markers[i + 1][0] if i + 1 < len(markers) else len(text)
            return text[content_start:content_end]

    # Page not found
    return ""
