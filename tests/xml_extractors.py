"""Extract plain text from PAGE XML and ALTO XML formats.

These are the two standard interchange formats for OCR ground truth:
  - PAGE XML: used by Transkribus, dinglehopper, IMPACT
  - ALTO XML: used by ABBYY, BnF, dinglehopper (alternate)
"""

import xml.etree.ElementTree as ET


# ---------------------------------------------------------------------------
# PAGE XML → plain text
# ---------------------------------------------------------------------------

# PAGE XML namespace (version-agnostic)
_PAGE_NS_PREFIX = "{http://schema.primaresearch.org/PAGE/gts/pagecontent/"


def _get_page_ns(root) -> str:
    """Detect PAGE XML namespace from root tag."""
    tag = root.tag
    if tag.startswith("{"):
        return tag[: tag.index("}") + 1]
    return ""


def _extract_region_lines(region, ns: str) -> list[str]:
    """Extract text lines from a single TextRegion."""
    lines = []
    for text_line in region.findall(f"{ns}TextLine"):
        text_equiv = text_line.find(f"{ns}TextEquiv")
        if text_equiv is not None:
            unicode_elem = text_equiv.find(f"{ns}Unicode")
            if unicode_elem is not None and unicode_elem.text:
                lines.append(unicode_elem.text)
    return lines


def extract_page_xml_text(path: str) -> str:
    """Extract plain text from a PAGE XML file.

    Respects <ReadingOrder> if present: only regions listed in the
    ReadingOrder are included, in the specified order.  Regions not in
    the reading order (headers, page numbers, decorative initials) are
    skipped — matching dinglehopper's extraction behavior.

    Falls back to document-order iteration when no ReadingOrder exists.

    Args:
        path: Path to the PAGE XML file

    Returns:
        Plain text string with lines separated by newlines
    """
    tree = ET.parse(path)
    root = tree.getroot()
    ns = _get_page_ns(root)

    page = root.find(f".//{ns}Page")
    if page is None:
        return ""

    # Build a map of region id → element
    regions_by_id = {}
    for region in page.findall(f"{ns}TextRegion"):
        rid = region.get("id")
        if rid:
            regions_by_id[rid] = region

    # Check for ReadingOrder → OrderedGroup → RegionRefIndexed
    reading_order = root.find(f".//{ns}ReadingOrder")
    if reading_order is not None:
        ordered_group = reading_order.find(f".//{ns}OrderedGroup")
        if ordered_group is not None:
            refs = ordered_group.findall(f"{ns}RegionRefIndexed")
            refs.sort(key=lambda r: int(r.get("index", 0)))
            ordered_ids = [r.get("regionRef") for r in refs]

            lines = []
            for rid in ordered_ids:
                region = regions_by_id.get(rid)
                if region is not None:
                    lines.extend(_extract_region_lines(region, ns))
            return "\n".join(lines)

    # Fallback: iterate all TextLines in document order
    lines = []
    for text_line in root.iter(f"{ns}TextLine"):
        text_equiv = text_line.find(f"{ns}TextEquiv")
        if text_equiv is not None:
            unicode_elem = text_equiv.find(f"{ns}Unicode")
            if unicode_elem is not None and unicode_elem.text:
                lines.append(unicode_elem.text)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ALTO XML → plain text
# ---------------------------------------------------------------------------

_ALTO_NS_PREFIXES = [
    "{http://www.loc.gov/standards/alto/ns-v2#}",
    "{http://www.loc.gov/standards/alto/ns-v3#}",
    "{http://www.loc.gov/standards/alto/ns-v4#}",
    "",  # no namespace
]


def extract_alto_xml_text(path: str) -> str:
    """Extract plain text from an ALTO XML file.

    Processes direct children of each <TextLine> in order:
      - <String>  → append CONTENT attribute
      - <SP>      → append a space character

    This respects ALTO v2/v3 semantics where SP elements are explicit
    word separators.  Adjacent Strings without an SP are concatenated
    directly (e.g. hyphenated fragments across line breaks).

    Args:
        path: Path to the ALTO XML file

    Returns:
        Plain text string with lines separated by newlines
    """
    tree = ET.parse(path)
    root = tree.getroot()

    # Detect namespace from root tag
    ns = ""
    tag = root.tag
    if tag.startswith("{"):
        ns = tag[: tag.index("}") + 1]

    lines = []
    for text_line in root.iter(f"{ns}TextLine"):
        parts = []
        for child in text_line:
            # Strip namespace for tag comparison
            local_tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if local_tag == "String":
                content = child.get("CONTENT", "")
                parts.append(content)
            elif local_tag == "SP":
                parts.append(" ")
        line_text = "".join(parts)
        if line_text:
            lines.append(line_text)

    return "\n".join(lines)
