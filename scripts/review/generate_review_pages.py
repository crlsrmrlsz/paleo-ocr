#!/usr/bin/env python3
"""Generate side-by-side HTML review pages for GT/image comparison.

Produces one HTML file per subset with a clickable table of contents,
manuscript facsimile images alongside their ground truth transcriptions.

  --subset {codea,toledo,all}

Output:
  data/evaluation/review/codea_review.html
      3-col × 2-row grid: facsimile (spanning both rows) |
      critical source / paleographic source |
      critical normalized / paleographic normalized
  data/evaluation/review/toledo_review.html  (2 columns: image | editorial)
"""

import argparse
import difflib
import html
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SUBSETS_DIR = PROJECT_ROOT / "data" / "subsets"
REVIEW_DIR = PROJECT_ROOT / "data" / "evaluation" / "review"

# Import shared pipeline config and Toledo helpers
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from pipeline_config import VALID_COMBOS, load_manifest  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "preparation"))
from toledo_common import estimate_page_offset, extract_toledo_page  # noqa: E402

# Metadata fields to display (in order), with human-readable labels
METADATA_FIELDS = [
    ("doc_id", "Document"),
    ("page", "Page"),
    ("siglo", "Century"),
    ("letra_normalized", "Script type"),
    ("tipologia", "Typology"),
    ("copista", "Scribe"),
]


def load_gt_text(subset: str, edition: str, page_id: str) -> str:
    """Read normalized ground truth text file. Returns empty string if missing."""
    gt_path = SUBSETS_DIR / subset / "ground_truth" / edition / f"{page_id}.txt"
    if gt_path.exists():
        return gt_path.read_text(encoding="utf-8")
    return ""


def load_raw_gt_text(entry: dict, edition: str) -> str:
    """Read raw (source) ground truth text for an entry.

    For CODEA: reads the per-page file directly (already one file per page).
    For Toledo: reads the shared transcripcion.md and extracts the target
    page's section without applying any cleaning/normalization.
    """
    gt_paths = entry.get("gt_paths", {})
    if edition not in gt_paths:
        return ""

    gt_path = PROJECT_ROOT / gt_paths[edition]
    if not gt_path.exists():
        return ""

    raw_text = gt_path.read_text(encoding="utf-8")

    dataset = entry.get("dataset", "")
    if dataset == "codea":
        # Per-page file — return as-is
        return raw_text

    if dataset == "toledo":
        # Shared file — strip YAML frontmatter and extract target page
        text = raw_text
        if text.startswith("---"):
            end = text.find("---", 3)
            if end != -1:
                text = text[end + 3:].strip()

        page_num = int(entry["page"].lstrip("0") or "1")
        page_offset = estimate_page_offset(text, entry.get("doc_id", ""))
        adjusted_page = page_num - page_offset

        if adjusted_page < 1:
            return ""

        return extract_toledo_page(text, adjusted_page).strip()

    return raw_text


def _diff_words(words_a: list[str], words_b: list[str]) -> tuple[list[str], list[str]]:
    """Diff two word lists, returning HTML-annotated word lists."""
    sm = difflib.SequenceMatcher(None, words_a, words_b, autojunk=False)
    out_a: list[str] = []
    out_b: list[str] = []
    for op, a1, a2, b1, b2 in sm.get_opcodes():
        if op == "equal":
            out_a.extend(html.escape(w) for w in words_a[a1:a2])
            out_b.extend(html.escape(w) for w in words_b[b1:b2])
        elif op == "replace":
            out_a.extend(
                f'<span class="diff-change">{html.escape(w)}</span>'
                for w in words_a[a1:a2]
            )
            out_b.extend(
                f'<span class="diff-change">{html.escape(w)}</span>'
                for w in words_b[b1:b2]
            )
        elif op == "delete":
            out_a.extend(
                f'<span class="diff-del">{html.escape(w)}</span>'
                for w in words_a[a1:a2]
            )
        elif op == "insert":
            out_b.extend(
                f'<span class="diff-ins">{html.escape(w)}</span>'
                for w in words_b[b1:b2]
            )
    return out_a, out_b


def diff_highlight(text_a: str, text_b: str) -> tuple[str, str]:
    """Return HTML-annotated versions of *text_a* and *text_b*.

    Diffs line-by-line, then word-by-word within each line pair so that
    the line-break structure is preserved while intra-line changes are
    highlighted with ``<span>`` tags.

    CSS classes used:
    - ``diff-del``    — word only in text_a (deletion)
    - ``diff-ins``    — word only in text_b (insertion)
    - ``diff-change`` — word differs between the two texts
    """
    lines_a = text_a.splitlines()
    lines_b = text_b.splitlines()

    # Align lines with SequenceMatcher so inserted/deleted lines are handled
    sm = difflib.SequenceMatcher(None, lines_a, lines_b, autojunk=False)
    result_a: list[str] = []
    result_b: list[str] = []

    for op, a1, a2, b1, b2 in sm.get_opcodes():
        if op == "equal":
            # Lines match — still diff words in case of subtle spacing diffs
            for la, lb in zip(lines_a[a1:a2], lines_b[b1:b2]):
                wa, wb = _diff_words(la.split(), lb.split())
                result_a.append(" ".join(wa))
                result_b.append(" ".join(wb))
        elif op == "replace":
            # Pair up changed lines; diff words within each pair
            chunk_a = lines_a[a1:a2]
            chunk_b = lines_b[b1:b2]
            pairs = max(len(chunk_a), len(chunk_b))
            for i in range(pairs):
                la = chunk_a[i] if i < len(chunk_a) else ""
                lb = chunk_b[i] if i < len(chunk_b) else ""
                wa, wb = _diff_words(la.split(), lb.split())
                if la:
                    result_a.append(" ".join(wa))
                if lb:
                    result_b.append(" ".join(wb))
        elif op == "delete":
            for la in lines_a[a1:a2]:
                words = [
                    f'<span class="diff-del">{html.escape(w)}</span>'
                    for w in la.split()
                ]
                result_a.append(" ".join(words))
        elif op == "insert":
            for lb in lines_b[b1:b2]:
                words = [
                    f'<span class="diff-ins">{html.escape(w)}</span>'
                    for w in lb.split()
                ]
                result_b.append(" ".join(words))

    return "\n".join(result_a), "\n".join(result_b)


def image_rel_path(entry: dict) -> str:
    """Compute relative path from data/evaluation/review/ to the actual image file.

    Uses the manifest's ``image_path`` (which points to the original file)
    instead of the symlink under data/subsets/, because browsers cannot
    resolve symlinks when opening file:// URLs on WSL2.
    """
    image_abs = PROJECT_ROOT / entry["image_path"]
    return os.path.relpath(image_abs, REVIEW_DIR)


def build_page_section(entry: dict, subset: str, editions: tuple[str, ...]) -> str:
    """Build one HTML <section> for a single manuscript page."""
    page_id = entry["page_id"]
    canonical_name = entry["canonical_name"]

    # Metadata header — only show non-empty fields
    meta_parts = []
    for field, label in METADATA_FIELDS:
        value = entry.get(field, "")
        if value:
            meta_parts.append(
                f'<span class="meta-item"><strong>{html.escape(label)}:</strong> '
                f'{html.escape(str(value))}</span>'
            )
    meta_html = " &middot; ".join(meta_parts)

    # Image panel — use original file path (not symlink) for WSL2 compat
    img_path = image_rel_path(entry)
    image_abs = PROJECT_ROOT / entry["image_path"]
    img_panel_class = "panel image-span" if len(editions) == 2 else "panel"
    if image_abs.exists():
        img_panel = (
            f'<div class="{img_panel_class}">\n'
            f'  <h3>Facsimile</h3>\n'
            f'  <div class="panel-content">\n'
            f'    <img src="{html.escape(img_path)}" alt="{html.escape(page_id)}" '
            f'loading="lazy">\n'
            f'  </div>\n'
            f'</div>'
        )
    else:
        img_panel = (
            f'<div class="{img_panel_class}">\n'
            f'  <h3>Facsimile</h3>\n'
            f'  <div class="panel-content">\n'
            f'    <p class="warning">Image not found: {html.escape(canonical_name)}</p>\n'
            f'  </div>\n'
            f'</div>'
        )

    # Text panels for each GT edition (source + normalized)
    gt_texts = {ed: load_gt_text(subset, ed, page_id) for ed in editions}
    raw_texts = {ed: load_raw_gt_text(entry, ed) for ed in editions}

    # When two editions exist, highlight differences between normalized texts
    if len(editions) == 2 and all(gt_texts[ed].strip() for ed in editions):
        ed_a, ed_b = editions
        highlighted_a, highlighted_b = diff_highlight(gt_texts[ed_a], gt_texts[ed_b])
        gt_html = {ed_a: highlighted_a, ed_b: highlighted_b}
    else:
        gt_html = {}

    if len(editions) == 2:
        # CODEA: 3-col × 2-row grid (image | source | normalized)
        # Reverse so critical edition appears in top row
        display_order = list(reversed(editions))

        text_panels = []
        for edition in display_order:
            edition_label = edition.capitalize()
            gt_text = gt_texts[edition]
            raw_text = raw_texts[edition]

            # Source panel
            if raw_text.strip():
                source_content = f'<pre>{html.escape(raw_text)}</pre>'
            else:
                source_content = '<p class="warning">No source available</p>'

            text_panels.append(
                f'<div class="panel">\n'
                f'  <h3>{html.escape(edition_label)} — Source</h3>\n'
                f'  <div class="panel-content">\n'
                f'    {source_content}\n'
                f'  </div>\n'
                f'</div>'
            )

            # Normalized panel
            if gt_text.strip():
                if edition in gt_html:
                    norm_content = f'<pre>{gt_html[edition]}</pre>'
                else:
                    norm_content = f'<pre>{html.escape(gt_text)}</pre>'
            else:
                norm_content = '<p class="warning">No transcription available</p>'

            text_panels.append(
                f'<div class="panel">\n'
                f'  <h3>{html.escape(edition_label)} — Normalized</h3>\n'
                f'  <div class="panel-content">\n'
                f'    {norm_content}\n'
                f'  </div>\n'
                f'</div>'
            )

        grid_cols = "1fr 1fr 1fr"
    else:
        # Toledo: stacked Source + Normalized in single panel per edition
        text_panels = []
        for edition in editions:
            gt_text = gt_texts[edition]
            raw_text = raw_texts[edition]
            edition_label = edition.capitalize()

            if raw_text.strip():
                source_content = f'<pre>{html.escape(raw_text)}</pre>'
            else:
                source_content = '<p class="warning">No source available</p>'

            if gt_text.strip():
                if edition in gt_html:
                    norm_content = f'<pre>{gt_html[edition]}</pre>'
                else:
                    norm_content = f'<pre>{html.escape(gt_text)}</pre>'
            else:
                norm_content = '<p class="warning">No transcription available</p>'

            text_panels.append(
                f'<div class="panel">\n'
                f'  <h3>{html.escape(edition_label)}</h3>\n'
                f'  <div class="panel-content">\n'
                f'    <h4 class="sub-header">Source</h4>\n'
                f'    {source_content}\n'
                f'    <h4 class="sub-header">Normalized</h4>\n'
                f'    {norm_content}\n'
                f'  </div>\n'
                f'</div>'
            )

        n_cols = 1 + len(editions)
        grid_cols = " ".join(["1fr"] * n_cols)

    return (
        f'<section id="{html.escape(page_id)}">\n'
        f'  <div class="section-header">\n'
        f'    <h2>{html.escape(page_id)}</h2>\n'
        f'    <div class="metadata">{meta_html}</div>\n'
        f'  </div>\n'
        f'  <div class="grid" style="grid-template-columns: {grid_cols};">\n'
        f'    {img_panel}\n'
        f'    {"".join(text_panels)}\n'
        f'  </div>\n'
        f'  <div class="back-to-top"><a href="#toc">Back to top</a></div>\n'
        f'</section>\n'
    )


def build_toc(entries: list[dict]) -> str:
    """Build table of contents grouped by doc_id."""
    grouped: OrderedDict[str, list[dict]] = OrderedDict()
    for entry in entries:
        doc_id = entry["doc_id"]
        grouped.setdefault(doc_id, []).append(entry)

    toc_parts = ['<nav id="toc">\n<h2>Table of Contents</h2>\n']
    for doc_id, pages in grouped.items():
        toc_parts.append(f'<div class="toc-group">\n')
        toc_parts.append(f'  <h3>{html.escape(doc_id)}</h3>\n')
        toc_parts.append(f'  <ul>\n')
        for entry in pages:
            page_id = entry["page_id"]
            page_label = entry.get("page", page_id)
            toc_parts.append(
                f'    <li><a href="#{html.escape(page_id)}">'
                f'{html.escape(page_label)}</a></li>\n'
            )
        toc_parts.append(f'  </ul>\n')
        toc_parts.append(f'</div>\n')
    toc_parts.append('</nav>\n')
    return "".join(toc_parts)


def build_html(subset: str, entries: list[dict], editions: tuple[str, ...]) -> str:
    """Build the full HTML document with inline CSS."""
    edition_labels = ", ".join(e.capitalize() for e in editions)
    title = f"{subset.upper()} Review — {edition_labels}"

    toc = build_toc(entries)
    sections = "\n".join(
        build_page_section(entry, subset, editions) for entry in entries
    )

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(title)}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f5f5f5;
    color: #333;
    line-height: 1.5;
    padding: 1rem;
  }}
  h1 {{
    text-align: center;
    padding: 1rem 0;
    color: #1a1a1a;
    border-bottom: 2px solid #ccc;
    margin-bottom: 1rem;
  }}
  /* Table of Contents */
  #toc {{
    background: #fff;
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 1.5rem;
    margin-bottom: 2rem;
  }}
  #toc h2 {{
    margin-bottom: 1rem;
    font-size: 1.3rem;
  }}
  .toc-group {{
    margin-bottom: 0.75rem;
  }}
  .toc-group h3 {{
    font-size: 1rem;
    color: #555;
    margin-bottom: 0.25rem;
  }}
  .toc-group ul {{
    list-style: none;
    display: flex;
    flex-wrap: wrap;
    gap: 0.25rem 1rem;
    padding-left: 0.5rem;
  }}
  .toc-group a {{
    color: #2563eb;
    text-decoration: none;
    font-size: 0.9rem;
  }}
  .toc-group a:hover {{
    text-decoration: underline;
  }}
  /* Page sections */
  section {{
    background: #fff;
    border: 1px solid #ddd;
    border-radius: 6px;
    margin-bottom: 2rem;
    padding: 1.5rem;
    page-break-after: always;
  }}
  .section-header {{
    margin-bottom: 1rem;
    border-bottom: 1px solid #eee;
    padding-bottom: 0.75rem;
  }}
  .section-header h2 {{
    font-size: 1.2rem;
    color: #1a1a1a;
    margin-bottom: 0.5rem;
  }}
  .metadata {{
    font-size: 0.85rem;
    color: #666;
  }}
  .meta-item {{
    white-space: nowrap;
  }}
  /* Grid layout */
  .grid {{
    display: grid;
    gap: 1rem;
  }}
  .panel {{
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    overflow: hidden;
  }}
  .panel h3 {{
    background: #f8f8f8;
    padding: 0.5rem 0.75rem;
    font-size: 0.9rem;
    color: #444;
    border-bottom: 1px solid #e0e0e0;
  }}
  .panel-content {{
    overflow: auto;
    max-height: 800px;
    padding: 0.5rem;
  }}
  .panel-content img {{
    width: 100%;
    height: auto;
    display: block;
  }}
  .image-span {{
    grid-row: span 2;
  }}
  .image-span .panel-content {{
    max-height: none;
  }}
  .panel-content pre {{
    white-space: pre-wrap;
    word-wrap: break-word;
    font-family: "Courier New", Courier, monospace;
    font-size: 0.72rem;
    line-height: 1.4;
    padding: 0.5rem;
    margin: 0;
  }}
  .sub-header {{
    font-size: 0.75rem;
    color: #666;
    background: #f0f0f0;
    padding: 0.25rem 0.5rem;
    margin: 0.5rem 0 0 0;
    border-top: 1px solid #e0e0e0;
  }}
  .sub-header:first-child {{
    margin-top: 0;
  }}
  .warning {{
    color: #b45309;
    background: #fef3c7;
    padding: 1rem;
    border-radius: 4px;
    text-align: center;
    font-style: italic;
  }}
  /* Diff highlighting between editions */
  .diff-del {{
    background: #fee2e2;
    border-radius: 2px;
    padding: 0 2px;
  }}
  .diff-ins {{
    background: #dcfce7;
    border-radius: 2px;
    padding: 0 2px;
  }}
  .diff-change {{
    background: #fef9c3;
    border-radius: 2px;
    padding: 0 2px;
  }}
  .back-to-top {{
    text-align: right;
    margin-top: 0.75rem;
    font-size: 0.85rem;
  }}
  .back-to-top a {{
    color: #2563eb;
    text-decoration: none;
  }}
  .back-to-top a:hover {{
    text-decoration: underline;
  }}
  /* Print styles */
  @media print {{
    body {{ padding: 0; background: #fff; }}
    #toc {{ page-break-after: always; }}
    section {{ border: none; box-shadow: none; }}
    .back-to-top {{ display: none; }}
    .panel-content {{ max-height: none; overflow: visible; }}
  }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<p style="text-align:center; color:#666; margin-bottom:1.5rem;">
  {len(entries)} pages &middot; Editions: {html.escape(edition_labels)}
</p>
{toc}
{sections}
</body>
</html>"""


def generate_review(subset: str) -> None:
    """Orchestrate: load manifest, build sections, write HTML."""
    entries = load_manifest(subset)
    if not entries:
        print(f"  No manifest found for {subset}, skipping.")
        return

    editions = VALID_COMBOS.get(subset)
    if not editions:
        print(f"  No valid editions for {subset}, skipping.")
        return

    print(f"  {subset}: {len(entries)} pages, editions: {', '.join(editions)}")

    html_content = build_html(subset, entries, editions)

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REVIEW_DIR / f"{subset}_review.html"
    output_path.write_text(html_content, encoding="utf-8")

    size_kb = output_path.stat().st_size / 1024
    print(f"  Written: {output_path.relative_to(PROJECT_ROOT)} ({size_kb:.0f} KB)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate HTML review pages for GT/image comparison",
    )
    parser.add_argument(
        "--subset",
        choices=["codea", "toledo", "all"],
        default="all",
        help="Which dataset subset to generate (default: all)",
    )
    args = parser.parse_args()

    if args.subset == "all":
        subsets = list(VALID_COMBOS.keys())
    else:
        subsets = [args.subset]

    print(f"Generating review pages for: {', '.join(subsets)}")
    for subset in subsets:
        generate_review(subset)
    print("Done.")


if __name__ == "__main__":
    main()
