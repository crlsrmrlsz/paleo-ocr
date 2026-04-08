#!/usr/bin/env python3
"""Generate interactive HTML for selecting optimization images from non-eval CODEA pages.

Scans all CODEA documents NOT in the evaluation benchmark, displays each page's
facsimile alongside its paleographic ground truth, and provides checkboxes for
the user to select which pages are high-quality enough for DSPy optimization.

  python scripts/review/generate_optimization_review.py

Output:
  data/evaluation/review/optimization_candidates.html

User workflow:
  1. Open the HTML in a browser
  2. Review images — check the ones with good quality
  3. Click "Export Selection" to get a JSON array of selected page IDs
  4. Save the JSON to optimized/dataset_selection.json
  5. Use as input to the optimization dataset preparation script
"""

import html
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CODEA_DIR = PROJECT_ROOT / "data" / "corpora" / "codea" / "documents"
REVIEW_DIR = PROJECT_ROOT / "data" / "evaluation" / "review"

# Import shared pipeline config for evaluation manifest doc IDs
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from pipeline_config import load_manifest  # noqa: E402

# Evaluation doc IDs to EXCLUDE (entire documents, not just pages)
EVAL_DOC_IDS = {
    entry["doc_id"]
    for entry in load_manifest("codea")
}

METADATA_FIELDS = [
    ("doc_id", "Document"),
    ("date", "Date"),
    ("siglo", "Century"),
    ("letra", "Script"),
    ("tipologia", "Typology"),
    ("copista", "Scribe"),
    ("provincia", "Province"),
]


def scan_non_eval_documents():
    """Scan CODEA documents not in evaluation set, return page entries."""
    entries = []
    for doc_dir in sorted(CODEA_DIR.iterdir()):
        if not doc_dir.is_dir():
            continue
        doc_id = doc_dir.name
        if doc_id in EVAL_DOC_IDS:
            continue

        metadata_path = doc_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        extra = metadata.get("extra", {})

        gt_dir = doc_dir / "ground_truth"
        img_dir = doc_dir / "facsimiles"
        resolutions = metadata.get("facsimile_resolution", {})

        for page_id in sorted(metadata.get("facsimile_pages", [])):
            img_path = img_dir / f"{page_id}.jpg"
            gt_path = gt_dir / f"paleographic_{page_id}.txt"

            # Image quality info
            res = resolutions.get(page_id, {})
            width = res.get("width", 0)
            height = res.get("height", 0)
            megapixels = (width * height) / 1_000_000 if width and height else 0
            file_size_kb = img_path.stat().st_size / 1024 if img_path.exists() else 0

            entries.append({
                "doc_id": doc_id,
                "page_id": page_id,
                "full_page_id": f"{doc_id}_{page_id}",
                "image_path": str(img_path.relative_to(PROJECT_ROOT)),
                "image_abs": str(img_path),
                "image_exists": img_path.exists(),
                "gt_path": str(gt_path.relative_to(PROJECT_ROOT)),
                "gt_exists": gt_path.exists(),
                "gt_text": gt_path.read_text(encoding="utf-8") if gt_path.exists() else "",
                "width": width,
                "height": height,
                "megapixels": megapixels,
                "file_size_kb": file_size_kb,
                # Metadata
                "date": metadata.get("date", ""),
                "title": metadata.get("title", ""),
                "letra": metadata.get("letra", extra.get("letra", "")),
                "siglo": extra.get("siglo", ""),
                "tipologia": extra.get("tipologia", ""),
                "copista": extra.get("copista", ""),
                "provincia": extra.get("provincia", ""),
            })

    return entries


def build_quality_badge(entry):
    """Build HTML badge showing image quality indicators."""
    w, h = entry["width"], entry["height"]
    mp = entry["megapixels"]
    kb = entry["file_size_kb"]

    if not entry["image_exists"]:
        return '<span class="badge badge-error">Missing image</span>'
    if not entry["gt_exists"]:
        return '<span class="badge badge-error">Missing GT</span>'

    parts = []
    parts.append(f'{w}&times;{h}')
    parts.append(f'{mp:.1f} MP')
    parts.append(f'{kb:.0f} KB')

    badge_class = "badge-ok"
    if mp < 1.0:
        badge_class = "badge-warn"
    elif mp < 0.5:
        badge_class = "badge-error"

    return f'<span class="badge {badge_class}">{" &middot; ".join(parts)}</span>'


def build_page_section(entry):
    """Build HTML section for a single manuscript page with checkbox."""
    page_id = entry["full_page_id"]
    safe_id = html.escape(page_id)

    # Metadata
    meta_parts = []
    for field, label in METADATA_FIELDS:
        value = entry.get(field, "")
        if value:
            meta_parts.append(
                f'<span class="meta-item"><strong>{html.escape(label)}:</strong> '
                f'{html.escape(str(value))}</span>'
            )
    meta_html = " &middot; ".join(meta_parts)

    # Quality badge
    quality_badge = build_quality_badge(entry)

    # Image panel
    img_rel = os.path.relpath(entry["image_abs"], REVIEW_DIR)
    if entry["image_exists"]:
        img_panel = (
            f'<div class="panel image-span">\n'
            f'  <h3>Facsimile</h3>\n'
            f'  <div class="panel-content">\n'
            f'    <img src="{html.escape(img_rel)}" alt="{safe_id}" loading="lazy">\n'
            f'  </div>\n'
            f'</div>'
        )
    else:
        img_panel = (
            f'<div class="panel image-span">\n'
            f'  <h3>Facsimile</h3>\n'
            f'  <div class="panel-content">\n'
            f'    <p class="warning">Image not found</p>\n'
            f'  </div>\n'
            f'</div>'
        )

    # GT panel
    gt_text = entry["gt_text"]
    if gt_text.strip():
        gt_content = f'<pre>{html.escape(gt_text)}</pre>'
    else:
        gt_content = '<p class="warning">No paleographic transcription available</p>'

    gt_panel = (
        f'<div class="panel">\n'
        f'  <h3>Paleographic GT</h3>\n'
        f'  <div class="panel-content">\n'
        f'    {gt_content}\n'
        f'  </div>\n'
        f'</div>'
    )

    # Checkbox disabled if missing image or GT
    disabled = ""
    if not entry["image_exists"] or not entry["gt_exists"] or not gt_text.strip():
        disabled = "disabled"

    return (
        f'<section id="{safe_id}" class="page-section" data-doc="{html.escape(entry["doc_id"])}">\n'
        f'  <div class="section-header">\n'
        f'    <label class="checkbox-label">\n'
        f'      <input type="checkbox" class="page-checkbox" '
        f'value="{safe_id}" data-doc="{html.escape(entry["doc_id"])}" {disabled}>\n'
        f'      <h2>{safe_id}</h2>\n'
        f'    </label>\n'
        f'    <div class="header-right">\n'
        f'      {quality_badge}\n'
        f'    </div>\n'
        f'  </div>\n'
        f'  <div class="metadata">{meta_html}</div>\n'
        f'  <div class="grid" style="grid-template-columns: 1fr 1fr;">\n'
        f'    {img_panel}\n'
        f'    {gt_panel}\n'
        f'  </div>\n'
        f'  <div class="back-to-top"><a href="#toc">Back to top</a></div>\n'
        f'</section>\n'
    )


def build_toc(entries):
    """Build table of contents grouped by doc_id with per-doc select all."""
    grouped = OrderedDict()
    for entry in entries:
        grouped.setdefault(entry["doc_id"], []).append(entry)

    toc_parts = ['<nav id="toc">\n<h2>Table of Contents</h2>\n']
    for doc_id, pages in grouped.items():
        n_valid = sum(
            1 for p in pages
            if p["image_exists"] and p["gt_exists"] and p["gt_text"].strip()
        )
        safe_doc = html.escape(doc_id)
        toc_parts.append(f'<div class="toc-group">\n')
        toc_parts.append(
            f'  <h3>'
            f'<label class="doc-select-label">'
            f'<input type="checkbox" class="doc-checkbox" data-doc="{safe_doc}" '
            f'title="Select/deselect all pages in {safe_doc}"> '
            f'{safe_doc}</label>'
            f' <span class="toc-count">({n_valid} pages)</span></h3>\n'
        )
        toc_parts.append(f'  <ul>\n')
        for entry in pages:
            page_id = entry["full_page_id"]
            page_label = entry["page_id"]
            mp = entry["megapixels"]
            warn = ""
            if not entry["image_exists"] or not entry["gt_exists"]:
                warn = ' class="toc-warn"'
            elif mp < 1.0:
                warn = ' class="toc-low-res"'
            toc_parts.append(
                f'    <li{warn}><a href="#{html.escape(page_id)}">'
                f'{html.escape(page_label)}</a></li>\n'
            )
        toc_parts.append(f'  </ul>\n')
        toc_parts.append(f'</div>\n')
    toc_parts.append('</nav>\n')
    return "".join(toc_parts)


def build_html(entries):
    """Build the full HTML document with inline CSS and JavaScript."""
    total = len(entries)
    valid = sum(
        1 for e in entries
        if e["image_exists"] and e["gt_exists"] and e["gt_text"].strip()
    )
    n_docs = len(set(e["doc_id"] for e in entries))

    toc = build_toc(entries)
    sections = "\n".join(build_page_section(e) for e in entries)

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DSPy Optimization — Image Selection</title>
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
    margin-bottom: 0.5rem;
  }}
  .subtitle {{
    text-align: center;
    color: #666;
    margin-bottom: 1rem;
    font-size: 0.95rem;
  }}
  /* Toolbar */
  .toolbar {{
    position: sticky;
    top: 0;
    z-index: 100;
    background: #1e293b;
    color: #f8fafc;
    padding: 0.75rem 1.5rem;
    border-radius: 6px;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
  }}
  .toolbar-left {{
    display: flex;
    align-items: center;
    gap: 1rem;
  }}
  .counter {{
    font-size: 1.1rem;
    font-weight: 600;
  }}
  .counter .num {{
    color: #60a5fa;
    font-size: 1.3rem;
  }}
  .toolbar button {{
    padding: 0.5rem 1rem;
    border: 1px solid #475569;
    border-radius: 4px;
    background: #334155;
    color: #f8fafc;
    cursor: pointer;
    font-size: 0.85rem;
    transition: background 0.15s;
  }}
  .toolbar button:hover {{
    background: #475569;
  }}
  .toolbar button.primary {{
    background: #2563eb;
    border-color: #3b82f6;
  }}
  .toolbar button.primary:hover {{
    background: #1d4ed8;
  }}
  .export-feedback {{
    font-size: 0.85rem;
    color: #86efac;
    display: none;
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
  .toc-count {{
    font-weight: normal;
    color: #888;
    font-size: 0.85rem;
  }}
  .doc-select-label {{
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
  }}
  .doc-select-label input {{
    cursor: pointer;
  }}
  .toc-group ul {{
    list-style: none;
    display: flex;
    flex-wrap: wrap;
    gap: 0.25rem 1rem;
    padding-left: 1.75rem;
  }}
  .toc-group a {{
    color: #2563eb;
    text-decoration: none;
    font-size: 0.9rem;
  }}
  .toc-group a:hover {{
    text-decoration: underline;
  }}
  .toc-warn a {{
    color: #dc2626;
    text-decoration: line-through;
  }}
  .toc-low-res a {{
    color: #d97706;
  }}
  /* Page sections */
  section {{
    background: #fff;
    border: 1px solid #ddd;
    border-radius: 6px;
    margin-bottom: 2rem;
    padding: 1.5rem;
  }}
  section.selected {{
    border-color: #3b82f6;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3);
  }}
  .section-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.5rem;
    border-bottom: 1px solid #eee;
    padding-bottom: 0.75rem;
  }}
  .checkbox-label {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    cursor: pointer;
  }}
  .checkbox-label input {{
    width: 18px;
    height: 18px;
    cursor: pointer;
  }}
  .checkbox-label h2 {{
    font-size: 1.2rem;
    color: #1a1a1a;
    margin: 0;
  }}
  .header-right {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }}
  .metadata {{
    font-size: 0.85rem;
    color: #666;
    margin-bottom: 0.75rem;
  }}
  .meta-item {{
    white-space: nowrap;
  }}
  /* Badges */
  .badge {{
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 500;
    white-space: nowrap;
  }}
  .badge-ok {{
    background: #dcfce7;
    color: #166534;
  }}
  .badge-warn {{
    background: #fef3c7;
    color: #92400e;
  }}
  .badge-error {{
    background: #fee2e2;
    color: #991b1b;
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
  .warning {{
    color: #b45309;
    background: #fef3c7;
    padding: 1rem;
    border-radius: 4px;
    text-align: center;
    font-style: italic;
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
    .toolbar {{ display: none; }}
    body {{ padding: 0; background: #fff; }}
    #toc {{ page-break-after: always; }}
    section {{ border: none; box-shadow: none; page-break-after: always; }}
    .back-to-top {{ display: none; }}
    .panel-content {{ max-height: none; overflow: visible; }}
  }}
</style>
</head>
<body>
<h1>DSPy Optimization &mdash; Image Selection</h1>
<p class="subtitle">
  {n_docs} documents &middot; {total} pages total &middot; {valid} with valid image+GT
  &middot; Select high-quality pages for DSPy prompt optimization
</p>

<div class="toolbar">
  <div class="toolbar-left">
    <span class="counter"><span class="num" id="selected-count">0</span> / {valid} selected</span>
    <button onclick="selectAll()">Select All Valid</button>
    <button onclick="deselectAll()">Deselect All</button>
    <button onclick="selectHighRes()">Select &ge;2 MP</button>
  </div>
  <div>
    <button class="primary" onclick="exportSelection()">Export Selection (JSON)</button>
    <span class="export-feedback" id="export-feedback">Copied!</span>
  </div>
</div>

{toc}
{sections}

<script>
// --- Selection logic ---
function updateCounter() {{
  const checked = document.querySelectorAll('.page-checkbox:checked').length;
  document.getElementById('selected-count').textContent = checked;

  // Update section highlighting
  document.querySelectorAll('.page-checkbox').forEach(cb => {{
    const section = cb.closest('section');
    if (section) {{
      section.classList.toggle('selected', cb.checked);
    }}
  }});

  // Update doc-level checkboxes
  document.querySelectorAll('.doc-checkbox').forEach(docCb => {{
    const doc = docCb.dataset.doc;
    const pageCbs = document.querySelectorAll(`.page-checkbox[data-doc="${{doc}}"]:not(:disabled)`);
    const checkedCbs = document.querySelectorAll(`.page-checkbox[data-doc="${{doc}}"]:checked`);
    docCb.checked = pageCbs.length > 0 && pageCbs.length === checkedCbs.length;
    docCb.indeterminate = checkedCbs.length > 0 && checkedCbs.length < pageCbs.length;
  }});
}}

function selectAll() {{
  document.querySelectorAll('.page-checkbox:not(:disabled)').forEach(cb => cb.checked = true);
  updateCounter();
}}

function deselectAll() {{
  document.querySelectorAll('.page-checkbox').forEach(cb => cb.checked = false);
  updateCounter();
}}

function selectHighRes() {{
  deselectAll();
  document.querySelectorAll('.page-checkbox:not(:disabled)').forEach(cb => {{
    // Check badge content for MP value
    const section = cb.closest('section');
    const badge = section ? section.querySelector('.badge') : null;
    if (badge) {{
      const match = badge.textContent.match(/([0-9.]+) *MP/);
      if (match && parseFloat(match[1]) >= 2.0) {{
        cb.checked = true;
      }}
    }}
  }});
  updateCounter();
}}

// Document-level select all
document.querySelectorAll('.doc-checkbox').forEach(docCb => {{
  docCb.addEventListener('change', () => {{
    const doc = docCb.dataset.doc;
    document.querySelectorAll(`.page-checkbox[data-doc="${{doc}}"]:not(:disabled)`).forEach(cb => {{
      cb.checked = docCb.checked;
    }});
    updateCounter();
  }});
}});

// Individual checkbox change
document.querySelectorAll('.page-checkbox').forEach(cb => {{
  cb.addEventListener('change', updateCounter);
}});

function exportSelection() {{
  const selected = Array.from(document.querySelectorAll('.page-checkbox:checked'))
    .map(cb => cb.value);

  const data = JSON.stringify(selected, null, 2);

  // Try clipboard first
  if (navigator.clipboard) {{
    navigator.clipboard.writeText(data).then(() => {{
      showFeedback('Copied to clipboard!');
    }}).catch(() => {{
      downloadJson(data);
    }});
  }} else {{
    downloadJson(data);
  }}
}}

function downloadJson(data) {{
  const blob = new Blob([data], {{ type: 'application/json' }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'dataset_selection.json';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  showFeedback('Downloaded!');
}}

function showFeedback(msg) {{
  const el = document.getElementById('export-feedback');
  el.textContent = msg;
  el.style.display = 'inline';
  setTimeout(() => {{ el.style.display = 'none'; }}, 2000);
}}

// Initialize counter
updateCounter();
</script>
</body>
</html>"""


def main():
    print("Scanning non-evaluation CODEA documents...")
    entries = scan_non_eval_documents()

    n_docs = len(set(e["doc_id"] for e in entries))
    n_valid = sum(
        1 for e in entries
        if e["image_exists"] and e["gt_exists"] and e["gt_text"].strip()
    )
    print(f"  Found {len(entries)} pages from {n_docs} documents")
    print(f"  {n_valid} pages have both image and paleographic GT")
    print(f"  Excluded {len(EVAL_DOC_IDS)} evaluation documents")

    html_content = build_html(entries)

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REVIEW_DIR / "optimization_candidates.html"
    output_path.write_text(html_content, encoding="utf-8")

    size_kb = output_path.stat().st_size / 1024
    print(f"  Written: {output_path.relative_to(PROJECT_ROOT)} ({size_kb:.0f} KB)")
    print("Done. Open the HTML file in a browser to select optimization images.")


if __name__ == "__main__":
    main()
