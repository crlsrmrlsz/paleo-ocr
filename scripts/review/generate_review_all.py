#!/usr/bin/env python3
"""Generate interactive HTML review of ALL evaluated pages.

Shows a sortable table with T1-T4 Romein metrics per page, ordered by CER.
Click any row to expand the character-level error comparison (image + GT + model).
Images load via relative paths (open locally after cloning).

Usage:
    python scripts/review/generate_review_all.py
    python scripts/review/generate_review_all.py --model pipeline/11_yolo_crop_anti_halluc
"""

import argparse
import json
import re
import sys
from pathlib import Path

from rapidfuzz.distance import Levenshtein

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SUBSETS_DIR = PROJECT_ROOT / "data" / "subsets"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
REPORTS_DIR = PROJECT_ROOT / "data" / "evaluation" / "reports"

sys.path.insert(0, str(PROJECT_ROOT))
from scripts.pipeline_config import EXCLUDED_PAGES

# ---------------------------------------------------------------------------
# GT marker stripping — same logic as compute_metrics.py content mode
# ---------------------------------------------------------------------------

_MARKERS_WITH_TEXT = [
    "tachado", "raspado", "sobre raspado", "sobrescrito",
    "interlineado", "firma", "margen", r"mano \d+",
    "encabezamiento", r"título", r"lat\.", "nota",
]
_MARKERS_EMPTY = [
    "rúbrica", "cruz", "signo", "sello", "crismón",
    "blanco", "roto", "doblez", "mancha", "ilegible", "quirógrafo",
]
_KEEP_TEXT = {"interlineado", "sobrescrito", "encabezamiento", "firma", "mano"}


def strip_markers_content(text: str) -> str:
    for name in _MARKERS_WITH_TEXT:
        if any(k in name for k in _KEEP_TEXT):
            text = re.sub(rf"\[{name}\s*:\s*(.*?)\]", r"\1", text, flags=re.IGNORECASE | re.DOTALL)
        else:
            text = re.sub(rf"\[{name}\s*:\s*.*?\]", "", text, flags=re.IGNORECASE | re.DOTALL)
    for name in _MARKERS_EMPTY:
        if name == "blanco":
            text = re.sub(r"\[blanco\]", " ", text, flags=re.IGNORECASE)
        else:
            text = re.sub(rf"\[{name}\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[([a-záéíóúñç]{1,5})\]", r"\1", text)
    return text


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def compute_alignment(gt: str, hyp: str):
    opcodes = Levenshtein.opcodes(gt, hyp)
    segments = []
    for op, gs, ge, hs, he in opcodes:
        segments.append((op, gt[gs:ge], hyp[hs:he]))
    return segments


def compute_stats(segments):
    correct = subs = dels = ins = 0
    for op, g_span, h_span in segments:
        if op == "equal":
            correct += len(g_span)
        elif op == "replace":
            gl, hl = len(g_span), len(h_span)
            subs += min(gl, hl)
            if gl > hl:
                dels += gl - hl
            else:
                ins += hl - gl
        elif op == "delete":
            dels += len(g_span)
        elif op == "insert":
            ins += len(h_span)
    gt_len = correct + subs + dels
    cer = (subs + dels + ins) / gt_len if gt_len > 0 else 0
    return {"correct": correct, "substitutions": subs, "deletions": dels,
            "insertions": ins, "gt_chars": gt_len, "cer": cer}


def _esc(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def render_spans(segments, side="gt"):
    parts = []
    for op, g_span, h_span in segments:
        if op == "equal":
            text = g_span if side == "gt" else h_span
            parts.append(f'<span class="ok">{_esc(text)}</span>')
        elif op == "replace":
            if side == "gt":
                parts.append(f'<span class="sub" title="Model: {_esc(h_span)}">{_esc(g_span)}</span>')
            else:
                parts.append(f'<span class="sub" title="GT: {_esc(g_span)}">{_esc(h_span)}</span>')
        elif op == "delete" and side == "gt":
            parts.append(f'<span class="del" title="Missing in model">{_esc(g_span)}</span>')
        elif op == "insert" and side == "hyp":
            parts.append(f'<span class="ins" title="Not in GT">{_esc(h_span)}</span>')
    return "".join(parts)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_pages(model: str):
    """Load all evaluated pages with GT, hyp, metrics, and alignment."""
    pages = []

    # Load Romein metrics
    romein = {}
    for metrics_file in REPORTS_DIR.glob("*_metrics.json"):
        data = json.loads(metrics_file.read_text())
        subset = data["subset"]
        edition = data["edition"]
        for r in data["results"]:
            if r["model"] == model:
                key = (r["page_id"], subset, edition)
                romein[key] = r.get("romein", {})

    # Process each subset
    for subset in ["codea", "toledo"]:
        manifest_path = SUBSETS_DIR / subset / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())

        edition = "paleographic" if subset == "codea" else "editorial"

        for entry in manifest:
            page_id = entry["page_id"]
            if page_id in EXCLUDED_PAGES:
                continue

            gt_path = SUBSETS_DIR / subset / "ground_truth" / edition / f"{page_id}.txt"
            hyp_path = RESULTS_DIR / model / "normalized" / f"{page_id}.txt"

            if not gt_path.exists() or not hyp_path.exists():
                continue

            raw_gt = gt_path.read_text(encoding="utf-8").strip()
            stripped_gt = strip_markers_content(raw_gt)
            gt_norm = re.sub(r"\s+", " ", stripped_gt).strip()
            hyp_norm = re.sub(r"\s+", " ", hyp_path.read_text(encoding="utf-8").strip()).strip()

            segments = compute_alignment(gt_norm, hyp_norm)
            stats = compute_stats(segments)

            # Relative image path from HTML location
            img_rel = f"../../subsets/{subset}/images/{entry['canonical_name']}"

            rm = romein.get((page_id, subset, edition), {})

            pages.append({
                "page_id": page_id,
                "subset": subset,
                "edition": edition,
                "doc_id": entry.get("doc_id", ""),
                "date": entry.get("date", ""),
                "siglo": entry.get("siglo", ""),
                "letra": entry.get("letra_normalized", ""),
                "img_rel": img_rel,
                "stats": stats,
                "t1": rm.get("T1_raw", stats["cer"]),
                "t2": rm.get("T2_nospace", 0),
                "t3": rm.get("T3_lowercase", 0),
                "t4": rm.get("T4_alnum", 0),
                "gt_html": render_spans(segments, "gt"),
                "hyp_html": render_spans(segments, "hyp"),
            })

    pages.sort(key=lambda p: p["t1"])
    return pages


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def generate_html(pages, model):
    # Build table rows
    table_rows = []
    for i, p in enumerate(pages):
        s = p["stats"]
        table_rows.append(
            f'<tr class="page-row" data-idx="{i}" data-subset="{p["subset"]}">'
            f'<td class="rank">{i+1}</td>'
            f'<td class="pid">{p["page_id"]}</td>'
            f'<td>{p["subset"]}</td>'
            f'<td>{p["doc_id"]}</td>'
            f'<td>{p["date"]}</td>'
            f'<td>{p["letra"]}</td>'
            f'<td class="num t1">{p["t1"]:.1%}</td>'
            f'<td class="num">{p["t2"]:.1%}</td>'
            f'<td class="num">{p["t3"]:.1%}</td>'
            f'<td class="num t4">{p["t4"]:.1%}</td>'
            f'<td class="num">{s["substitutions"]}</td>'
            f'<td class="num">{s["deletions"]}</td>'
            f'<td class="num">{s["insertions"]}</td>'
            f'<td class="num">{s["gt_chars"]}</td>'
            f'</tr>'
        )

    # Build detail panels (hidden, shown on click)
    detail_panels = []
    for i, p in enumerate(pages):
        s = p["stats"]
        detail_panels.append(
            f'<tr class="detail-row" id="detail-{i}" style="display:none">'
            f'<td colspan="14">'
            f'<div class="detail-panel">'
            f'<div class="detail-header">'
            f'<span class="detail-title">{p["page_id"]}</span>'
            f'<span class="detail-meta">CER: {s["cer"]:.1%} = ({s["substitutions"]}S+{s["deletions"]}D+{s["insertions"]}I)/{s["gt_chars"]}</span>'
            f'</div>'
            f'<div class="detail-cols">'
            f'<div class="detail-img"><div class="img-label">Manuscript</div><img src="{p["img_rel"]}" loading="lazy"></div>'
            f'<div class="detail-text"><div class="text-label">Ground Truth ({s["gt_chars"]} chars)</div><div class="text-content">{p["gt_html"]}</div></div>'
            f'<div class="detail-text"><div class="text-label">Model Output (errors highlighted)</div><div class="text-content">{p["hyp_html"]}</div></div>'
            f'</div>'
            f'</div>'
            f'</td>'
            f'</tr>'
        )

    n_codea = sum(1 for p in pages if p["subset"] == "codea")
    n_toledo = sum(1 for p in pages if p["subset"] == "toledo")
    mean_t1 = sum(p["t1"] for p in pages) / len(pages) if pages else 0
    mean_t4 = sum(p["t4"] for p in pages) / len(pages) if pages else 0

    rows_html = "\n".join(r + d for r, d in zip(table_rows, detail_panels))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>HTR Error Review — All Pages</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #1a1a2e; color: #eee; }}

header {{ padding: 20px 30px; background: #16213e; border-bottom: 1px solid #333; }}
header h1 {{ font-size: 1.3em; font-weight: 600; }}
header p {{ color: #999; font-size: 0.85em; margin-top: 4px; }}

.summary {{ display: flex; gap: 24px; padding: 12px 30px; background: #0f3460;
            font-size: 0.85em; flex-wrap: wrap; border-bottom: 1px solid #333; }}
.summary span {{ color: #aaa; }}
.summary strong {{ color: #e94560; }}

.controls {{ padding: 10px 30px; background: #16213e; display: flex; gap: 16px;
             align-items: center; border-bottom: 1px solid #333; flex-wrap: wrap; }}
.controls label {{ font-size: 0.85em; color: #aaa; cursor: pointer; }}
.controls input[type=checkbox] {{ margin-right: 4px; }}
.controls select {{ background: #0f3460; color: #eee; border: 1px solid #555;
                    padding: 4px 8px; border-radius: 3px; font-size: 0.85em; }}

.legend {{ display: flex; gap: 16px; padding: 8px 30px; background: #0f3460;
           font-size: 0.78em; flex-wrap: wrap; border-bottom: 1px solid #333; }}
.legend-item {{ display: flex; align-items: center; gap: 4px; }}
.legend-swatch {{ width: 12px; height: 12px; border-radius: 2px; }}

table {{ width: 100%; border-collapse: collapse; font-size: 0.82em; }}
thead {{ position: sticky; top: 0; z-index: 20; }}
th {{ background: #16213e; padding: 10px 8px; text-align: left; border-bottom: 2px solid #e94560;
     cursor: pointer; user-select: none; white-space: nowrap; }}
th:hover {{ background: #1a2744; }}
th.sorted-asc::after {{ content: ' ▲'; color: #e94560; }}
th.sorted-desc::after {{ content: ' ▼'; color: #e94560; }}

.page-row {{ cursor: pointer; transition: background 0.15s; }}
.page-row:hover {{ background: #16213e; }}
.page-row.active {{ background: #0f3460; }}
.page-row td {{ padding: 7px 8px; border-bottom: 1px solid #2a2a4a; }}
.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.t1 {{ color: #e94560; font-weight: 600; }}
.t4 {{ color: #27ae60; }}
.rank {{ color: #666; text-align: center; }}
.pid {{ font-family: monospace; font-size: 0.95em; }}

tr[data-subset="codea"] .pid {{ color: #f5d76e; }}
tr[data-subset="toledo"] .pid {{ color: #7ec8e3; }}

.detail-row td {{ padding: 0; border-bottom: 2px solid #e94560; }}
.detail-panel {{ background: #111; }}
.detail-header {{ padding: 12px 16px; background: #0a0a1a; display: flex;
                   justify-content: space-between; align-items: center; }}
.detail-title {{ font-weight: 600; font-size: 1.1em; }}
.detail-meta {{ color: #e94560; font-family: monospace; }}

.detail-cols {{ display: grid; grid-template-columns: 1fr 1fr 1fr; min-height: 400px; max-height: 70vh; }}
.detail-img {{ overflow-y: auto; background: #000; }}
.detail-img img {{ width: 100%; display: block; }}
.img-label, .text-label {{ position: sticky; top: 0; background: #16213e; padding: 6px 12px;
                            font-size: 0.8em; font-weight: 600; border-bottom: 1px solid #333; z-index: 5; }}
.detail-text {{ overflow-y: auto; border-left: 1px solid #333; }}
.text-content {{ padding: 12px; font-family: 'Courier New', monospace; font-size: 12px;
                  line-height: 1.9; white-space: pre-wrap; word-wrap: break-word; }}

.ok {{ color: #888; }}
.sub {{ background: rgba(243,156,18,0.25); color: #f5d76e; border-bottom: 2px solid #f39c12; cursor: help; }}
.del {{ background: rgba(231,76,60,0.2); color: #e88; border-bottom: 2px solid #e74c3c; cursor: help; }}
.ins {{ background: rgba(52,152,219,0.2); color: #7ec8e3; border-bottom: 2px solid #3498db; cursor: help; }}

body.dim-ok .ok {{ opacity: 0.15; }}

footer {{ padding: 12px 30px; background: #16213e; border-top: 1px solid #333;
          font-size: 0.75em; color: #666; }}
</style>
</head>
<body>

<header>
  <h1>HTR Error Review — Character-Level Analysis</h1>
  <p>Model: <strong>{model}</strong>. Click any row to expand the error comparison. Romein tiers progressively strip editorial conventions (T1→T4).</p>
</header>

<div class="summary">
  <span>Pages: <strong>{len(pages)}</strong> ({n_codea} CODEA + {n_toledo} Toledo)</span>
  <span>Mean T1 (CER): <strong>{mean_t1:.1%}</strong></span>
  <span>Mean T4 (pure reading): <strong>{mean_t4:.1%}</strong></span>
  <span>Convention cost (T1−T4): <strong>{mean_t1 - mean_t4:.1%}</strong></span>
</div>

<div class="legend">
  <div class="legend-item"><div class="legend-swatch" style="background:#888"></div> Correct</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#f39c12"></div> Substitution</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#e74c3c"></div> Deletion (in GT, missing in model)</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#3498db"></div> Insertion (in model, not in GT)</div>
</div>

<div class="controls">
  <label><input type="checkbox" id="dimOk"> Dim correct characters</label>
  <label>Filter:
    <select id="filterSubset">
      <option value="all">All pages</option>
      <option value="codea">CODEA only</option>
      <option value="toledo">Toledo only</option>
    </select>
  </label>
</div>

<table>
<thead>
<tr>
  <th data-col="rank" data-type="num">#</th>
  <th data-col="pid" data-type="str">Page ID</th>
  <th data-col="subset" data-type="str">Corpus</th>
  <th data-col="doc" data-type="str">Document</th>
  <th data-col="date" data-type="str">Date</th>
  <th data-col="letra" data-type="str">Script</th>
  <th data-col="t1" data-type="num">T1 raw</th>
  <th data-col="t2" data-type="num">T2 no-sp</th>
  <th data-col="t3" data-type="num">T3 lower</th>
  <th data-col="t4" data-type="num">T4 alnum</th>
  <th data-col="s" data-type="num">S</th>
  <th data-col="d" data-type="num">D</th>
  <th data-col="i" data-type="num">I</th>
  <th data-col="gt" data-type="num">GT chars</th>
</tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>

<footer>
  Romein tiers: T1=raw CER, T2=remove spaces, T3=+lowercase, T4=+strip special chars (pure reading).
  Character alignment uses Levenshtein opcodes, identical to evaluation pipeline.
  Generated by <a href="https://github.com/crlsrmrlsz/paleo-ocr" style="color:#e94560">paleo-ocr</a>.
</footer>

<script>
// Toggle dim correct
document.getElementById('dimOk').addEventListener('change', function() {{
  document.body.classList.toggle('dim-ok', this.checked);
}});

// Filter by subset
document.getElementById('filterSubset').addEventListener('change', function() {{
  const v = this.value;
  document.querySelectorAll('.page-row').forEach(tr => {{
    const show = v === 'all' || tr.dataset.subset === v;
    tr.style.display = show ? '' : 'none';
    const detail = document.getElementById('detail-' + tr.dataset.idx);
    if (!show && detail) detail.style.display = 'none';
  }});
}});

// Click to expand/collapse detail
document.querySelectorAll('.page-row').forEach(tr => {{
  tr.addEventListener('click', function() {{
    const idx = this.dataset.idx;
    const detail = document.getElementById('detail-' + idx);
    const wasOpen = detail.style.display === '';
    // Close all
    document.querySelectorAll('.detail-row').forEach(d => d.style.display = 'none');
    document.querySelectorAll('.page-row').forEach(r => r.classList.remove('active'));
    // Toggle
    if (!wasOpen) {{
      detail.style.display = '';
      this.classList.add('active');
      detail.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
    }}
  }});
}});

// Sortable columns
document.querySelectorAll('th[data-col]').forEach(th => {{
  th.addEventListener('click', function() {{
    const tbody = document.querySelector('tbody');
    const col = this.cellIndex;
    const isNum = this.dataset.type === 'num';
    const wasAsc = this.classList.contains('sorted-asc');
    document.querySelectorAll('th').forEach(h => h.classList.remove('sorted-asc', 'sorted-desc'));
    const dir = wasAsc ? 'desc' : 'asc';
    this.classList.add('sorted-' + dir);

    // Collect page-row + detail-row pairs
    const pairs = [];
    const rows = Array.from(tbody.querySelectorAll('.page-row'));
    rows.forEach(r => {{
      const detail = document.getElementById('detail-' + r.dataset.idx);
      const val = r.cells[col].textContent.replace('%', '');
      pairs.push({{ row: r, detail: detail, val: isNum ? parseFloat(val) || 0 : val }});
    }});

    pairs.sort((a, b) => {{
      let cmp = isNum ? a.val - b.val : a.val.localeCompare(b.val);
      return dir === 'desc' ? -cmp : cmp;
    }});

    pairs.forEach(p => {{
      tbody.appendChild(p.row);
      tbody.appendChild(p.detail);
    }});
  }});
}});
</script>
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive error review HTML")
    parser.add_argument("--model", default="pipeline/11_yolo_crop_anti_halluc", help="Model name")
    parser.add_argument("--output", default=None, help="Output HTML path")
    args = parser.parse_args()

    print(f"Loading all pages for {args.model}...")
    pages = load_all_pages(args.model)
    print(f"Loaded {len(pages)} pages")

    html = generate_html(pages, args.model)

    output_path = Path(args.output) if args.output else PROJECT_ROOT / "data" / "evaluation" / "review" / "error_comparison.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Saved to {output_path} ({output_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
