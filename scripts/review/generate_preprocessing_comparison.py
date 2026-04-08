#!/usr/bin/env python3
"""Generate HTML comparing image preprocessing variants against the baseline pipeline.

Shows side-by-side character-level diff for baseline (v11) vs each preprocessing
variant (CLAHE, sharpen, denoise), with CER deltas and tabbed navigation.

Usage:
    python scripts/review/generate_preprocessing_comparison.py
"""

import json
import re
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image as PILImage
from rapidfuzz.distance import Levenshtein

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SUBSETS_DIR = PROJECT_ROOT / "data" / "subsets"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from pipeline_config import EXCLUDED_PAGES

# ---------------------------------------------------------------------------
# Models to compare
# ---------------------------------------------------------------------------

BASELINE = "pipeline/11_yolo_crop_anti_halluc"
VARIANTS = {
    "pipeline/14_clahe": {"label": "CLAHE", "color": "#9b59b6"},
    "pipeline/15_sharpen": {"label": "Sharpen", "color": "#e67e22"},
    "pipeline/16_denoise": {"label": "Denoise", "color": "#1abc9c"},
}

IMG_MAX_WIDTH = 700

# ---------------------------------------------------------------------------
# Marker stripping — same as compute_metrics.py content mode
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


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Alignment & rendering
# ---------------------------------------------------------------------------

def compute_stats(gt, hyp):
    opcodes = Levenshtein.opcodes(gt, hyp)
    correct = subs = dels = ins = 0
    for op, gs, ge, hs, he in opcodes:
        if op == "equal":
            correct += ge - gs
        elif op == "replace":
            gl, hl = ge - gs, he - hs
            subs += min(gl, hl)
            if gl > hl:
                dels += gl - hl
            else:
                ins += hl - gl
        elif op == "delete":
            dels += ge - gs
        elif op == "insert":
            ins += he - hs
    gt_len = correct + subs + dels
    cer = (subs + dels + ins) / gt_len if gt_len > 0 else 0
    return opcodes, {"correct": correct, "substitutions": subs, "deletions": dels,
                     "insertions": ins, "gt_chars": gt_len, "cer": cer}


def _esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def render_side(gt, hyp, opcodes, side="gt"):
    parts = []
    for op, gs, ge, hs, he in opcodes:
        if op == "equal":
            t = gt[gs:ge] if side == "gt" else hyp[hs:he]
            parts.append(f'<span class="correct">{_esc(t)}</span>')
        elif op == "replace":
            if side == "gt":
                parts.append(f'<span class="substitution" title="Model: {_esc(hyp[hs:he])}">{_esc(gt[gs:ge])}</span>')
            else:
                parts.append(f'<span class="substitution" title="GT: {_esc(gt[gs:ge])}">{_esc(hyp[hs:he])}</span>')
        elif op == "delete" and side == "gt":
            parts.append(f'<span class="deletion" title="Missing in model">{_esc(gt[gs:ge])}</span>')
        elif op == "insert" and side == "hyp":
            parts.append(f'<span class="insertion" title="Not in GT">{_esc(hyp[hs:he])}</span>')
    return "".join(parts)


def resize_image_b64(image_path):
    try:
        img = PILImage.open(str(image_path))
        if img.width > IMG_MAX_WIDTH:
            ratio = IMG_MAX_WIDTH / img.width
            img = img.resize((IMG_MAX_WIDTH, int(img.height * ratio)), PILImage.LANCZOS)
        buf = BytesIO()
        if img.mode == "RGBA":
            img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=70)
        img.close()
        import base64
        return base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Per-page section
# ---------------------------------------------------------------------------

def render_page(page_id, idx, image_path, gt_raw, baseline_text, variant_texts, metadata, excluded=False):
    gt_n = normalize_ws(strip_markers_content(gt_raw))
    base_n = normalize_ws(baseline_text)

    # Baseline stats
    base_ops, base_stats = compute_stats(gt_n, base_n)
    base_gt_html = render_side(gt_n, base_n, base_ops, "gt")
    base_hyp_html = render_side(gt_n, base_n, base_ops, "hyp")

    # Variant stats
    var_data = {}
    for model_key, info in VARIANTS.items():
        vtxt = variant_texts.get(model_key, "")
        v_n = normalize_ws(vtxt)
        v_ops, v_stats = compute_stats(gt_n, v_n)
        var_data[model_key] = {
            "label": info["label"],
            "color": info["color"],
            "stats": v_stats,
            "gt_html": render_side(gt_n, v_n, v_ops, "gt"),
            "hyp_html": render_side(gt_n, v_n, v_ops, "hyp"),
        }

    # Image
    img_b64 = resize_image_b64(image_path)
    img_tag = (f'<img src="data:image/jpeg;base64,{img_b64}" alt="{page_id}" style="width:100%">'
               if img_b64 else '<div style="padding:40px;color:#666">Image not available</div>')

    # CER comparison table
    cer_rows = ""
    base_cer = base_stats["cer"]
    for mk, vd in var_data.items():
        vc = vd["stats"]["cer"]
        delta = vc - base_cer
        delta_cls = "delta-better" if delta < -0.005 else ("delta-worse" if delta > 0.005 else "delta-neutral")
        delta_sign = "+" if delta > 0 else ""
        cer_rows += (
            f'<tr>'
            f'<td><span class="variant-dot" style="background:{vd["color"]}"></span>{vd["label"]}</td>'
            f'<td>{vc:.1%}</td>'
            f'<td class="{delta_cls}">{delta_sign}{delta:.1%}</td>'
            f'<td>S={vd["stats"]["substitutions"]} D={vd["stats"]["deletions"]} I={vd["stats"]["insertions"]}</td>'
            f'</tr>'
        )

    # Tab buttons & tab content
    tab_ids = []
    tab_buttons = ""
    tab_panels = ""
    for j, (mk, vd) in enumerate(var_data.items()):
        tid = f"tab-{idx}-{j}"
        tab_ids.append(tid)
        active = " active" if j == 0 else ""
        tab_buttons += f'<button class="tab-btn{active}" data-tab="{tid}" style="border-bottom-color:{vd["color"]}">{vd["label"]}</button>'

        display = "grid" if j == 0 else "none"
        vc = vd["stats"]["cer"]
        delta = vc - base_cer
        delta_sign = "+" if delta > 0 else ""
        delta_cls = "delta-better" if delta < -0.005 else ("delta-worse" if delta > 0.005 else "delta-neutral")
        tab_panels += f"""
        <div class="tab-panel" id="{tid}" style="display:{display}">
          <div class="text-col">
            <div class="col-label baseline-label">Baseline (v11) — CER {base_cer:.1%}</div>
            <div class="text-content">{base_hyp_html}</div>
          </div>
          <div class="text-col">
            <div class="col-label" style="background:{vd["color"]}30;border-left:3px solid {vd["color"]}">{vd["label"]} — CER {vc:.1%} <span class="{delta_cls}">({delta_sign}{delta:.1%})</span></div>
            <div class="text-content">{vd["hyp_html"]}</div>
          </div>
        </div>"""

    excluded_class = " excluded" if excluded else ""
    excluded_banner = ""
    if excluded:
        reason = "ink from reverse page" if "3623" in page_id else "incomplete GT"
        excluded_banner = f'<div class="excluded-banner">EXCLUDED — {reason}</div>'

    m = metadata
    return f"""
    <div class="page-section{excluded_class}" id="page-{idx}">
      {excluded_banner}
      <div class="page-header">
        <h2>{page_id}</h2>
        <span class="page-meta">{m.get('doc_id','')} · {m.get('siglo','')} · {m.get('letra_normalized','')}</span>
        <span class="cer-badge">Baseline CER {base_cer:.1%}</span>
      </div>
      <div class="cer-table-wrap">
        <table class="cer-table">
          <tr><th>Variant</th><th>CER</th><th>Δ</th><th>Errors</th></tr>
          {cer_rows}
        </table>
      </div>
      <div class="page-body">
        <div class="page-image">
          {img_tag}
        </div>
        <div class="page-text-area">
          <div class="tab-bar">{tab_buttons}</div>
          {tab_panels}
        </div>
      </div>
    </div>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    output_path = PROJECT_ROOT / "data" / "evaluation" / "review" / "preprocessing_comparison.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pages = []
    for dataset in ["codea", "toledo"]:
        manifest_path = SUBSETS_DIR / dataset / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        edition = "paleographic" if dataset == "codea" else "editorial"

        for entry in manifest:
            pid = entry["page_id"]
            image_path = SUBSETS_DIR / dataset / "images" / entry["canonical_name"]
            gt_path = SUBSETS_DIR / dataset / "ground_truth" / edition / f"{pid}.txt"
            base_path = RESULTS_DIR / BASELINE / "normalized" / f"{pid}.txt"

            if not gt_path.exists() or not base_path.exists():
                continue

            gt_raw = gt_path.read_text(encoding="utf-8").strip()
            base_text = base_path.read_text(encoding="utf-8").strip()

            # Load variant texts
            variant_texts = {}
            for mk in VARIANTS:
                vp = RESULTS_DIR / mk / "normalized" / f"{pid}.txt"
                if vp.exists():
                    variant_texts[mk] = vp.read_text(encoding="utf-8").strip()

            # Compute baseline CER for sorting
            gt_n = normalize_ws(strip_markers_content(gt_raw))
            base_n = normalize_ws(base_text)
            import editdistance
            base_cer = editdistance.eval(base_n, gt_n) / len(gt_n) if gt_n else 0

            # Compute best variant delta for sorting
            best_delta = 0
            for mk, vtxt in variant_texts.items():
                v_n = normalize_ws(vtxt)
                v_cer = editdistance.eval(v_n, gt_n) / len(gt_n) if gt_n else 0
                delta = v_cer - base_cer
                if delta < best_delta:
                    best_delta = delta

            pages.append({
                "page_id": pid, "dataset": dataset,
                "image_path": str(image_path), "gt_raw": gt_raw,
                "base_text": base_text, "variant_texts": variant_texts,
                "metadata": entry, "base_cer": base_cer, "best_delta": best_delta,
            })

    # Sort: pages where preprocessing helped most first
    pages.sort(key=lambda p: p["best_delta"])

    n_excluded = sum(1 for p in pages if p["page_id"] in EXCLUDED_PAGES)
    n_active = len(pages) - n_excluded
    print(f"Generating preprocessing comparison for {len(pages)} pages ({n_excluded} excluded)...")

    # Build TOC rows
    toc_rows = ""
    for i, p in enumerate(pages):
        is_excl = p["page_id"] in EXCLUDED_PAGES
        excl_cls = " toc-excluded" if is_excl else ""
        cer_cls = "toc-good" if p["base_cer"] < 0.30 else ("toc-mid" if p["base_cer"] < 0.60 else "toc-bad")
        d = p["best_delta"]
        d_cls = "delta-better" if d < -0.005 else ("delta-worse" if d > 0.005 else "delta-neutral")
        d_sign = "+" if d > 0 else ""
        toc_rows += (
            f'<tr class="{cer_cls}{excl_cls}" onclick="document.getElementById(\'page-{i}\').scrollIntoView({{behavior:\'smooth\'}})">'
            f'<td>{i+1}</td><td>{p["page_id"]}</td><td>{p["dataset"]}</td>'
            f'<td>{p["base_cer"]:.1%}</td>'
            f'<td class="{d_cls}">{d_sign}{d:.1%}</td>'
            f'<td>{p["metadata"].get("letra_normalized","")}</td></tr>'
        )

    # Build page sections
    sections = []
    for i, p in enumerate(pages):
        is_excl = p["page_id"] in EXCLUDED_PAGES
        label = " [EXCL]" if is_excl else ""
        print(f"  [{i+1}/{len(pages)}] {p['page_id']}: base CER={p['base_cer']:.1%}, best Δ={p['best_delta']:+.1%}{label}")
        sections.append(render_page(
            p["page_id"], i, p["image_path"], p["gt_raw"],
            p["base_text"], p["variant_texts"], p["metadata"],
            excluded=is_excl,
        ))

    # Aggregate summary
    active_pages = [p for p in pages if p["page_id"] not in EXCLUDED_PAGES]
    n_helped = sum(1 for p in active_pages if p["best_delta"] < -0.005)
    n_hurt = sum(1 for p in active_pages if p["best_delta"] > 0.005)
    n_neutral = len(active_pages) - n_helped - n_hurt

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Preprocessing Comparison — {n_active} pages</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #1a1a2e; color: #eee; }}

  /* Header */
  header {{ padding: 15px 24px; background: #16213e; border-bottom: 1px solid #333; position: sticky; top: 0; z-index: 100; }}
  header h1 {{ font-size: 1.15em; }}
  header p {{ color: #888; font-size: 0.82em; margin-top: 4px; }}
  .summary-pills {{ display: flex; gap: 10px; margin-top: 8px; font-size: 0.8em; }}
  .pill {{ padding: 3px 10px; border-radius: 12px; font-weight: 600; }}
  .pill-better {{ background: #27ae6033; color: #2ecc71; border: 1px solid #27ae6066; }}
  .pill-worse {{ background: #e74c3c33; color: #e74c3c; border: 1px solid #e74c3c66; }}
  .pill-neutral {{ background: #ffffff11; color: #888; border: 1px solid #ffffff22; }}

  /* Controls */
  .controls {{ padding: 8px 24px; background: #0f3460; border-bottom: 1px solid #333; display: flex; gap: 16px; align-items: center; position: sticky; top: 52px; z-index: 99; }}
  .controls label {{ font-size: 0.82em; color: #aaa; cursor: pointer; }}
  .controls input {{ margin-right: 4px; }}

  /* TOC */
  .toc {{ padding: 10px 24px; background: #16213e; border-bottom: 1px solid #333; max-height: 300px; overflow-y: auto; display: none; }}
  body.show-toc .toc {{ display: block; }}
  .toc table {{ width: 100%; border-collapse: collapse; font-size: 0.78em; }}
  .toc th {{ text-align: left; color: #888; padding: 4px 8px; border-bottom: 1px solid #333; position: sticky; top: 0; background: #16213e; }}
  .toc td {{ padding: 4px 8px; cursor: pointer; }}
  .toc tr:hover {{ background: #0f3460; }}
  .toc-good td:nth-child(4) {{ color: #27ae60; }}
  .toc-mid td:nth-child(4) {{ color: #f39c12; }}
  .toc-bad td:nth-child(4) {{ color: #e74c3c; }}
  .toc-excluded {{ opacity: 0.4; }}
  .toc-excluded td:nth-child(2) {{ text-decoration: line-through; }}

  /* Legend */
  .legend {{ padding: 6px 24px; background: #0f3460; font-size: 0.72em; display: flex; gap: 14px; flex-wrap: wrap; border-bottom: 1px solid #333; }}
  .legend-item {{ display: flex; align-items: center; gap: 4px; }}
  .legend-swatch {{ width: 10px; height: 10px; border-radius: 2px; }}

  /* Page sections */
  .page-section {{ border-bottom: 3px solid #0f3460; padding-bottom: 8px; }}
  .page-section.excluded {{ opacity: 0.45; border-left: 4px solid #e74c3c; }}
  .excluded-banner {{ background: #e74c3c; color: #fff; padding: 5px 24px; font-size: 0.82em; font-weight: 600; }}
  .page-header {{ padding: 10px 24px; background: #16213e; display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }}
  .page-header h2 {{ font-size: 1.05em; }}
  .page-meta {{ font-size: 0.78em; color: #888; }}
  .cer-badge {{ background: #e94560; color: #fff; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 0.82em; }}

  /* CER comparison table */
  .cer-table-wrap {{ padding: 6px 24px 8px; }}
  .cer-table {{ border-collapse: collapse; font-size: 0.78em; }}
  .cer-table th {{ text-align: left; color: #888; padding: 3px 10px; border-bottom: 1px solid #333; }}
  .cer-table td {{ padding: 3px 10px; }}
  .variant-dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }}

  /* Delta colors */
  .delta-better {{ color: #2ecc71; font-weight: 600; }}
  .delta-worse {{ color: #e74c3c; font-weight: 600; }}
  .delta-neutral {{ color: #888; }}

  /* Page body: image + text area */
  .page-body {{ display: grid; grid-template-columns: minmax(250px, 1fr) 2fr; min-height: 350px; }}
  .page-image {{ background: #111; overflow: hidden; max-height: 700px; overflow-y: auto; }}
  .page-image img {{ width: 100%; display: block; }}

  /* Tabs */
  .page-text-area {{ display: flex; flex-direction: column; overflow: hidden; }}
  .tab-bar {{ display: flex; gap: 0; background: #16213e; border-bottom: 2px solid #333; }}
  .tab-btn {{ background: transparent; color: #888; border: none; padding: 8px 18px; font-size: 0.85em; cursor: pointer; border-bottom: 3px solid transparent; transition: all 0.15s; }}
  .tab-btn:hover {{ color: #ccc; background: #0f346044; }}
  .tab-btn.active {{ color: #fff; border-bottom-color: currentColor; }}

  /* Tab panels: 2 columns for baseline vs variant */
  .tab-panel {{ display: none; grid-template-columns: 1fr 1fr; flex: 1; overflow: hidden; }}
  .tab-panel[style*="display: grid"], .tab-panel[style*="display:grid"] {{ display: grid !important; }}
  .text-col {{ display: flex; flex-direction: column; border-right: 1px solid #333; overflow: hidden; }}
  .text-col:last-child {{ border-right: none; }}
  .col-label {{ padding: 5px 12px; font-size: 0.72em; color: #bbb; font-weight: 600; background: #0f3460; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .baseline-label {{ background: #16213e; }}
  .text-content {{ padding: 10px 12px; font-family: 'Courier New', monospace; font-size: 11.5px; line-height: 1.75; white-space: pre-wrap; word-wrap: break-word; overflow-y: auto; flex: 1; max-height: 550px; }}

  /* Character diff highlights */
  .correct {{ color: #999; }}
  .substitution {{ background: rgba(243,156,18,0.22); color: #f5d76e; border-bottom: 2px solid #f39c12; cursor: help; }}
  .deletion {{ background: rgba(231,76,60,0.18); color: #e88; border-bottom: 2px solid #e74c3c; cursor: help; }}
  .insertion {{ background: rgba(52,152,219,0.18); color: #7ec8e3; border-bottom: 2px solid #3498db; cursor: help; }}
  body.hide-correct .correct {{ opacity: 0.12; }}

  /* Footer */
  footer {{ padding: 12px 24px; background: #16213e; border-top: 1px solid #333; font-size: 0.72em; color: #555; }}
  footer a {{ color: #e94560; }}

  @media print {{
    .controls, .toc, .tab-bar {{ display: none !important; }}
    .tab-panel {{ display: grid !important; }}
    .page-image {{ max-height: none; }}
  }}
</style>
</head>
<body>
<header>
  <h1>Image Preprocessing Comparison — {n_active} pages</h1>
  <p>Baseline (v11 anti-halluc) vs CLAHE / Sharpen / Denoise. Character-level diff against GT. Sorted by best improvement first.</p>
  <div class="summary-pills">
    <span class="pill pill-better">{n_helped} pages improved</span>
    <span class="pill pill-worse">{n_hurt} pages worse</span>
    <span class="pill pill-neutral">{n_neutral} neutral</span>
  </div>
</header>
<div class="controls">
  <label><input type="checkbox" id="dimCorrect"> Dim correct chars</label>
  <label><input type="checkbox" id="showToc" checked> Table of contents</label>
</div>
<div class="toc">
  <table>
    <tr><th>#</th><th>Page</th><th>Dataset</th><th>Base CER</th><th>Best Δ</th><th>Script</th></tr>
    {toc_rows}
  </table>
</div>
<div class="legend">
  <div class="legend-item"><div class="legend-swatch" style="background:#999"></div> Correct</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#f39c12"></div> Substitution</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#e74c3c"></div> Deletion</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#3498db"></div> Insertion</div>
  <div class="legend-item" style="margin-left:16px"><div class="legend-swatch" style="background:#9b59b6"></div> CLAHE</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#e67e22"></div> Sharpen</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#1abc9c"></div> Denoise</div>
</div>
{"".join(sections)}
<footer>
  CER = (S+D+I) / |GT|. Content-mode marker stripping. Whitespace-normalized. Both columns show model transcription diff'd against GT.
  <a href="https://github.com/crlsrmrlsz/paleo-ocr">paleo-ocr</a>
</footer>
<script>
  // Dim correct
  document.getElementById('dimCorrect').addEventListener('change', function() {{
    document.body.classList.toggle('hide-correct', this.checked);
  }});
  // TOC
  document.getElementById('showToc').addEventListener('change', function() {{
    document.body.classList.toggle('show-toc', this.checked);
  }});
  document.body.classList.add('show-toc');

  // Tab switching
  document.querySelectorAll('.tab-btn').forEach(function(btn) {{
    btn.addEventListener('click', function() {{
      var section = this.closest('.page-section');
      section.querySelectorAll('.tab-btn').forEach(function(b) {{ b.classList.remove('active'); }});
      section.querySelectorAll('.tab-panel').forEach(function(p) {{ p.style.display = 'none'; }});
      this.classList.add('active');
      document.getElementById(this.dataset.tab).style.display = 'grid';
    }});
  }});
</script>
</body></html>"""

    output_path.write_text(html, encoding="utf-8")
    size_mb = len(html) / (1024 * 1024)
    print(f"\nSaved to {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
