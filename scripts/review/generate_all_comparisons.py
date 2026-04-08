#!/usr/bin/env python3
"""Generate single HTML with all 60 evaluation pages — character-level CER visualization.

Each page gets its own section: manuscript image | GT | transcription with diff.
Navigable via table of contents with CER for each page.

Usage:
    python scripts/review/generate_all_comparisons.py
"""

import base64
import json
import re
import sys
from pathlib import Path

from rapidfuzz.distance import Levenshtein

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SUBSETS_DIR = PROJECT_ROOT / "data" / "subsets"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Marker stripping — same as compute_metrics.py content mode (with DOTALL)
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
# YOLO detection
# ---------------------------------------------------------------------------

_yolo_model = None

def detect_layout(image_path: str):
    global _yolo_model
    try:
        from scripts.dspy_pipeline.layout import load_yolo_model, detect_regions
        from PIL import Image

        if _yolo_model is None:
            _yolo_model = load_yolo_model()

        regions = detect_regions(_yolo_model, str(image_path), confidence=0.10)
        img = Image.open(str(image_path))
        img_w, img_h = img.size
        img.close()

        boxes = []
        for r in regions:
            x1, y1, x2, y2 = r.bbox
            boxes.append({
                "x1_pct": x1 / img_w * 100, "y1_pct": y1 / img_h * 100,
                "w_pct": (x2 - x1) / img_w * 100, "h_pct": (y2 - y1) / img_h * 100,
                "type": r.region_type, "confidence": r.confidence,
            })
        return boxes
    except Exception as e:
        return []


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def compute_char_alignment(gt, hyp):
    return Levenshtein.opcodes(gt, hyp)


def compute_stats(gt, hyp, opcodes):
    correct = subs = dels = ins = 0
    for op, gs, ge, hs, he in opcodes:
        if op == "equal":
            correct += ge - gs
        elif op == "replace":
            gl, hl = ge - gs, he - hs
            subs += min(gl, hl)
            if gl > hl: dels += gl - hl
            else: ins += hl - gl
        elif op == "delete":
            dels += ge - gs
        elif op == "insert":
            ins += he - hs
    gt_len = correct + subs + dels
    cer = (subs + dels + ins) / gt_len if gt_len > 0 else 0
    return {"correct": correct, "substitutions": subs, "deletions": dels,
            "insertions": ins, "gt_chars": gt_len, "cer": cer}


def _esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


REGION_COLORS = {"main_text": "#27ae60", "margin": "#e94560", "graphic": "#f39c12", "stamp": "#3498db"}


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


IMG_MAX_WIDTH = 800  # Resize images to this max width for HTML embedding


def resize_image_for_html(image_path):
    """Resize image to IMG_MAX_WIDTH for HTML embedding. Returns base64 JPEG."""
    from PIL import Image as PILImage
    try:
        img = PILImage.open(str(image_path))
        if img.width > IMG_MAX_WIDTH:
            ratio = IMG_MAX_WIDTH / img.width
            img = img.resize((IMG_MAX_WIDTH, int(img.height * ratio)), PILImage.LANCZOS)
        from io import BytesIO
        buf = BytesIO()
        if img.mode == "RGBA":
            img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=75)
        img.close()
        return base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return None


def render_page_section(page_id, model, image_path, gt_raw, hyp_text, metadata, idx, excluded=False):
    gt_stripped = strip_markers_content(gt_raw)
    gt_n = re.sub(r"\s+", " ", gt_stripped).strip()
    hyp_n = re.sub(r"\s+", " ", hyp_text).strip()

    opcodes = compute_char_alignment(gt_n, hyp_n)
    stats = compute_stats(gt_n, hyp_n, opcodes)
    s = stats

    gt_html = render_side(gt_n, hyp_n, opcodes, "gt")
    hyp_html = render_side(gt_n, hyp_n, opcodes, "hyp")

    # Image — resized for smaller HTML
    img_b64 = resize_image_for_html(image_path)
    if img_b64:
        img_tag = f'<img src="data:image/jpeg;base64,{img_b64}" alt="{page_id}" style="width:100%">'
    else:
        img_tag = '<div style="padding:40px;color:#666">Image not available</div>'

    # YOLO boxes
    yolo_boxes = detect_layout(str(image_path))
    yolo_html = ""
    for box in yolo_boxes:
        color = REGION_COLORS.get(box["type"], "#888")
        yolo_html += (
            f'<div class="yolo-box" style="left:{box["x1_pct"]:.1f}%;top:{box["y1_pct"]:.1f}%;'
            f'width:{box["w_pct"]:.1f}%;height:{box["h_pct"]:.1f}%;border-color:{color}">'
            f'<span class="yolo-label" style="background:{color}">{box["type"]} ({box["confidence"]:.0%})</span></div>'
        )

    # Markers note
    raw_markers = re.findall(r'\[[^\]]+\]', gt_raw)
    markers_note = ""
    if raw_markers:
        markers_note = f'<div class="markers-note">Stripped markers: {", ".join(_esc(m) for m in raw_markers[:6])}{"..." if len(raw_markers) > 6 else ""}</div>'

    # Exclusion banner
    excluded_class = " excluded" if excluded else ""
    excluded_banner = ""
    if excluded:
        reason = "ink from reverse page too visible" if "3623" in page_id else "GT only covers first paragraph"
        excluded_banner = f'<div class="excluded-banner">EXCLUDED FROM METRICS — {reason}</div>'

    m = metadata
    return f"""
    <div class="page-section{excluded_class}" id="page-{idx}">
      {excluded_banner}
      <div class="page-header">
        <h2>{page_id}</h2>
        <div class="page-meta">
          <span>Model: <strong>{model}</strong></span>
          <span>Doc: {m.get('doc_id','')}</span>
          <span>Century: {m.get('siglo','')}</span>
          <span>Script: {m.get('letra_normalized','')}</span>
          <span class="cer-badge">CER {s['cer']:.1%}</span>
        </div>
        <div class="page-stats">
          S={s['substitutions']} D={s['deletions']} I={s['insertions']} / {s['gt_chars']} chars
        </div>
        {markers_note}
      </div>
      <div class="page-columns">
        <div class="page-col page-col-image">
          <div class="img-container">{img_tag}{yolo_html}</div>
        </div>
        <div class="page-col page-col-text">
          <div class="col-label">Ground Truth</div>
          <div class="text-content">{gt_html}</div>
        </div>
        <div class="page-col page-col-text">
          <div class="col-label">Model Transcription</div>
          <div class="text-content">{hyp_html}</div>
        </div>
      </div>
    </div>"""


def main():
    output_path = PROJECT_ROOT / "data" / "evaluation" / "review" / "error_comparison.html"

    # Collect all pages
    pages = []
    for dataset in ["codea", "toledo"]:
        manifest_path = SUBSETS_DIR / dataset / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        edition = "paleographic" if dataset == "codea" else "editorial"

        model = "pipeline/11_yolo_crop_anti_halluc"

        for entry in manifest:
            pid = entry["page_id"]
            image_path = SUBSETS_DIR / dataset / "images" / entry["canonical_name"]
            gt_path = SUBSETS_DIR / dataset / "ground_truth" / edition / f"{pid}.txt"
            hyp_path = RESULTS_DIR / model / "normalized" / f"{pid}.txt"

            if not gt_path.exists() or not hyp_path.exists():
                continue

            gt_raw = gt_path.read_text(encoding="utf-8").strip()
            hyp_text = hyp_path.read_text(encoding="utf-8").strip()
            gt_stripped = strip_markers_content(gt_raw)
            gt_n = re.sub(r"\s+", " ", gt_stripped).strip()
            hyp_n = re.sub(r"\s+", " ", hyp_text).strip()

            import editdistance
            cer = editdistance.eval(hyp_n, gt_n) / len(gt_n) if gt_n else 0

            pages.append({
                "page_id": pid, "model": model, "dataset": dataset,
                "image_path": str(image_path), "gt_raw": gt_raw,
                "hyp_text": hyp_text, "metadata": entry, "cer": cer,
            })

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from pipeline_config import EXCLUDED_PAGES

    pages.sort(key=lambda p: p["cer"])
    n_excluded = sum(1 for p in pages if p["page_id"] in EXCLUDED_PAGES)
    n_active = len(pages) - n_excluded
    print(f"Generating single HTML for {len(pages)} pages ({n_excluded} excluded, images resized to {IMG_MAX_WIDTH}px)...")

    output_path = PROJECT_ROOT / "data" / "evaluation" / "review" / "error_comparison.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build TOC
    toc_rows = []
    for i, p in enumerate(pages):
        is_excluded = p["page_id"] in EXCLUDED_PAGES
        cer_class = "toc-excluded" if is_excluded else ("toc-good" if p["cer"] < 0.30 else ("toc-mid" if p["cer"] < 0.60 else "toc-bad"))
        exc_mark = " [EXCLUDED]" if is_excluded else ""
        toc_rows.append(
            f'<tr class="{cer_class}" onclick="document.getElementById(\'page-{i}\').scrollIntoView({{behavior:\'smooth\'}})">'
            f'<td>{i+1}</td><td>{p["page_id"]}{exc_mark}</td><td>{p["dataset"]}</td>'
            f'<td>{p["cer"]:.1%}</td><td>{p["metadata"].get("letra_normalized","")}</td></tr>'
        )

    # Build page sections
    sections = []
    for i, p in enumerate(pages):
        is_excluded = p["page_id"] in EXCLUDED_PAGES
        exc_label = " [EXCLUDED]" if is_excluded else ""
        print(f"  [{i+1}/{len(pages)}] {p['page_id']}: CER={p['cer']:.1%}{exc_label}")
        sections.append(render_page_section(
            p["page_id"], p["model"], p["image_path"],
            p["gt_raw"], p["hyp_text"], p["metadata"], i,
            excluded=is_excluded,
        ))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CER Visualization — {len(pages)} pages ({n_active} evaluated, {n_excluded} excluded)</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #1a1a2e; color: #eee; }}
  header {{ padding: 15px 30px; background: #16213e; border-bottom: 1px solid #333; position: sticky; top: 0; z-index: 100; }}
  header h1 {{ font-size: 1.2em; }}
  header p {{ color: #888; font-size: 0.85em; margin-top: 4px; }}
  .controls {{ padding: 8px 30px; background: #0f3460; border-bottom: 1px solid #333; display: flex; gap: 16px; position: sticky; top: 56px; z-index: 99; }}
  .controls label {{ font-size: 0.85em; color: #aaa; cursor: pointer; }}
  .controls input {{ margin-right: 4px; }}
  .toc {{ padding: 10px 30px; background: #16213e; border-bottom: 1px solid #333; max-height: 300px; overflow-y: auto; display: none; }}
  body.show-toc .toc {{ display: block; }}
  .toc table {{ width: 100%; border-collapse: collapse; font-size: 0.8em; }}
  .toc th {{ text-align: left; color: #888; padding: 4px 8px; border-bottom: 1px solid #333; }}
  .toc td {{ padding: 4px 8px; cursor: pointer; }}
  .toc tr:hover {{ background: #0f3460; }}
  .toc-good td:nth-child(4) {{ color: #27ae60; }}
  .toc-mid td:nth-child(4) {{ color: #f39c12; }}
  .toc-bad td:nth-child(4) {{ color: #e74c3c; }}
  .toc-excluded {{ opacity: 0.4; }}
  .toc-excluded td:nth-child(2) {{ text-decoration: line-through; }}
  .legend {{ padding: 6px 30px; background: #0f3460; font-size: 0.75em; display: flex; gap: 16px; flex-wrap: wrap; border-bottom: 1px solid #333; }}
  .legend-item {{ display: flex; align-items: center; gap: 4px; }}
  .legend-swatch {{ width: 10px; height: 10px; border-radius: 2px; }}
  .page-section {{ border-bottom: 3px solid #0f3460; padding: 16px 0; }}
  .page-section.excluded {{ opacity: 0.5; border-left: 4px solid #e74c3c; }}
  .excluded-banner {{ background: #e74c3c; color: #fff; padding: 6px 30px; font-size: 0.85em; font-weight: 600; }}
  .page-header {{ padding: 10px 30px; background: #16213e; }}
  .page-header h2 {{ font-size: 1.1em; display: inline; }}
  .page-meta {{ display: inline; margin-left: 16px; font-size: 0.8em; color: #888; }}
  .page-meta span {{ margin-right: 12px; }}
  .cer-badge {{ background: #e94560; color: #fff; padding: 2px 8px; border-radius: 4px; font-weight: 700; }}
  .page-stats {{ font-size: 0.75em; color: #666; margin-top: 4px; padding-left: 30px; }}
  .markers-note {{ font-size: 0.7em; color: #555; padding: 2px 30px; }}
  .page-columns {{ display: grid; grid-template-columns: 1fr 1fr 1fr; min-height: 400px; }}
  .page-col {{ border-right: 1px solid #333; overflow: hidden; }}
  .page-col:last-child {{ border-right: none; }}
  .page-col-image {{ background: #111; }}
  .page-col-image img {{ width: 100%; display: block; }}
  .img-container {{ position: relative; }}
  .col-label {{ background: #0f3460; padding: 4px 12px; font-size: 0.75em; color: #888; font-weight: 600; }}
  .page-col-text {{ overflow-y: auto; max-height: 600px; }}
  .text-content {{ padding: 12px; font-family: 'Courier New', monospace; font-size: 12px; line-height: 1.8; white-space: pre-wrap; word-wrap: break-word; }}
  .correct {{ color: #aaa; }}
  .substitution {{ background: rgba(243,156,18,0.25); color: #f5d76e; border-bottom: 2px solid #f39c12; cursor: help; }}
  .deletion {{ background: rgba(231,76,60,0.2); color: #e88; border-bottom: 2px solid #e74c3c; cursor: help; }}
  .insertion {{ background: rgba(52,152,219,0.2); color: #7ec8e3; border-bottom: 2px solid #3498db; cursor: help; }}
  body.hide-correct .correct {{ opacity: 0.12; }}
  .yolo-box {{ position: absolute; border: 2px solid; pointer-events: none; display: none; }}
  .yolo-label {{ position: absolute; top: -16px; left: 0; font-size: 9px; color: #fff; padding: 1px 4px; border-radius: 2px; white-space: nowrap; }}
  body.show-yolo .yolo-box {{ display: block; }}
  footer {{ padding: 12px 30px; background: #16213e; border-top: 1px solid #333; font-size: 0.75em; color: #555; }}
  footer a {{ color: #e94560; }}
</style>
</head>
<body>
<header>
  <h1>CER Visualization — {n_active} pages evaluated, {n_excluded} excluded</h1>
  <p>Character-level alignment matching evaluation pipeline. Sorted by CER (best first). Model: pipeline yolo_crop_bare_merge.</p>
</header>
<div class="controls">
  <label><input type="checkbox" id="dimCorrect"> Dim correct</label>
  <label><input type="checkbox" id="showYolo"> YOLO regions</label>
  <label><input type="checkbox" id="showToc" checked> Table of contents</label>
</div>
<div class="toc">
  <table>
    <tr><th>#</th><th>Page</th><th>Dataset</th><th>CER</th><th>Script</th></tr>
    {"".join(toc_rows)}
  </table>
</div>
<div class="legend">
  <div class="legend-item"><div class="legend-swatch" style="background:#aaa"></div> Correct</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#f39c12"></div> Substitution</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#e74c3c"></div> Deletion</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#3498db"></div> Insertion</div>
  <div class="legend-item" style="margin-left:12px"><div class="legend-swatch" style="border:2px solid #27ae60"></div> YOLO: main</div>
  <div class="legend-item"><div class="legend-swatch" style="border:2px solid #e94560"></div> YOLO: margin</div>
</div>
{"".join(sections)}
<footer>
  CER = (S+D+I) / |GT|. Content-mode marker stripping (re.DOTALL). Whitespace-normalized.
  <a href="https://github.com/crlsrmrlsz/paleo-ocr">paleo-ocr</a>
</footer>
<script>
  document.getElementById('dimCorrect').addEventListener('change', function() {{
    document.body.classList.toggle('hide-correct', this.checked);
  }});
  document.getElementById('showYolo').addEventListener('change', function() {{
    document.body.classList.toggle('show-yolo', this.checked);
  }});
  document.getElementById('showToc').addEventListener('change', function() {{
    document.body.classList.toggle('show-toc', this.checked);
  }});
  document.body.classList.add('show-toc');
</script>
</body></html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"\nSaved to {output_path} ({len(html) // (1024*1024)} MB)")


if __name__ == "__main__":
    main()
