#!/usr/bin/env python3
"""Generate interactive HTML comparison: manuscript image | ground truth | model transcription.

Character-level alignment matching exactly how CER is computed:
- Same content-mode marker stripping as compute_metrics.py
- Same whitespace normalization (collapse to single spaces)
- Character-level Levenshtein opcodes for S/D/I counts
- Displayed CER = (S+D+I) / len(GT) — identical to evaluation pipeline

Usage:
    python scripts/review/generate_error_comparison.py
    python scripts/review/generate_error_comparison.py --page codea_CODEA-2025_1r --model pipeline/10_yolo_crop_bare_merge
"""

import argparse
import base64
import json
import re
import sys
from pathlib import Path

from rapidfuzz.distance import Levenshtein

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SUBSETS_DIR = PROJECT_ROOT / "data" / "subsets"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


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
    """Strip GT markers using content mode (same as evaluation pipeline)."""
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
# Data loading
# ---------------------------------------------------------------------------

def load_page_data(page_id: str, model: str, edition: str = "paleographic"):
    dataset = page_id.split("_")[0]
    manifest_path = SUBSETS_DIR / dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = next(e for e in manifest if e["page_id"] == page_id)

    image_path = SUBSETS_DIR / dataset / "images" / entry["canonical_name"]
    gt_path = SUBSETS_DIR / dataset / "ground_truth" / edition / f"{page_id}.txt"
    hyp_path = RESULTS_DIR / model / "normalized" / f"{page_id}.txt"

    raw_gt = gt_path.read_text(encoding="utf-8").strip()
    stripped_gt = strip_markers_content(raw_gt)

    return {
        "page_id": page_id,
        "model": model,
        "image_path": image_path,
        "gt_raw": raw_gt,
        "gt_text": stripped_gt,
        "hyp_text": hyp_path.read_text(encoding="utf-8").strip(),
        "metadata": entry,
    }


# ---------------------------------------------------------------------------
# YOLO layout detection
# ---------------------------------------------------------------------------

def detect_layout(image_path: str):
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from scripts.dspy_pipeline.layout import load_yolo_model, detect_regions
        from PIL import Image

        model = load_yolo_model()
        regions = detect_regions(model, str(image_path), confidence=0.10)

        img = Image.open(str(image_path))
        img_w, img_h = img.size
        img.close()

        boxes = []
        for r in regions:
            x1, y1, x2, y2 = r.bbox
            boxes.append({
                "x1_pct": x1 / img_w * 100,
                "y1_pct": y1 / img_h * 100,
                "w_pct": (x2 - x1) / img_w * 100,
                "h_pct": (y2 - y1) / img_h * 100,
                "type": r.region_type,
                "confidence": r.confidence,
            })
        return boxes
    except Exception as e:
        print(f"YOLO detection failed: {e}", file=sys.stderr)
        return []


# ---------------------------------------------------------------------------
# Character-level alignment — exactly matches CER computation
# ---------------------------------------------------------------------------

def compute_char_alignment(gt: str, hyp: str):
    """Character-level Levenshtein opcodes — same as CER computation.

    Returns list of (op, gt_chars, hyp_chars) tuples where op is
    'equal', 'replace', 'delete', or 'insert', and *_chars are the
    character spans involved.
    """
    opcodes = Levenshtein.opcodes(gt, hyp)
    segments = []
    for op, gs, ge, hs, he in opcodes:
        segments.append((op, gt[gs:ge], hyp[hs:he]))
    return segments


def compute_stats(segments):
    """Compute S/D/I from character-level alignment segments.

    Matches CER formula: CER = (S + D + I) / len(GT)
    """
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


def render_char_spans(segments, side="gt"):
    """Render character-level alignment as HTML spans.

    side='gt': shows GT text with deletions (missing in model) highlighted
    side='hyp': shows model text with insertions (not in GT) highlighted
    """
    parts = []
    for op, g_span, h_span in segments:
        if op == "equal":
            text = g_span if side == "gt" else h_span
            parts.append(f'<span class="correct">{_esc(text)}</span>')
        elif op == "replace":
            if side == "gt":
                parts.append(
                    f'<span class="substitution" title="Model: {_esc(h_span)}">{_esc(g_span)}</span>'
                )
            else:
                parts.append(
                    f'<span class="substitution" title="GT: {_esc(g_span)}">{_esc(h_span)}</span>'
                )
        elif op == "delete":
            if side == "gt":
                parts.append(
                    f'<span class="deletion" title="Missing in model">{_esc(g_span)}</span>'
                )
        elif op == "insert":
            if side == "hyp":
                parts.append(
                    f'<span class="insertion" title="Not in GT">{_esc(h_span)}</span>'
                )
    return "".join(parts)


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

REGION_COLORS = {
    "main_text": "#27ae60",
    "margin": "#e94560",
    "graphic": "#f39c12",
    "stamp": "#3498db",
}


def generate_html(data, segments, stats, yolo_boxes):
    img_bytes = data["image_path"].read_bytes()
    img_b64 = base64.standard_b64encode(img_bytes).decode("utf-8")

    gt_html = render_char_spans(segments, side="gt")
    hyp_html = render_char_spans(segments, side="hyp")
    meta = data["metadata"]

    # YOLO overlay divs
    yolo_divs = []
    for box in yolo_boxes:
        color = REGION_COLORS.get(box["type"], "#888")
        yolo_divs.append(
            f'<div class="yolo-box" style="left:{box["x1_pct"]:.2f}%;top:{box["y1_pct"]:.2f}%;'
            f'width:{box["w_pct"]:.2f}%;height:{box["h_pct"]:.2f}%;border-color:{color}">'
            f'<span class="yolo-label" style="background:{color}">'
            f'{box["type"]} ({box["confidence"]:.0%})</span></div>'
        )
    yolo_overlay_html = "\n".join(yolo_divs)

    # Markers note
    raw_markers = re.findall(r'\[[^\]]+\]', data["gt_raw"])
    markers_note = ""
    if raw_markers:
        markers_note = (
            f'<div class="note">GT markers stripped (content mode, same as evaluation): '
            f'{", ".join(_esc(m) for m in raw_markers[:8])}'
            f'{"..." if len(raw_markers) > 8 else ""}</div>'
        )

    s = stats

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>HTR CER Visualization — {data['page_id']}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #1a1a2e; color: #eee; }}

  header {{ padding: 20px 30px; background: #16213e; border-bottom: 1px solid #333; }}
  header h1 {{ font-size: 1.4em; font-weight: 600; }}
  header p {{ color: #999; font-size: 0.9em; margin-top: 4px; }}

  .meta-bar {{ display: flex; gap: 24px; padding: 12px 30px; background: #0f3460; font-size: 0.85em; flex-wrap: wrap; }}
  .meta-bar span {{ color: #aaa; }}
  .meta-bar strong {{ color: #e94560; }}

  .note {{ padding: 8px 30px; background: #1a1a2e; border-bottom: 1px solid #333; font-size: 0.8em; color: #888; }}

  .stats {{ display: flex; gap: 0; border-bottom: 1px solid #333; }}
  .stat {{ flex: 1; text-align: center; padding: 14px; border-right: 1px solid #333; }}
  .stat:last-child {{ border-right: none; }}
  .stat-value {{ font-size: 1.6em; font-weight: 700; }}
  .stat-label {{ font-size: 0.75em; color: #888; margin-top: 2px; }}
  .stat-cer .stat-value {{ color: #e94560; }}
  .stat-ok .stat-value {{ color: #27ae60; }}
  .stat-sub .stat-value {{ color: #f39c12; }}
  .stat-del .stat-value {{ color: #e74c3c; }}
  .stat-ins .stat-value {{ color: #3498db; }}
  .stat-formula {{ font-size: 0.7em; color: #666; margin-top: 4px; }}

  .controls {{ padding: 10px 30px; background: #16213e; display: flex; gap: 16px; align-items: center; border-bottom: 1px solid #333; }}
  .controls label {{ font-size: 0.85em; color: #aaa; cursor: pointer; }}
  .controls input[type=checkbox] {{ margin-right: 4px; }}

  .columns {{ display: grid; grid-template-columns: 1fr 1fr 1fr; height: calc(100vh - 280px); }}
  .col {{ overflow-y: auto; border-right: 1px solid #333; }}
  .col:last-child {{ border-right: none; }}
  .col-header {{ position: sticky; top: 0; background: #16213e; padding: 10px 16px; font-weight: 600;
                  font-size: 0.9em; border-bottom: 1px solid #333; z-index: 10; }}

  .col-image {{ background: #111; }}
  .img-container {{ position: relative; }}
  .img-container img {{ width: 100%; display: block; }}

  .yolo-box {{ position: absolute; border: 2px solid; pointer-events: none; display: none; }}
  .yolo-label {{ position: absolute; top: -18px; left: 0; font-size: 10px; color: #fff;
                  padding: 1px 5px; border-radius: 2px; white-space: nowrap; }}
  body.show-yolo .yolo-box {{ display: block; }}

  .col-text {{ padding: 16px; font-family: 'Courier New', monospace; font-size: 13px;
               line-height: 1.9; background: #1a1a2e; white-space: pre-wrap; word-wrap: break-word; }}

  .correct {{ color: #ccc; }}
  .substitution {{ background: rgba(243, 156, 18, 0.25); color: #f5d76e; border-bottom: 2px solid #f39c12; cursor: help; }}
  .deletion {{ background: rgba(231, 76, 60, 0.2); color: #e88; border-bottom: 2px solid #e74c3c; cursor: help; }}
  .insertion {{ background: rgba(52, 152, 219, 0.2); color: #7ec8e3; border-bottom: 2px solid #3498db; cursor: help; }}

  body.hide-correct .correct {{ opacity: 0.15; }}

  .legend {{ display: flex; gap: 20px; padding: 8px 30px; background: #0f3460; font-size: 0.8em; flex-wrap: wrap; }}
  .legend-item {{ display: flex; align-items: center; gap: 5px; }}
  .legend-swatch {{ width: 12px; height: 12px; border-radius: 2px; }}

  footer {{ padding: 12px 30px; background: #16213e; border-top: 1px solid #333; font-size: 0.8em; color: #666; }}
  footer a {{ color: #e94560; }}
</style>
</head>
<body>

<header>
  <h1>CER Visualization — Character-Level Error Analysis</h1>
  <p>Exact character-level alignment as computed by the evaluation pipeline. Each highlighted character is a substitution, deletion, or insertion as counted in CER.</p>
</header>

<div class="meta-bar">
  <span>Page: <strong>{data['page_id']}</strong></span>
  <span>Model: <strong>{data['model']}</strong></span>
  <span>Document: <strong>{meta.get('doc_id', '')}</strong></span>
  <span>Century: <strong>{meta.get('siglo', '')}</strong></span>
  <span>Script: <strong>{meta.get('letra_normalized', '')}</strong></span>
</div>

{markers_note}

<div class="stats">
  <div class="stat stat-cer">
    <div class="stat-value">{s['cer']:.1%}</div>
    <div class="stat-label">CER</div>
    <div class="stat-formula">({s['substitutions']}+{s['deletions']}+{s['insertions']}) / {s['gt_chars']}</div>
  </div>
  <div class="stat stat-ok"><div class="stat-value">{s['correct']}</div><div class="stat-label">Correct chars</div></div>
  <div class="stat stat-sub"><div class="stat-value">{s['substitutions']}</div><div class="stat-label">Substitutions (S)</div></div>
  <div class="stat stat-del"><div class="stat-value">{s['deletions']}</div><div class="stat-label">Deletions (D)</div></div>
  <div class="stat stat-ins"><div class="stat-value">{s['insertions']}</div><div class="stat-label">Insertions (I)</div></div>
</div>

<div class="legend">
  <div class="legend-item"><div class="legend-swatch" style="background:#ccc"></div> Correct</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#f39c12"></div> Substitution (wrong char)</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#e74c3c"></div> Deletion (in GT, missing in model)</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#3498db"></div> Insertion (in model, not in GT)</div>
  <div class="legend-item" style="margin-left:20px"><div class="legend-swatch" style="border:2px solid #27ae60"></div> YOLO: main_text</div>
  <div class="legend-item"><div class="legend-swatch" style="border:2px solid #e94560"></div> YOLO: margin</div>
  <div class="legend-item"><div class="legend-swatch" style="border:2px solid #f39c12"></div> YOLO: graphic</div>
</div>

<div class="controls">
  <label><input type="checkbox" id="dimCorrect"> Dim correct characters</label>
  <label><input type="checkbox" id="showYolo"> Show YOLO layout regions</label>
  <label><input type="checkbox" id="syncScroll" checked> Sync scroll</label>
</div>

<div class="columns">
  <div class="col col-image" id="colImage">
    <div class="col-header">Manuscript Image</div>
    <div class="img-container">
      <img src="data:image/jpeg;base64,{img_b64}" alt="Manuscript page">
      {yolo_overlay_html}
    </div>
  </div>
  <div class="col" id="colGT">
    <div class="col-header">Ground Truth — {s['gt_chars']} chars (markers stripped)</div>
    <div class="col-text" id="gtText">{gt_html}</div>
  </div>
  <div class="col" id="colHyp">
    <div class="col-header">Model Transcription — errors highlighted</div>
    <div class="col-text" id="hypText">{hyp_html}</div>
  </div>
</div>

<footer>
  CER = (S+D+I) / |GT| = ({s['substitutions']}+{s['deletions']}+{s['insertions']}) / {s['gt_chars']} = {s['cer']:.4f}.
  GT processed with content-mode marker stripping and whitespace normalization, identical to evaluation pipeline.
  Generated by <a href="https://github.com/crlsrmrlsz/paleo-ocr">paleo-ocr</a>.
  Layout: <a href="https://huggingface.co/biglam/medieval-manuscript-yolov11">biglam/medieval-manuscript-yolov11</a>.
</footer>

<script>
  document.getElementById('dimCorrect').addEventListener('change', function() {{
    document.body.classList.toggle('hide-correct', this.checked);
  }});
  document.getElementById('showYolo').addEventListener('change', function() {{
    document.body.classList.toggle('show-yolo', this.checked);
  }});
  const colGT = document.getElementById('colGT');
  const colHyp = document.getElementById('colHyp');
  let syncing = false;
  function syncScroll(source, target) {{
    if (syncing || !document.getElementById('syncScroll').checked) return;
    syncing = true;
    const ratio = source.scrollTop / (source.scrollHeight - source.clientHeight || 1);
    target.scrollTop = ratio * (target.scrollHeight - target.clientHeight);
    syncing = false;
  }}
  colGT.addEventListener('scroll', () => syncScroll(colGT, colHyp));
  colHyp.addEventListener('scroll', () => syncScroll(colHyp, colGT));
</script>
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser(description="Generate CER visualization HTML")
    parser.add_argument("--page", default="codea_CODEA-2025_1r", help="Page ID")
    parser.add_argument("--model", default="pipeline/11_yolo_crop_anti_halluc", help="Model name")
    parser.add_argument("--edition", default="paleographic", help="GT edition")
    parser.add_argument("--output", default=None, help="Output HTML path")
    args = parser.parse_args()

    data = load_page_data(args.page, args.model, args.edition)

    # Same normalization as evaluation pipeline
    gt_norm = re.sub(r"\s+", " ", data["gt_text"]).strip()
    hyp_norm = re.sub(r"\s+", " ", data["hyp_text"]).strip()

    segments = compute_char_alignment(gt_norm, hyp_norm)
    stats = compute_stats(segments)

    # Verify CER matches
    import editdistance
    verify_cer = editdistance.eval(hyp_norm, gt_norm) / len(gt_norm)
    assert abs(stats["cer"] - verify_cer) < 0.001, f"CER mismatch: {stats['cer']} vs {verify_cer}"

    yolo_boxes = detect_layout(data["image_path"])

    print(f"Page: {data['page_id']}, Model: {data['model']}")
    print(f"CER: {stats['cer']:.4f} = ({stats['substitutions']}+{stats['deletions']}+{stats['insertions']})/{stats['gt_chars']}")
    print(f"Verified: editdistance CER = {verify_cer:.4f}")
    print(f"YOLO regions: {len(yolo_boxes)}")

    html = generate_html(data, segments, stats, yolo_boxes)

    output_path = Path(args.output) if args.output else PROJECT_ROOT / "data" / "visualization" / "error_comparison.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
