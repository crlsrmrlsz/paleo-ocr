#!/usr/bin/env python3
"""Generate HTML comparing original manuscript images against preprocessing variants.

Shows original | CLAHE | Sharpen | Denoise side-by-side for every evaluation page.
Includes zoom, synchronized scroll, and per-page metadata.

Usage:
    python scripts/review/generate_image_preprocessing_comparison.py
"""

import base64
import json
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image as PILImage

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SUBSETS_DIR = PROJECT_ROOT / "data" / "subsets"

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from pipeline_config import EXCLUDED_PAGES

# Import preprocessing functions
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.dspy_pipeline.image_utils import (
    preprocess_clahe,
    preprocess_denoise,
    preprocess_sharpen,
)

IMG_MAX_WIDTH = 600  # Per-column image width


def image_to_b64(image_path: str, max_width: int = IMG_MAX_WIDTH) -> str | None:
    """Resize and encode image as base64 JPEG."""
    try:
        img = PILImage.open(image_path)
        if img.width > max_width:
            ratio = max_width / img.width
            img = img.resize((max_width, int(img.height * ratio)), PILImage.LANCZOS)
        if img.mode == "RGBA":
            img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=80)
        img.close()
        return base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"  Warning: could not process {image_path}: {e}")
        return None


def render_page(page_id, idx, image_path, metadata, excluded=False):
    """Render one page section with 4 image columns."""
    # Generate preprocessed versions from the original full-page image
    clahe_path = preprocess_clahe(str(image_path))
    sharpen_path = preprocess_sharpen(str(image_path))
    denoise_path = preprocess_denoise(str(image_path))

    # Encode all four
    orig_b64 = image_to_b64(str(image_path))
    clahe_b64 = image_to_b64(clahe_path)
    sharpen_b64 = image_to_b64(sharpen_path)
    denoise_b64 = image_to_b64(denoise_path)

    def img_tag(b64, alt):
        if b64:
            return f'<img src="data:image/jpeg;base64,{b64}" alt="{alt}" loading="lazy">'
        return '<div class="no-img">Image unavailable</div>'

    m = metadata
    excluded_cls = " excluded" if excluded else ""
    excluded_banner = ""
    if excluded:
        reason = "ink from reverse page" if "3623" in page_id else "incomplete GT"
        excluded_banner = f'<div class="excluded-banner">EXCLUDED — {reason}</div>'

    return f"""
    <div class="page-section{excluded_cls}" id="page-{idx}">
      {excluded_banner}
      <div class="page-header">
        <h2>{page_id}</h2>
        <span class="meta">{m.get('doc_id','')} · {m.get('siglo','')} · {m.get('letra_normalized','')}</span>
      </div>
      <div class="image-grid" data-page="{idx}">
        <div class="image-col">
          <div class="col-label">Original</div>
          <div class="img-wrap" data-col="0">{img_tag(orig_b64, 'Original')}</div>
        </div>
        <div class="image-col">
          <div class="col-label clahe-label">CLAHE</div>
          <div class="img-wrap" data-col="1">{img_tag(clahe_b64, 'CLAHE')}</div>
        </div>
        <div class="image-col">
          <div class="col-label sharpen-label">Sharpen</div>
          <div class="img-wrap" data-col="2">{img_tag(sharpen_b64, 'Sharpen')}</div>
        </div>
        <div class="image-col">
          <div class="col-label denoise-label">Denoise</div>
          <div class="img-wrap" data-col="3">{img_tag(denoise_b64, 'Denoise')}</div>
        </div>
      </div>
    </div>"""


def main():
    output_path = PROJECT_ROOT / "data" / "evaluation" / "review" / "image_preprocessing_comparison.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect pages
    pages = []
    for dataset in ["codea", "toledo"]:
        manifest_path = SUBSETS_DIR / dataset / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest:
            pid = entry["page_id"]
            image_path = SUBSETS_DIR / dataset / "images" / entry["canonical_name"]
            if not image_path.exists():
                continue
            pages.append({
                "page_id": pid, "dataset": dataset,
                "image_path": str(image_path), "metadata": entry,
            })

    pages.sort(key=lambda p: p["page_id"])
    n_excluded = sum(1 for p in pages if p["page_id"] in EXCLUDED_PAGES)
    n_active = len(pages) - n_excluded
    print(f"Generating image comparison for {len(pages)} pages ({n_excluded} excluded)...")

    # TOC
    toc_rows = ""
    for i, p in enumerate(pages):
        is_excl = p["page_id"] in EXCLUDED_PAGES
        excl_cls = " toc-excluded" if is_excl else ""
        toc_rows += (
            f'<tr class="{excl_cls}" onclick="document.getElementById(\'page-{i}\').scrollIntoView({{behavior:\'smooth\'}})">'
            f'<td>{i+1}</td><td>{p["page_id"]}</td><td>{p["dataset"]}</td>'
            f'<td>{p["metadata"].get("siglo","")}</td>'
            f'<td>{p["metadata"].get("letra_normalized","")}</td></tr>'
        )

    # Page sections
    sections = []
    for i, p in enumerate(pages):
        is_excl = p["page_id"] in EXCLUDED_PAGES
        label = " [EXCL]" if is_excl else ""
        print(f"  [{i+1}/{len(pages)}] {p['page_id']}{label}")
        sections.append(render_page(
            p["page_id"], i, p["image_path"], p["metadata"], excluded=is_excl,
        ))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Image Preprocessing Comparison — {n_active} pages</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #1a1a2e; color: #eee; }}

  header {{ padding: 15px 24px; background: #16213e; border-bottom: 1px solid #333; position: sticky; top: 0; z-index: 100; }}
  header h1 {{ font-size: 1.15em; }}
  header p {{ color: #888; font-size: 0.82em; margin-top: 4px; }}

  .controls {{ padding: 8px 24px; background: #0f3460; border-bottom: 1px solid #333; display: flex; gap: 16px; align-items: center; position: sticky; top: 52px; z-index: 99; flex-wrap: wrap; }}
  .controls label {{ font-size: 0.82em; color: #aaa; cursor: pointer; }}
  .controls input[type=checkbox] {{ margin-right: 4px; }}
  .controls select {{ background: #16213e; color: #eee; border: 1px solid #444; padding: 3px 8px; border-radius: 4px; font-size: 0.82em; }}

  .toc {{ padding: 10px 24px; background: #16213e; border-bottom: 1px solid #333; max-height: 280px; overflow-y: auto; display: none; }}
  body.show-toc .toc {{ display: block; }}
  .toc table {{ width: 100%; border-collapse: collapse; font-size: 0.78em; }}
  .toc th {{ text-align: left; color: #888; padding: 4px 8px; border-bottom: 1px solid #333; position: sticky; top: 0; background: #16213e; }}
  .toc td {{ padding: 4px 8px; cursor: pointer; }}
  .toc tr:hover {{ background: #0f3460; }}
  .toc-excluded {{ opacity: 0.4; }}
  .toc-excluded td:nth-child(2) {{ text-decoration: line-through; }}

  .page-section {{ border-bottom: 3px solid #0f3460; }}
  .page-section.excluded {{ opacity: 0.45; border-left: 4px solid #e74c3c; }}
  .excluded-banner {{ background: #e74c3c; color: #fff; padding: 5px 24px; font-size: 0.82em; font-weight: 600; }}
  .page-header {{ padding: 10px 24px; background: #16213e; display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }}
  .page-header h2 {{ font-size: 1em; }}
  .meta {{ font-size: 0.78em; color: #888; }}

  .image-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 0; }}
  /* When columns are hidden */
  body.hide-clahe .image-grid .image-col:nth-child(2),
  body.hide-sharpen .image-grid .image-col:nth-child(3),
  body.hide-denoise .image-grid .image-col:nth-child(4) {{ display: none; }}
  body.hide-clahe .image-grid {{ grid-template-columns: repeat(3, 1fr); }}
  body.hide-sharpen .image-grid {{ grid-template-columns: repeat(3, 1fr); }}
  body.hide-denoise .image-grid {{ grid-template-columns: repeat(3, 1fr); }}
  body.hide-clahe.hide-sharpen .image-grid {{ grid-template-columns: repeat(2, 1fr); }}
  body.hide-clahe.hide-denoise .image-grid {{ grid-template-columns: repeat(2, 1fr); }}
  body.hide-sharpen.hide-denoise .image-grid {{ grid-template-columns: repeat(2, 1fr); }}
  body.hide-clahe.hide-sharpen.hide-denoise .image-grid {{ grid-template-columns: 1fr; }}

  .image-col {{ border-right: 1px solid #333; }}
  .image-col:last-child {{ border-right: none; }}
  .col-label {{ padding: 5px 12px; font-size: 0.72em; font-weight: 600; color: #bbb; background: #0f3460; text-align: center; }}
  .clahe-label {{ background: #9b59b622; border-top: 2px solid #9b59b6; }}
  .sharpen-label {{ background: #e67e2222; border-top: 2px solid #e67e22; }}
  .denoise-label {{ background: #1abc9c22; border-top: 2px solid #1abc9c; }}
  .img-wrap {{ background: #111; overflow: hidden; cursor: crosshair; position: relative; }}
  .img-wrap img {{ width: 100%; display: block; transition: transform 0.1s ease-out; transform-origin: 0 0; }}
  .no-img {{ padding: 40px; color: #666; text-align: center; }}

  /* Zoom lens overlay */
  .zoom-lens {{ display: none; position: fixed; width: 300px; height: 300px; border: 2px solid #e94560; border-radius: 8px; overflow: hidden; z-index: 200; pointer-events: none; box-shadow: 0 4px 20px rgba(0,0,0,0.6); }}
  .zoom-lens img {{ position: absolute; transform-origin: 0 0; }}
  body.zoom-on .zoom-lens {{ display: block; }}

  footer {{ padding: 12px 24px; background: #16213e; border-top: 1px solid #333; font-size: 0.72em; color: #555; }}
  footer a {{ color: #e94560; }}

  @media print {{
    .controls, .toc {{ display: none !important; }}
    .page-section {{ page-break-inside: avoid; }}
  }}
</style>
</head>
<body>
<header>
  <h1>Image Preprocessing Comparison — {n_active} pages</h1>
  <p>Original manuscript images vs CLAHE (contrast) / Sharpen (edges) / Denoise (bilateral filter). Hover to compare details.</p>
</header>
<div class="controls">
  <label><input type="checkbox" id="showToc" checked> TOC</label>
  <label><input type="checkbox" id="toggleZoom"> Zoom lens</label>
  <label>Zoom:
    <select id="zoomLevel">
      <option value="2">2×</option>
      <option value="3" selected>3×</option>
      <option value="4">4×</option>
      <option value="5">5×</option>
    </select>
  </label>
  <span style="border-left:1px solid #555;height:20px;margin:0 4px"></span>
  <label><input type="checkbox" id="showClahe" checked> <span style="color:#9b59b6">CLAHE</span></label>
  <label><input type="checkbox" id="showSharpen" checked> <span style="color:#e67e22">Sharpen</span></label>
  <label><input type="checkbox" id="showDenoise" checked> <span style="color:#1abc9c">Denoise</span></label>
  <span style="border-left:1px solid #555;height:20px;margin:0 4px"></span>
  <label><input type="checkbox" id="syncScroll" checked> Sync scroll</label>
</div>
<div class="toc">
  <table>
    <tr><th>#</th><th>Page</th><th>Dataset</th><th>Century</th><th>Script</th></tr>
    {toc_rows}
  </table>
</div>
<div class="zoom-lens" id="zoomLens"><img id="zoomImg" src="" alt="zoom"></div>
{"".join(sections)}
<footer>
  CLAHE: clip_limit=2.0, tile=8×8 (LAB L-channel). Sharpen: UnsharpMask r=2 p=150 t=3. Denoise: bilateral d=9 σ_color=75 σ_space=75.
  <a href="https://github.com/crlsrmrlsz/paleo-ocr">paleo-ocr</a>
</footer>
<script>
  // TOC toggle
  document.getElementById('showToc').addEventListener('change', function() {{
    document.body.classList.toggle('show-toc', this.checked);
  }});
  document.body.classList.add('show-toc');

  // Column visibility
  document.getElementById('showClahe').addEventListener('change', function() {{
    document.body.classList.toggle('hide-clahe', !this.checked);
  }});
  document.getElementById('showSharpen').addEventListener('change', function() {{
    document.body.classList.toggle('hide-sharpen', !this.checked);
  }});
  document.getElementById('showDenoise').addEventListener('change', function() {{
    document.body.classList.toggle('hide-denoise', !this.checked);
  }});

  // Zoom lens
  var zoomOn = false;
  var zoomLevel = 3;
  var zoomLens = document.getElementById('zoomLens');
  var zoomImg = document.getElementById('zoomImg');

  document.getElementById('toggleZoom').addEventListener('change', function() {{
    zoomOn = this.checked;
    document.body.classList.toggle('zoom-on', zoomOn);
  }});
  document.getElementById('zoomLevel').addEventListener('change', function() {{
    zoomLevel = parseInt(this.value);
  }});

  document.addEventListener('mousemove', function(e) {{
    if (!zoomOn) return;
    var wrap = e.target.closest('.img-wrap');
    if (!wrap) {{ zoomLens.style.display = 'none'; return; }}
    var img = wrap.querySelector('img');
    if (!img) return;

    zoomLens.style.display = 'block';
    zoomLens.style.left = (e.clientX + 20) + 'px';
    zoomLens.style.top = (e.clientY - 150) + 'px';

    // Clamp to viewport
    var rect = zoomLens.getBoundingClientRect();
    if (rect.right > window.innerWidth) zoomLens.style.left = (e.clientX - 320) + 'px';
    if (rect.bottom > window.innerHeight) zoomLens.style.top = (e.clientY - 320) + 'px';
    if (rect.top < 0) zoomLens.style.top = '10px';

    zoomImg.src = img.src;
    var imgRect = img.getBoundingClientRect();
    var natW = img.naturalWidth;
    var natH = img.naturalHeight;
    var scaleX = natW / imgRect.width;
    var scaleY = natH / imgRect.height;
    var mouseX = (e.clientX - imgRect.left) * scaleX;
    var mouseY = (e.clientY - imgRect.top) * scaleY;

    var lensW = 300;
    var lensH = 300;
    zoomImg.style.width = (natW * zoomLevel) + 'px';
    zoomImg.style.height = (natH * zoomLevel) + 'px';
    zoomImg.style.left = (-mouseX * zoomLevel + lensW / 2) + 'px';
    zoomImg.style.top = (-mouseY * zoomLevel + lensH / 2) + 'px';
  }});

  // Synchronized scroll within each page grid
  var syncing = false;
  document.getElementById('syncScroll').addEventListener('change', function() {{
    syncing = this.checked;
  }});
  syncing = true;

  document.querySelectorAll('.img-wrap').forEach(function(wrap) {{
    wrap.addEventListener('scroll', function() {{
      if (!syncing) return;
      var grid = this.closest('.image-grid');
      if (!grid) return;
      var scrollTop = this.scrollTop;
      var scrollLeft = this.scrollLeft;
      grid.querySelectorAll('.img-wrap').forEach(function(other) {{
        if (other !== wrap) {{
          other.scrollTop = scrollTop;
          other.scrollLeft = scrollLeft;
        }}
      }});
    }});
  }});
</script>
</body></html>"""

    output_path.write_text(html, encoding="utf-8")
    size_mb = len(html) / (1024 * 1024)
    print(f"\nSaved to {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
