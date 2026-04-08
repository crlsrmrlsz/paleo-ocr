"""Per-block manuscript transcription pipeline.

Orchestrates: full-page rough pass → YOLO layout detection → per-block
high-res transcription → intelligent strip merge → block merge.

Uses ManuscriptTranscriber for the rough pass, BlockTranscriber for each
region/strip, and StripMerger to deduplicate overlapping strip transcriptions.
"""

import logging
from itertools import groupby

import dspy

from .image_utils import ensure_under_size_limit
from .layout import Region, detect_and_crop

logger = logging.getLogger(__name__)


def transcribe_page_blocks(
    image_path: str,
    block_program,
    yolo_model,
    model_name: str | None = None,
    strip_merger=None,
    domain_knowledge: dict | None = None,
    rlm_corrector=None,
    critic=None,
    document_context: str = "",
    fallback_program=None,
    **kwargs,
) -> str:
    """Transcribe a manuscript page using per-block processing.

    Steps:
        1. YOLO layout detection on full-resolution image
        2. Subdivide large regions into strips for max model resolution
        3. Per-strip transcription at high resolution
        4. Strip merge (LLM deduplicates overlapping strips)
        5. Block-level merge with region-type formatting
        6. Critic review (optional, different model)

    Falls back to fallback_program (full-page) if no regions detected.
    """
    # 1. Layout detection + strip subdivision
    logger.info("  Layout detection...")
    regions = detect_and_crop(yolo_model, image_path, model_name=model_name)

    if not regions:
        logger.info("  No regions detected — fallback to full-page transcription")
        if fallback_program is not None:
            safe_path = ensure_under_size_limit(image_path)
            result = fallback_program(page_image=dspy.Image(safe_path))
            return result.transcription
        return ""

    logger.info("  %d regions/strips: %s",
                len(regions),
                ", ".join(f"{r.region_type}({r.confidence:.2f})" for r in regions))

    # 4. Per-strip transcription
    strip_results: list[tuple[Region, str]] = []
    for i, region in enumerate(regions):
        logger.info("  Strip %d/%d [%s]...", i + 1, len(regions), region.region_type)
        try:
            kwargs = dict(
                block_image=dspy.Image(region.crop_path),
                region_type=region.region_type,
            )
            if domain_knowledge:
                kwargs.update(domain_knowledge)
            result = block_program(**kwargs)
            result_text = result.transcription

            # RLM abbreviation lookup on this strip
            if rlm_corrector is not None:
                try:
                    rlm_result = rlm_corrector(strip_transcription=result_text)
                    corrected = rlm_result.corrected_transcription
                    logger.info("    → RLM: %d → %d chars", len(result_text), len(corrected))
                    result_text = corrected
                except Exception as e:
                    logger.warning("    → RLM skip: %s", e)

            strip_results.append((region, result_text))
            logger.info("    → %d chars", len(result_text))
        except Exception as e:
            logger.warning("    → ERROR: %s (skipping strip)", e)

    if not strip_results:
        logger.info("  All strips failed")
        return ""

    # 5. Group consecutive strips of same type+confidence (same parent region)
    #    and merge them intelligently
    block_results = _merge_strip_groups(
        strip_results, strip_merger, domain_knowledge
    )

    # 6. Final block-level merge
    merged = merge_blocks(block_results)
    logger.info("  Merged: %d chars from %d blocks", len(merged), len(block_results))

    # 7. Critic review (if available)
    if critic is not None:
        logger.info("  Critic review...")
        try:
            review = critic(
                merged_transcription=merged,
                rough_transcription="",
                page_context=document_context,
            )
            logger.info("  Critic: %s", review.review[:200])

            # Apply critic's corrections
            if review.corrected_transcription and review.corrected_transcription.strip():
                merged = review.corrected_transcription

            # Reprocess flagged strips (1 retry max)
            reprocess = review.strips_to_reprocess.strip().lower()
            if reprocess and reprocess != "none":
                try:
                    indices = [int(x.strip()) for x in reprocess.split(",")]
                    logger.info("  Reprocessing strips: %s", indices)
                    for idx in indices:
                        if 0 <= idx < len(regions):
                            region = regions[idx]
                            logger.info("    Re-strip %d [%s]...", idx, region.region_type)
                            kwargs = dict(
                                block_image=dspy.Image(region.crop_path),
                                region_type=region.region_type,
                            )
                            if domain_knowledge:
                                kwargs.update(domain_knowledge)
                            re_result = block_program(**kwargs)
                            logger.info("    → %d chars", len(re_result.transcription))
                except (ValueError, IndexError) as e:
                    logger.warning("  Reprocess parse error: %s", e)
        except Exception as e:
            logger.warning("  Critic ERROR: %s", e)

    return merged


def _merge_strip_groups(
    strip_results: list[tuple[Region, str]],
    strip_merger,
    domain_knowledge: dict | None = None,
) -> list[tuple[str, str]]:
    """Group consecutive strips from the same parent region and merge each group.

    Strips from the same YOLO region share the same confidence value (used as
    a grouping key). Single-strip groups pass through unchanged. Multi-strip
    groups go through the StripMerger for intelligent deduplication.
    """
    block_results = []

    # Group consecutive strips by (region_type, confidence) — strips from
    # the same parent region have identical confidence values
    for key, group in groupby(strip_results, key=lambda x: (x[0].region_type, x[0].confidence)):
        strips = list(group)
        region_type = key[0]

        if len(strips) == 1:
            # Single strip — no merging needed
            block_results.append((region_type, strips[0][1]))
        elif strip_merger is not None:
            # Multiple strips from same region — use LLM to merge
            labeled = []
            for i, (region, text) in enumerate(strips):
                labeled.append(f"---STRIP {i + 1}---\n{text.strip()}")
            strip_text = "\n\n".join(labeled)

            logger.info("  Merging %d strips [%s] via LLM...", len(strips), region_type)
            try:
                kwargs = dict(
                    strip_transcriptions=strip_text,
                    region_type=region_type,
                )
                if domain_knowledge:
                    kwargs.update(domain_knowledge)
                result = strip_merger(**kwargs)
                block_results.append((region_type, result.transcription))
                logger.info("    → merged to %d chars", len(result.transcription))
            except Exception as e:
                logger.warning("    → Merge ERROR: %s (concatenating instead)", e)
                # Fallback: simple concatenation
                combined = "\n".join(s[1].strip() for s in strips)
                block_results.append((region_type, combined))
        else:
            # No merger available — simple concatenation
            combined = "\n".join(s[1].strip() for s in strips)
            block_results.append((region_type, combined))

    return block_results


def merge_blocks(block_results: list[tuple[str, str]]) -> str:
    """Merge block transcriptions in reading order.

    - main_text: insert as-is
    - margin: wrap as [margen: TEXT] (matches CODEA GT format)
    - graphic/stamp: insert as-is (model produces [rúbrica], [cruz], [signo])
    """
    parts = []
    for region_type, text in block_results:
        text = text.strip()
        if not text:
            continue
        if region_type == "margin":
            parts.append(f"[margen: {text}]")
        else:
            parts.append(text)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Model-First Pipeline
# ---------------------------------------------------------------------------


def transcribe_page_model_first(
    image_path: str,
    page_analyzer,
    region_decider,
    strip_transcriber,
    yolo_model,
    model_name: str | None = None,
    strip_merger=None,
    fallback_program=None,
    **kwargs,
) -> str:
    """Model-first pipeline: analyze → decide → transcribe → merge.

    1. PageAnalyzer describes the full page
    2. YOLO detects regions with confidence
    3. RegionDecider evaluates each region
    4. Only 'transcribe' regions get per-strip processing
    5. StripMerger deduplicates overlaps
    6. Plain text output, no markers
    """
    from PIL import Image

    from .layout import (
        _merge_bboxes,
        crop_region,
        detect_regions,
        get_max_px,
        resolve_overlaps,
        sort_reading_order,
        subdivide_region,
    )
    from .region_decider import format_yolo_regions, parse_decisions

    # 1. Page analysis
    safe_path = ensure_under_size_limit(image_path)
    logger.info("  Page analysis...")
    analysis = page_analyzer(page_image=dspy.Image(safe_path))
    page_description = analysis.page_description
    logger.info("  Description: %s", page_description[:150])

    # 2. YOLO detection + overlap resolution
    logger.info("  YOLO detection...")
    regions = detect_regions(yolo_model, image_path)
    if not regions:
        logger.info("  No regions — fallback")
        if fallback_program:
            return fallback_program(page_image=dspy.Image(safe_path)).transcription
        return ""

    regions = resolve_overlaps(regions)
    if not regions:
        logger.info("  No regions after overlap resolution — fallback")
        if fallback_program:
            return fallback_program(page_image=dspy.Image(safe_path)).transcription
        return ""

    img = Image.open(image_path)
    img_w, img_h = img.width, img.height
    img.close()

    regions = sort_reading_order(regions, img_h)

    # 3. Region decision
    logger.info("  Region decision (%d regions)...", len(regions))
    yolo_text = format_yolo_regions(regions, img_w, img_h)
    decision_result = region_decider(
        page_description=page_description,
        yolo_regions=yolo_text,
    )
    decisions = parse_decisions(decision_result.decisions, len(regions))

    for i, (action, detail) in enumerate(decisions):
        r = regions[i]
        logger.info("    Region %d [%s %.0f%%]: %s",
                     i, r.region_type, r.confidence * 100, action)

    # 4. Build transcribe list (handle merge decisions)
    transcribe_regions = []
    merged_into = set()
    for i, (action, detail) in enumerate(decisions):
        if i in merged_into:
            continue
        if action == "transcribe":
            transcribe_regions.append(regions[i])
        elif action == "merge" and detail:
            try:
                target = int(detail)
                if 0 <= target < len(regions):
                    merged = Region(
                        bbox=_merge_bboxes(regions[i].bbox, regions[target].bbox),
                        region_type="main_text",
                        confidence=max(regions[i].confidence, regions[target].confidence),
                    )
                    transcribe_regions.append(merged)
                    merged_into.add(target)
            except (ValueError, IndexError):
                transcribe_regions.append(regions[i])

    if not transcribe_regions:
        logger.info("  All regions skipped — fallback")
        if fallback_program:
            return fallback_program(page_image=dspy.Image(safe_path)).transcription
        return ""

    # 5. Strip splitting + transcription
    max_px = get_max_px(model_name) if model_name else 1568
    all_strips = []
    for region in transcribe_regions:
        strips = subdivide_region(image_path, region, max_px)
        all_strips.extend(strips)

    logger.info("  Transcribing %d strips...", len(all_strips))
    strip_results = []
    for i, strip in enumerate(all_strips):
        logger.info("    Strip %d/%d...", i + 1, len(all_strips))
        try:
            if not strip.crop_path:
                crop_region(image_path, strip)
            result = strip_transcriber(
                block_image=dspy.Image(strip.crop_path),
            )
            strip_results.append((strip, result.transcription))
            logger.info("      → %d chars", len(result.transcription))
        except Exception as e:
            logger.warning("      → ERROR: %s", e)

    if not strip_results:
        return ""

    # 6. Merge — plain text, no markers
    block_results = _merge_strip_groups(strip_results, strip_merger)
    parts = [text.strip() for _, text in block_results if text.strip()]
    merged = "\n".join(parts)
    logger.info("  Merged: %d chars", len(merged))
    return merged


# ---------------------------------------------------------------------------
# Strips + Page Description Pipeline
# ---------------------------------------------------------------------------


def auto_crop_margins(image_path: str, padding_pct: float = 0.02, yolo_model=None):
    """Crop margins to focus on text content.

    If yolo_model is provided, uses YOLO (conf=0.05) to detect text regions
    and crops only left/right margins (keeps full height). Falls back to
    full image if YOLO detection covers less than 50% of the page.

    Without yolo_model, uses simple ink threshold (< 200).

    Returns (x1, y1, x2, y2) of the content area, or None if no ink found.
    """
    from PIL import Image as PILImage

    img = PILImage.open(image_path)
    w, h = img.size

    if yolo_model is not None:
        from .layout import detect_regions, _merge_bboxes

        regions = detect_regions(yolo_model, image_path, confidence=0.05)
        text_regions = [r for r in regions if r.region_type in ('main_text', 'graphic')]

        if text_regions:
            bbox = text_regions[0].bbox
            for r in text_regions[1:]:
                bbox = _merge_bboxes(bbox, r.bbox)

            # Safety: skip crop if area < 50% of page or confidence < 0.6
            det_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            max_conf = max(r.confidence for r in text_regions)
            if det_area >= 0.5 * w * h and max_conf >= 0.6:
                # Use only horizontal crop, keep full height
                pad_x = int(w * padding_pct)
                x1 = max(0, bbox[0] - pad_x)
                x2 = min(w, bbox[2] + pad_x)
                img.close()
                logger.info("  YOLO crop: %d→%d width (was %d, saved %d%%, conf=%.2f)",
                            w, x2 - x1, w, (1 - (x2 - x1) / w) * 100, max_conf)
                return (x1, 0, x2, h)
            elif det_area < 0.5 * w * h:
                logger.info("  YOLO detection too small (%.0f%% of page) — using full image",
                            det_area / (w * h) * 100)
            else:
                logger.info("  YOLO low confidence (%.2f) — using full image", max_conf)
        else:
            logger.info("  YOLO: no detection — using full image")

        img.close()
        return (0, 0, w, h)

    # Fallback: simple ink threshold
    import numpy as np

    arr = np.array(img.convert("L"))
    ink_mask = arr < 200
    rows = np.any(ink_mask, axis=1)
    cols = np.any(ink_mask, axis=0)

    if not rows.any() or not cols.any():
        img.close()
        return None

    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]

    pad_y = int(h * padding_pct)
    pad_x = int(w * padding_pct)
    r_min = max(0, r_min - pad_y)
    r_max = min(h - 1, r_max + pad_y)
    c_min = max(0, c_min - pad_x)
    c_max = min(w - 1, c_max + pad_x)

    img.close()
    return (c_min, r_min, c_max + 1, r_max + 1)


def fixed_strip_split(image_path: str, content_bbox, max_px: int, overlap: int = 200):
    """Split content area into fixed-height horizontal strips.

    Returns list of Region objects with crop_path set.
    """
    import tempfile
    from pathlib import Path
    from PIL import Image as PILImage
    from .layout import Region
    from .image_utils import ensure_under_size_limit

    x1, y1, x2, y2 = content_bbox
    content_h = y2 - y1

    if content_h <= max_px:
        # Single strip — no splitting needed
        img = PILImage.open(image_path)
        crop = img.crop(content_bbox)
        if crop.mode == "RGBA":
            crop = crop.convert("RGB")
        tmp = Path(tempfile.gettempdir()) / f"{Path(image_path).stem}_full.jpg"
        crop.save(str(tmp), format="JPEG", quality=95)
        crop.close()
        img.close()
        final = ensure_under_size_limit(str(tmp))
        return [Region(bbox=content_bbox, region_type="main_text", confidence=1.0, crop_path=final)]

    # Split into strips
    step = max_px - overlap
    if step <= 0:
        step = max_px

    img = PILImage.open(image_path)
    strips = []
    strip_y = y1
    idx = 0
    while strip_y < y2:
        strip_y2 = min(strip_y + max_px, y2)

        # Absorb tiny trailing strip into last one
        if strip_y2 - strip_y < max_px * 0.2 and strips:
            last = strips[-1]
            lx1, ly1, lx2, _ = last.bbox
            strips[-1] = Region(bbox=(lx1, ly1, lx2, y2), region_type="main_text", confidence=1.0)
            # Re-crop
            crop = img.crop((lx1, ly1, lx2, y2))
            if crop.mode == "RGBA":
                crop = crop.convert("RGB")
            tmp = Path(tempfile.gettempdir()) / f"{Path(image_path).stem}_strip{len(strips)-1}.jpg"
            crop.save(str(tmp), format="JPEG", quality=95)
            crop.close()
            strips[-1].crop_path = ensure_under_size_limit(str(tmp))
            break

        bbox = (x1, strip_y, x2, strip_y2)
        crop = img.crop(bbox)
        if crop.mode == "RGBA":
            crop = crop.convert("RGB")
        tmp = Path(tempfile.gettempdir()) / f"{Path(image_path).stem}_strip{idx}.jpg"
        crop.save(str(tmp), format="JPEG", quality=95)
        crop.close()
        final = ensure_under_size_limit(str(tmp))
        strips.append(Region(bbox=bbox, region_type="main_text", confidence=1.0, crop_path=final))

        strip_y += step
        idx += 1

    img.close()
    return strips


def tile_grid_split(
    image_path: str,
    content_bbox,
    max_px: int,
    overlap: int = 200,
) -> list[list]:
    """Split content area into a 2D grid of tiles that fit within max_px on both axes.

    Returns a list of rows, each row a list of Region objects with crop_path set.
    Grid order: top→bottom, left→right.
    """
    import tempfile
    from pathlib import Path
    from PIL import Image as PILImage
    from .layout import Region
    from .image_utils import ensure_under_size_limit

    x1, y1, x2, y2 = content_bbox

    # Compute column positions
    step = max_px - overlap
    if step <= 0:
        step = max_px

    col_starts = []
    cx = x1
    while cx < x2:
        col_starts.append(cx)
        cx += step
    # Absorb tiny trailing column
    if len(col_starts) > 1:
        last_w = x2 - col_starts[-1]
        if last_w < max_px * 0.2:
            col_starts.pop()

    # Compute row positions
    row_starts = []
    ry = y1
    while ry < y2:
        row_starts.append(ry)
        ry += step
    # Absorb tiny trailing row
    if len(row_starts) > 1:
        last_h = y2 - row_starts[-1]
        if last_h < max_px * 0.2:
            row_starts.pop()

    img = PILImage.open(image_path)
    grid = []

    for ri, ry in enumerate(row_starts):
        ry2 = min(ry + max_px, y2)
        if ri == len(row_starts) - 1:
            ry2 = y2  # last row extends to bottom

        row = []
        for ci, cx in enumerate(col_starts):
            cx2 = min(cx + max_px, x2)
            if ci == len(col_starts) - 1:
                cx2 = x2  # last col extends to right edge

            bbox = (cx, ry, cx2, ry2)
            crop = img.crop(bbox)
            if crop.mode == "RGBA":
                crop = crop.convert("RGB")
            tmp = Path(tempfile.gettempdir()) / f"{Path(image_path).stem}_r{ri}c{ci}.jpg"
            crop.save(str(tmp), format="JPEG", quality=95)
            crop.close()
            final = ensure_under_size_limit(str(tmp))
            row.append(Region(bbox=bbox, region_type="main_text", confidence=1.0, crop_path=final))

        grid.append(row)

    img.close()
    return grid


def transcribe_page_strips_page_desc(
    image_path: str,
    page_analyzer,
    strip_transcriber,
    strip_merger,
    model_name: str | None = None,
    fallback_program=None,
    **kwargs,
) -> str:
    """Strips + page description pipeline: analyze → auto-crop → fixed strips → transcribe → merge.

    No layout detection. Auto-crops blank margins, splits into fixed strips
    at MODEL_MAX_PX height, transcribes each strip, merges with page description.
    """
    from .layout import get_max_px

    # 1. Page analysis
    safe_path = ensure_under_size_limit(image_path)
    logger.info("  Page analysis...")
    analysis = page_analyzer(page_image=dspy.Image(safe_path))
    page_description = analysis.page_description
    logger.info("  Description: %s", page_description[:150])

    # 2. Auto-crop margins
    content_bbox = auto_crop_margins(image_path)
    if content_bbox is None:
        logger.info("  No ink found — fallback")
        if fallback_program:
            return fallback_program(page_image=dspy.Image(safe_path)).transcription
        return ""

    x1, y1, x2, y2 = content_bbox
    logger.info("  Content area: (%d,%d)-(%d,%d) = %dx%d", x1, y1, x2, y2, x2-x1, y2-y1)

    # 3. Fixed strip splitting
    max_px = get_max_px(model_name) if model_name else 1568
    strips = fixed_strip_split(image_path, content_bbox, max_px)
    logger.info("  %d strips (fixed %dpx height)", len(strips), max_px)

    # 4. Transcribe each strip
    strip_results = []
    for i, strip in enumerate(strips):
        logger.info("    Strip %d/%d...", i + 1, len(strips))
        try:
            result = strip_transcriber(block_image=dspy.Image(strip.crop_path))
            strip_results.append((strip, result.transcription))
            logger.info("      → %d chars", len(result.transcription))
        except Exception as e:
            logger.warning("      → ERROR: %s", e)

    if not strip_results:
        return ""

    # 5. Merge with page description context
    if len(strip_results) == 1:
        return strip_results[0][1].strip()

    labeled = []
    for i, (_, text) in enumerate(strip_results):
        labeled.append(f"---STRIP {i + 1}---\n{text.strip()}")
    strip_text = "\n\n".join(labeled)

    logger.info("  Merging %d strips...", len(strip_results))
    try:
        result = strip_merger(
            strip_transcriptions=strip_text,
            page_description=page_description,
        )
        merged = result.transcription.strip()
        logger.info("  Merged: %d chars", len(merged))
        return merged
    except Exception as e:
        logger.warning("  Merge ERROR: %s (concatenating)", e)
        return "\n".join(text.strip() for _, text in strip_results)


def transcribe_page_strips_no_context(
    image_path: str,
    strip_transcriber,
    strip_merger,
    model_name: str | None = None,
    fallback_program=None,
    **kwargs,
) -> str:
    """No-context strip pipeline: auto-crop → strips → transcribe → merge.

    Same as strips_page_desc but without page analysis. No context to transcriber or merger.
    Isolates strip-based splitting for comparison with tiles_2d.
    """
    from .layout import get_max_px

    # 1. Auto-crop margins
    content_bbox = auto_crop_margins(image_path)
    if content_bbox is None:
        logger.info("  No ink found — fallback")
        if fallback_program:
            safe_path = ensure_under_size_limit(image_path)
            return fallback_program(page_image=dspy.Image(safe_path)).transcription
        return ""

    x1, y1, x2, y2 = content_bbox
    logger.info("  Content area: (%d,%d)-(%d,%d) = %dx%d", x1, y1, x2, y2, x2 - x1, y2 - y1)

    # 2. Fixed strip splitting
    max_px = get_max_px(model_name) if model_name else 1568
    strips = fixed_strip_split(image_path, content_bbox, max_px)
    logger.info("  %d strips (fixed %dpx height)", len(strips), max_px)

    # 3. Transcribe each strip
    strip_results = []
    for i, strip in enumerate(strips):
        logger.info("    Strip %d/%d...", i + 1, len(strips))
        try:
            result = strip_transcriber(block_image=dspy.Image(strip.crop_path))
            strip_results.append((strip, result.transcription))
            logger.info("      → %d chars", len(result.transcription))
        except Exception as e:
            logger.warning("      → ERROR: %s", e)

    if not strip_results:
        return ""

    # 4. Merge
    if len(strip_results) == 1:
        return strip_results[0][1].strip()

    labeled = []
    for i, (_, text) in enumerate(strip_results):
        labeled.append(f"---STRIP {i + 1}---\n{text.strip()}")
    strip_text = "\n\n".join(labeled)

    logger.info("  Merging %d strips...", len(strip_results))
    try:
        result = strip_merger(
            strip_transcriptions=strip_text,
            region_type="main_text",
        )
        merged = result.transcription.strip()
        logger.info("  Merged: %d chars", len(merged))
        return merged
    except Exception as e:
        logger.warning("  Merge ERROR: %s (concatenating)", e)
        return "\n".join(text.strip() for _, text in strip_results)


def transcribe_page_yolo_crop_strips(
    image_path: str,
    strip_transcriber,
    strip_merger,
    yolo_model,
    model_name: str | None = None,
    fallback_program=None,
    **kwargs,
) -> str:
    """YOLO-cropped strip pipeline: YOLO h-crop → strips → transcribe → merge.

    Same as strips_no_context but uses YOLO (conf=0.05) to crop left/right margins before
    strip splitting. Keeps full page height. Falls back to full image if
    YOLO detection covers less than 50% of the page.
    """
    from .layout import get_max_px

    # 1. YOLO horizontal crop
    content_bbox = auto_crop_margins(image_path, yolo_model=yolo_model)
    if content_bbox is None:
        logger.info("  No ink found — fallback")
        if fallback_program:
            safe_path = ensure_under_size_limit(image_path)
            return fallback_program(page_image=dspy.Image(safe_path)).transcription
        return ""

    x1, y1, x2, y2 = content_bbox
    logger.info("  Content area: (%d,%d)-(%d,%d) = %dx%d", x1, y1, x2, y2, x2 - x1, y2 - y1)

    # 2. Fixed strip splitting
    max_px = get_max_px(model_name) if model_name else 1568
    strips = fixed_strip_split(image_path, content_bbox, max_px)
    logger.info("  %d strips (fixed %dpx height)", len(strips), max_px)

    # 3. Transcribe each strip
    strip_results = []
    for i, strip in enumerate(strips):
        logger.info("    Strip %d/%d...", i + 1, len(strips))
        try:
            result = strip_transcriber(block_image=dspy.Image(strip.crop_path))
            strip_results.append((strip, result.transcription))
            logger.info("      → %d chars", len(result.transcription))
        except Exception as e:
            logger.warning("      → ERROR: %s", e)

    if not strip_results:
        return ""

    # 4. Merge
    if len(strip_results) == 1:
        return strip_results[0][1].strip()

    labeled = []
    for i, (_, text) in enumerate(strip_results):
        labeled.append(f"---STRIP {i + 1}---\n{text.strip()}")
    strip_text = "\n\n".join(labeled)

    logger.info("  Merging %d strips...", len(strip_results))
    try:
        result = strip_merger(
            strip_transcriptions=strip_text,
            region_type="main_text",
        )
        merged = result.transcription.strip()
        logger.info("  Merged: %d chars", len(merged))
        return merged
    except Exception as e:
        logger.warning("  Merge ERROR: %s (concatenating)", e)
        return "\n".join(text.strip() for _, text in strip_results)


def transcribe_page_yolo_crop_noise_warn(
    image_path: str,
    page_analyzer,
    strip_transcriber,
    strip_merger,
    yolo_model,
    model_name: str | None = None,
    fallback_program=None,
    **kwargs,
) -> str:
    """YOLO-cropped + noise warnings pipeline.

    Like yolo_crop_strips but adds noise analysis on the cropped image. The noise_warnings
    tell the merger what visual artifacts are NOT text (bleed-through, shadows,
    stains) to prevent hallucination.
    """
    from .layout import get_max_px

    # 1. YOLO horizontal crop
    content_bbox = auto_crop_margins(image_path, yolo_model=yolo_model)
    if content_bbox is None:
        logger.info("  No ink found — fallback")
        if fallback_program:
            safe_path = ensure_under_size_limit(image_path)
            return fallback_program(page_image=dspy.Image(safe_path)).transcription
        return ""

    x1, y1, x2, y2 = content_bbox
    logger.info("  Content area: (%d,%d)-(%d,%d) = %dx%d", x1, y1, x2, y2, x2 - x1, y2 - y1)

    # 2. Noise analysis on cropped image
    import tempfile
    from pathlib import Path
    from PIL import Image as PILImage

    img = PILImage.open(image_path)
    crop = img.crop(content_bbox)
    if crop.mode == "RGBA":
        crop = crop.convert("RGB")
    tmp = Path(tempfile.gettempdir()) / f"{Path(image_path).stem}_crop_noise_warn.jpg"
    crop.save(str(tmp), format="JPEG", quality=95)
    crop.close()
    img.close()
    safe_crop = ensure_under_size_limit(str(tmp))

    logger.info("  Noise analysis...")
    analysis = page_analyzer(page_image=dspy.Image(safe_crop))
    noise_warnings = analysis.noise_warnings
    logger.info("  Noise: %s", noise_warnings[:120])

    # 3. Fixed strip splitting
    max_px = get_max_px(model_name) if model_name else 1568
    strips = fixed_strip_split(image_path, content_bbox, max_px)
    logger.info("  %d strips (fixed %dpx height)", len(strips), max_px)

    # 4. Transcribe each strip
    strip_results = []
    for i, strip in enumerate(strips):
        logger.info("    Strip %d/%d...", i + 1, len(strips))
        try:
            result = strip_transcriber(block_image=dspy.Image(strip.crop_path))
            strip_results.append((strip, result.transcription))
            logger.info("      → %d chars", len(result.transcription))
        except Exception as e:
            logger.warning("      → ERROR: %s", e)

    if not strip_results:
        return ""

    # 5. Merge with noise warnings
    if len(strip_results) == 1:
        return strip_results[0][1].strip()

    labeled = []
    for i, (_, text) in enumerate(strip_results):
        labeled.append(f"---STRIP {i + 1}---\n{text.strip()}")
    strip_text = "\n\n".join(labeled)

    logger.info("  Merging %d strips...", len(strip_results))
    try:
        result = strip_merger(
            strip_transcriptions=strip_text,
            noise_warnings=noise_warnings,
        )
        merged = result.transcription.strip()
        logger.info("  Merged: %d chars", len(merged))
        return merged
    except Exception as e:
        logger.warning("  Merge ERROR: %s (concatenating)", e)
        return "\n".join(text.strip() for _, text in strip_results)


# ---------------------------------------------------------------------------
# YOLO Crop Pipeline (base for anti_delete, bare_merge, anti_halluc)
# ---------------------------------------------------------------------------


def transcribe_page_yolo_crop(
    image_path: str,
    strip_transcriber,
    strip_merger,
    yolo_model,
    model_name: str | None = None,
    fallback_program=None,
    **kwargs,
) -> str:
    """YOLO crop pipeline: YOLO crop → strips → transcribe → minimal merge.

    Same architecture as yolo_crop_strips/yolo_crop_noise_warn (YOLO horizontal crop + strips) but with a
    minimal merger that only deduplicates overlap — no PageAnalyzer, no noise
    analysis, no deletion commands. Merger prompt varies by variant.
    """
    from .layout import get_max_px

    # 1. YOLO horizontal crop
    content_bbox = auto_crop_margins(image_path, yolo_model=yolo_model)
    if content_bbox is None:
        logger.info("  No ink found — fallback")
        if fallback_program:
            safe_path = ensure_under_size_limit(image_path)
            return fallback_program(page_image=dspy.Image(safe_path)).transcription
        return ""

    x1, y1, x2, y2 = content_bbox
    logger.info("  Content area: (%d,%d)-(%d,%d) = %dx%d", x1, y1, x2, y2, x2 - x1, y2 - y1)

    # 2. Fixed strip splitting
    max_px = get_max_px(model_name) if model_name else 1568
    strips = fixed_strip_split(image_path, content_bbox, max_px)
    logger.info("  %d strips (fixed %dpx height)", len(strips), max_px)

    # 3. Transcribe each strip
    strip_results = []
    for i, strip in enumerate(strips):
        logger.info("    Strip %d/%d...", i + 1, len(strips))
        try:
            result = strip_transcriber(block_image=dspy.Image(strip.crop_path))
            strip_results.append((strip, result.transcription))
            logger.info("      → %d chars", len(result.transcription))
        except Exception as e:
            logger.warning("      → ERROR: %s", e)

    if not strip_results:
        return ""

    # 4. Merge
    if len(strip_results) == 1:
        return strip_results[0][1].strip()

    labeled = []
    for i, (_, text) in enumerate(strip_results):
        labeled.append(f"---STRIP {i + 1}---\n{text.strip()}")
    strip_text = "\n\n".join(labeled)

    logger.info("  Merging %d strips...", len(strip_results))
    try:
        result = strip_merger(strip_transcriptions=strip_text)
        merged = result.transcription.strip()
        logger.info("  Merged: %d chars", len(merged))
        return merged
    except Exception as e:
        logger.warning("  Merge ERROR: %s (concatenating)", e)
        return "\n".join(text.strip() for _, text in strip_results)


def transcribe_page_preprocess(
    image_path: str,
    strip_transcriber,
    strip_merger,
    yolo_model,
    preprocess_fn=None,
    model_name: str | None = None,
    fallback_program=None,
    **kwargs,
) -> str:
    """Preprocessing pipeline: YOLO crop → preprocess strips → transcribe → merge.

    Same as yolo_crop but applies preprocess_fn to each strip image before
    sending to the transcriber. Used for testing CLAHE, sharpening, denoising.
    """
    from .layout import get_max_px

    # 1. YOLO horizontal crop
    content_bbox = auto_crop_margins(image_path, yolo_model=yolo_model)
    if content_bbox is None:
        logger.info("  No ink found — fallback")
        if fallback_program:
            safe_path = ensure_under_size_limit(image_path)
            return fallback_program(page_image=dspy.Image(safe_path)).transcription
        return ""

    x1, y1, x2, y2 = content_bbox
    logger.info("  Content area: (%d,%d)-(%d,%d) = %dx%d", x1, y1, x2, y2, x2 - x1, y2 - y1)

    # 2. Fixed strip splitting
    max_px = get_max_px(model_name) if model_name else 1568
    strips = fixed_strip_split(image_path, content_bbox, max_px)
    logger.info("  %d strips (fixed %dpx height)", len(strips), max_px)

    # 3. Preprocess each strip image
    if preprocess_fn:
        for strip in strips:
            strip.crop_path = preprocess_fn(strip.crop_path)
        logger.info("  Preprocessed strips with %s", preprocess_fn.__name__)

    # 4. Transcribe each strip
    strip_results = []
    for i, strip in enumerate(strips):
        logger.info("    Strip %d/%d...", i + 1, len(strips))
        try:
            result = strip_transcriber(block_image=dspy.Image(strip.crop_path))
            strip_results.append((strip, result.transcription))
            logger.info("      → %d chars", len(result.transcription))
        except Exception as e:
            logger.warning("      → ERROR: %s", e)

    if not strip_results:
        return ""

    # 5. Merge
    if len(strip_results) == 1:
        return strip_results[0][1].strip()

    labeled = []
    for i, (_, text) in enumerate(strip_results):
        labeled.append(f"---STRIP {i + 1}---\n{text.strip()}")
    strip_text = "\n\n".join(labeled)

    logger.info("  Merging %d strips...", len(strip_results))
    try:
        result = strip_merger(strip_transcriptions=strip_text)
        merged = result.transcription.strip()
        logger.info("  Merged: %d chars", len(merged))
        return merged
    except Exception as e:
        logger.warning("  Merge ERROR: %s (concatenating)", e)
        return "\n".join(text.strip() for _, text in strip_results)


def transcribe_page_yolo_crop_anti_halluc(
    image_path: str,
    strip_transcriber,
    strip_merger,
    yolo_model,
    model_name: str | None = None,
    fallback_program=None,
    max_dist: int = 3,
    top_n: int = 3,
    compound_split: bool = True,
    abbrev_partial: bool = True,
    **kwargs,
) -> str:
    """Anti-hallucination pipeline: YOLO crop base + vocabulary-guided post-correction.

    Runs yolo_crop pipeline (YOLO crop → strips → transcribe → bare merge),
    then flags unknown words against combined vocab, finds candidates
    by edit distance + compound split + abbreviation expansion, and
    uses LLM to accept/reject corrections.
    """
    # Run yolo_crop pipeline for the base transcription
    merged = transcribe_page_yolo_crop(
        image_path=image_path,
        strip_transcriber=strip_transcriber,
        strip_merger=strip_merger,
        yolo_model=yolo_model,
        model_name=model_name,
        fallback_program=fallback_program,
    )

    if not merged:
        return merged

    # Vocab correction
    from .vocab_correction import (
        apply_corrections,
        build_candidates_table,
        format_correction_prompt,
        load_abbreviations,
        load_combined_vocab,
        parse_corrections,
    )

    vocab = load_combined_vocab()
    abbrevs = load_abbreviations()

    flagged = build_candidates_table(
        merged, vocab, abbrevs,
        max_dist=max_dist,
        top_n=top_n,
        compound_split=compound_split,
        abbrev_partial=abbrev_partial,
    )

    if not flagged:
        logger.info("  Vocab correction: no unknown words")
        return merged

    correctable = [f for f in flagged if f["edit_candidates"] or f.get("compound") or f.get("abbrev")]
    logger.info("  Vocab correction: %d unknown, %d with candidates", len(flagged), len(correctable))

    if not correctable:
        return merged

    # LLM correction
    prompt = format_correction_prompt(merged, flagged)
    try:
        result = dspy.ChainOfThought("prompt -> corrections")(prompt=prompt)
        corrections = parse_corrections(result.corrections, flagged)
        if corrections:
            corrected = apply_corrections(merged, flagged, corrections)
            logger.info("  Applied %d corrections", len(corrections))
            return corrected
    except Exception as e:
        logger.warning("  Vocab correction LLM ERROR: %s", e)

    return merged


# ---------------------------------------------------------------------------
# Strips + Split Context Pipeline
# ---------------------------------------------------------------------------


def _merge_strips(strip_results, strip_merger, structural_context):
    """Merge strip transcriptions using the LLM merger."""
    labeled = []
    for i, (_, text) in enumerate(strip_results):
        labeled.append(f"---STRIP {i + 1}---\n{text.strip()}")
    strip_text = "\n\n".join(labeled)

    result = strip_merger(
        strip_transcriptions=strip_text,
        structural_context=structural_context,
    )
    return result.transcription.strip()


def transcribe_page_strips_split_context(
    image_path: str,
    page_analyzer,
    strip_transcriber,
    strip_merger,
    model_name: str | None = None,
    fallback_program=None,
    **kwargs,
) -> str:
    """Split-context pipeline: analyze → crop → strips → transcribe → merge.

    Like strips_page_desc but with split context: SplitContextAnalyzer produces two focused outputs:
      - reading_context  → strip transcriber
      - structural_context → strip merger
    """
    from .layout import get_max_px

    # 1. Page analysis — two separate context fields
    safe_path = ensure_under_size_limit(image_path)
    logger.info("  Page analysis (split context)...")
    analysis = page_analyzer(page_image=dspy.Image(safe_path))
    reading_context = analysis.reading_context
    structural_context = analysis.structural_context
    logger.info("  Reading ctx: %s", reading_context[:120])
    logger.info("  Structural ctx: %s", structural_context[:120])

    # 2. Auto-crop margins
    content_bbox = auto_crop_margins(image_path)
    if content_bbox is None:
        logger.info("  No ink found — fallback")
        if fallback_program:
            return fallback_program(page_image=dspy.Image(safe_path)).transcription
        return ""

    x1, y1, x2, y2 = content_bbox
    logger.info("  Content area: (%d,%d)-(%d,%d) = %dx%d", x1, y1, x2, y2, x2-x1, y2-y1)

    # 3. Fixed strip splitting
    max_px = get_max_px(model_name) if model_name else 1568
    strips = fixed_strip_split(image_path, content_bbox, max_px)
    logger.info("  %d strips (fixed %dpx height)", len(strips), max_px)

    # 4. Transcribe each strip with reading_context
    strip_results = []
    for i, strip in enumerate(strips):
        logger.info("    Strip %d/%d...", i + 1, len(strips))
        try:
            result = strip_transcriber(
                block_image=dspy.Image(strip.crop_path),
                reading_context=reading_context,
            )
            strip_results.append((strip, result.transcription))
            logger.info("      → %d chars", len(result.transcription))
        except Exception as e:
            logger.warning("      → ERROR: %s", e)

    if not strip_results:
        return ""

    # 5. Single strip — no merging needed
    if len(strip_results) == 1:
        merged = strip_results[0][1].strip()
    else:
        # 6. Merge with structural_context
        logger.info("  Merging %d strips...", len(strip_results))
        try:
            merged = _merge_strips(strip_results, strip_merger, structural_context)
            logger.info("  Merged: %d chars", len(merged))
        except Exception as e:
            logger.warning("  Merge ERROR: %s (concatenating)", e)
            return "\n".join(text.strip() for _, text in strip_results)

    return merged


def _merge_tiles(tile_results, tile_merger):
    """Merge tile transcriptions using the LLM merger (no context)."""
    labeled = []
    for i, (_, text) in enumerate(tile_results):
        labeled.append(f"---STRIP {i + 1}---\n{text.strip()}")
    strip_text = "\n\n".join(labeled)

    result = tile_merger(
        strip_transcriptions=strip_text,
        region_type="main_text",
    )
    return result.transcription.strip()


def transcribe_page_tiles_2d(
    image_path: str,
    tile_transcriber,
    tile_merger,
    model_name: str | None = None,
    fallback_program=None,
    **kwargs,
) -> str:
    """Full-resolution 2D tiling pipeline.

    Splits into a 2D grid of tiles (≤max_px × max_px) so the model sees
    every tile at native resolution. Two-pass merge: first horizontal
    (within each row), then vertical (across rows). No page analysis —
    transcriber and merger work from the image alone.
    """
    from .layout import get_max_px

    # 1. Auto-crop margins
    content_bbox = auto_crop_margins(image_path)
    if content_bbox is None:
        logger.info("  No ink found — fallback")
        if fallback_program:
            safe_path = ensure_under_size_limit(image_path)
            return fallback_program(page_image=dspy.Image(safe_path)).transcription
        return ""

    x1, y1, x2, y2 = content_bbox
    logger.info("  Content area: (%d,%d)-(%d,%d) = %dx%d", x1, y1, x2, y2, x2 - x1, y2 - y1)

    # 2. 2D tile grid
    max_px = get_max_px(model_name) if model_name else 1568
    grid = tile_grid_split(image_path, content_bbox, max_px)
    if not grid:
        logger.info("  No tiles — fallback")
        if fallback_program:
            safe_path = ensure_under_size_limit(image_path)
            return fallback_program(page_image=dspy.Image(safe_path)).transcription
        return ""
    n_rows = len(grid)
    n_cols = max(len(row) for row in grid)
    total = sum(len(row) for row in grid)
    logger.info("  %d tiles (%d rows × %d cols, %dpx max)", total, n_rows, n_cols, max_px)

    # 3. Transcribe all tiles
    tile_results = []  # list of rows, each row is list of (Region, text)
    tile_num = 0
    for ri, row in enumerate(grid):
        row_results = []
        for ci, tile in enumerate(row):
            tile_num += 1
            logger.info("    Tile %d/%d (R%dC%d)...", tile_num, total, ri + 1, ci + 1)
            try:
                result = tile_transcriber(
                    block_image=dspy.Image(tile.crop_path),
                )
                row_results.append((tile, result.transcription))
                logger.info("      → %d chars", len(result.transcription))
            except Exception as e:
                logger.warning("      → ERROR: %s", e)
        tile_results.append(row_results)

    # 4. Two-pass merge
    # Pass 1: merge tiles within each row (horizontal dedup)
    row_texts = []
    for ri, row_results in enumerate(tile_results):
        if not row_results:
            continue
        if len(row_results) == 1:
            row_texts.append(row_results[0][1].strip())
        else:
            logger.info("  Row %d merge (%d tiles)...", ri + 1, len(row_results))
            try:
                merged_row = _merge_tiles(row_results, tile_merger)
                row_texts.append(merged_row)
                logger.info("    → %d chars", len(merged_row))
            except Exception as e:
                logger.warning("    Row %d merge ERROR: %s (concatenating)", ri + 1, e)
                row_texts.append(" ".join(text.strip() for _, text in row_results))

    if not row_texts:
        return ""

    # Pass 2: merge rows (vertical dedup)
    if len(row_texts) == 1:
        merged = row_texts[0]
    else:
        logger.info("  Column merge (%d rows)...", len(row_texts))
        row_as_strips = [(None, text) for text in row_texts]
        try:
            merged = _merge_tiles(row_as_strips, tile_merger)
            logger.info("  Merged: %d chars", len(merged))
        except Exception as e:
            logger.warning("  Column merge ERROR: %s (concatenating)", e)
            merged = "\n".join(row_texts)

    return merged
