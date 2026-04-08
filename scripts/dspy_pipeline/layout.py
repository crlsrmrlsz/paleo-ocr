"""YOLO-based layout detection for manuscript page segmentation.

Uses biglam/medieval-manuscript-yolov11 (AABB, not OBB) pretrained on
CATMuS Medieval Segmentation dataset (8th-16th century manuscripts).
"""

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from .config import (
    DEFAULT_MAX_PX,
    MODEL_MAX_PX,
    REGION_MAP,
    STRIP_OVERLAP_PX,
    YOLO_CONFIDENCE,
    YOLO_MIN_AREA_RATIO,
    YOLO_PADDING,
    YOLO_REPO,
    YOLO_WEIGHTS,
)
from .image_utils import ensure_under_size_limit


@dataclass
class Region:
    """A detected layout region with bounding box and type."""

    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixel coords
    region_type: str  # "main_text", "margin", "graphic", "stamp"
    confidence: float
    crop_path: str | None = field(default=None, repr=False)


def load_yolo_model():
    """Download weights from HuggingFace Hub and return YOLO model."""
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO

    weights_path = hf_hub_download(YOLO_REPO, YOLO_WEIGHTS)
    return YOLO(weights_path)


def detect_regions(
    model,
    image_path: str,
    confidence: float = YOLO_CONFIDENCE,
    min_area_ratio: float = YOLO_MIN_AREA_RATIO,
) -> list[Region]:
    """Run YOLO on an image, filter to relevant region types.

    Returns list of Region objects for types in REGION_MAP,
    skipping regions smaller than min_area_ratio of the page area.
    """
    results = model(image_path, conf=confidence, verbose=False)

    if not results or len(results) == 0:
        return []

    result = results[0]
    if result.boxes is None or len(result.boxes.xyxy) == 0:
        return []

    # Get image dimensions for area filtering
    img = Image.open(image_path)
    page_area = img.width * img.height
    min_area = page_area * min_area_ratio
    img.close()

    # Extract detections
    xyxy = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)

    regions = []
    for i in range(len(xyxy)):
        class_name = model.names.get(int(cls_ids[i]), "")
        if class_name not in REGION_MAP:
            continue

        x1, y1, x2, y2 = int(xyxy[i][0]), int(xyxy[i][1]), int(xyxy[i][2]), int(xyxy[i][3])
        region_area = (x2 - x1) * (y2 - y1)
        if region_area < min_area:
            continue

        regions.append(Region(
            bbox=(x1, y1, x2, y2),
            region_type=REGION_MAP[class_name],
            confidence=float(conf[i]),
        ))

    return regions


def _overlap_fraction(a: tuple, b: tuple) -> float:
    """Fraction of region a's area that overlaps with region b."""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x1 >= x2 or y1 >= y2:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    return inter / area_a if area_a > 0 else 0.0


def _merge_bboxes(a: tuple, b: tuple) -> tuple:
    """Union bounding box of two regions."""
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def resolve_overlaps(regions: list[Region]) -> list[Region]:
    """Remove or merge overlapping regions to avoid text duplication/loss.

    Rules (applied pairwise, highest confidence first):
    1. Same type, >30% overlap of smaller → merge into union bbox
    2. Different types, smaller is >70% inside larger → drop smaller
    3. Different types, 30-70% overlap → trim smaller to non-overlapping area
    4. After resolution, drop regions with <15% confidence that are >80% covered
    """
    if len(regions) <= 1:
        return regions

    # Sort by confidence descending — process highest confidence first
    regions = sorted(regions, key=lambda r: -r.confidence)
    keep = list(regions)

    # Pass 1: merge same-type overlaps, drop contained different-type
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(keep):
            j = i + 1
            while j < len(keep):
                ri, rj = keep[i], keep[j]
                oi = _overlap_fraction(ri.bbox, rj.bbox)  # fraction of i inside j
                oj = _overlap_fraction(rj.bbox, ri.bbox)  # fraction of j inside i
                max_overlap = max(oi, oj)

                if max_overlap < 0.30:
                    j += 1
                    continue

                if ri.region_type == rj.region_type:
                    # Same type — merge
                    merged = Region(
                        bbox=_merge_bboxes(ri.bbox, rj.bbox),
                        region_type=ri.region_type,
                        confidence=max(ri.confidence, rj.confidence),
                    )
                    keep[i] = merged
                    keep.pop(j)
                    changed = True
                else:
                    # Different types — drop the one more contained
                    # But never drop main_text in favor of graphic/stamp
                    # (YOLO sometimes misclassifies text areas as graphic)
                    conf_ratio = min(ri.confidence, rj.confidence) / max(ri.confidence, rj.confidence)
                    threshold = 0.50 if conf_ratio < 0.3 else 0.70

                    ri_is_text = ri.region_type == "main_text"
                    rj_is_text = rj.region_type == "main_text"

                    if oj > threshold and not rj_is_text:
                        # j is mostly inside i and j is not main_text → drop j
                        keep.pop(j)
                        changed = True
                    elif oi > threshold and not ri_is_text:
                        # i is mostly inside j and i is not main_text → drop i
                        keep.pop(i)
                        changed = True
                        break
                    elif oj > threshold and rj_is_text and not ri_is_text:
                        # j is main_text inside a non-text i → drop i instead
                        keep.pop(i)
                        changed = True
                        break
                    elif oi > threshold and ri_is_text and not rj_is_text:
                        # i is main_text inside a non-text j → drop j instead
                        keep.pop(j)
                        changed = True
                    else:
                        j += 1
            i += 1

    # Pass 2: drop low-confidence regions mostly covered by others
    final = []
    for i, ri in enumerate(keep):
        if ri.confidence < 0.15:
            # Check how much of this region is covered by all others combined
            others = [rj for j, rj in enumerate(keep) if j != i]
            covered = max((_overlap_fraction(ri.bbox, rj.bbox) for rj in others), default=0)
            if covered > 0.80:
                continue  # drop — almost entirely covered
        final.append(ri)

    return final


def sort_reading_order(regions: list[Region], image_height: int) -> list[Region]:
    """Sort regions top-to-bottom, left-to-right for ties.

    Y tolerance is 2% of image height to group regions at similar vertical
    positions (e.g., a margin note beside main text).
    """
    y_tolerance = int(image_height * 0.02)

    def sort_key(region: Region):
        x1, y1, x2, y2 = region.bbox
        cy = (y1 + y2) // 2
        cx = (x1 + x2) // 2
        # Quantize Y to tolerance bands for grouping
        y_band = cy // y_tolerance if y_tolerance > 0 else cy
        return (y_band, cx)

    return sorted(regions, key=sort_key)


def crop_region(
    image_path: str,
    region: Region,
    padding: float = YOLO_PADDING,
) -> str:
    """Crop a region from the full-resolution image with padding.

    Saves crop as JPEG to a temp file and applies size limit check.
    Sets region.crop_path and returns the path.
    """
    img = Image.open(image_path)
    x1, y1, x2, y2 = region.bbox
    w, h = x2 - x1, y2 - y1

    # Add padding, clamped to image bounds
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(img.width, x2 + pad_x)
    y2 = min(img.height, y2 + pad_y)

    crop = img.crop((x1, y1, x2, y2))
    img.close()

    # Save to temp file
    stem = Path(image_path).stem
    suffix = f"_{x1}_{y1}_{x2}_{y2}.jpg"
    tmp_path = Path(tempfile.gettempdir()) / (stem + suffix)

    if crop.mode == "RGBA":
        crop = crop.convert("RGB")
    crop.save(str(tmp_path), format="JPEG", quality=95)
    crop.close()

    # Apply 5MB limit if needed
    final_path = ensure_under_size_limit(str(tmp_path))
    region.crop_path = final_path
    return final_path


def get_max_px(model_name: str) -> int:
    """Get the vision resolution limit for a model."""
    return MODEL_MAX_PX.get(model_name, DEFAULT_MAX_PX)


def compute_scale(width: int, height: int, max_px: int) -> float:
    """Compute the downscale factor a model's vision encoder will apply.

    Models scale images so the longest side fits within max_px.
    Returns 1.0 (no downscaling) if both dimensions ≤ max_px.
    """
    longest = max(width, height)
    if longest <= max_px:
        return 1.0
    return max_px / longest


def optimal_strip_height(crop_w: int, crop_h: int, max_px: int) -> int | None:
    """Calculate the optimal strip height to maximize effective resolution.

    Resolution math:
    - The model scales images so longest_side = max_px.
    - Scale factor = max_px / max(width, height).
    - For a strip of (crop_w × strip_h):
      - If strip_h ≤ crop_w: width is longest side → scale = max_px / crop_w
      - If strip_h > crop_w: height is longest side → scale = max_px / strip_h (worse)
    - To maximize scale: keep strip_h ≤ min(max_px, crop_w).

    Cases:
    - Both ≤ max_px: no subdivision needed → returns None
    - Width ≤ max_px, height > max_px: strip_h = max_px → scale = 1.0 (zero downscaling)
    - Width > max_px, height > max_px: strip_h = max_px → scale = max_px / crop_w (best achievable)
    - Width > max_px, height ≤ max_px: no subdivision helps → returns None

    Returns None if no subdivision needed, or the optimal strip height in pixels.
    """
    if max(crop_w, crop_h) <= max_px:
        return None  # Already fits — no downscaling

    if crop_h <= max_px:
        # Height fits, width is the problem. Subdividing height doesn't help
        # — scale is locked at max_px / crop_w regardless of strip height.
        return None

    # Height exceeds max_px — subdivide into strips.
    # strip_h = max_px ensures:
    #   - If crop_w ≤ max_px: max(crop_w, max_px) = max_px → scale = 1.0 (perfect)
    #   - If crop_w > max_px: max(crop_w, max_px) = crop_w → scale = max_px / crop_w (best achievable)
    return max_px


def subdivide_region(
    image_path: str,
    region: Region,
    max_px: int,
    overlap: int = STRIP_OVERLAP_PX,
    padding: float = YOLO_PADDING,
) -> list[Region]:
    """Split a region into horizontal strips sized for the model's resolution limit.

    Uses optimal_strip_height() to determine whether subdivision is needed
    and what strip height maximizes effective resolution per character.
    Adjacent strips overlap by `overlap` pixels to avoid cutting mid-line.
    """
    x1, y1, x2, y2 = region.bbox

    # Apply padding (same logic as crop_region)
    img = Image.open(image_path)
    w, h = x2 - x1, y2 - y1
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    px1 = max(0, x1 - pad_x)
    py1 = max(0, y1 - pad_y)
    px2 = min(img.width, x2 + pad_x)
    py2 = min(img.height, y2 + pad_y)

    crop_w = px2 - px1
    crop_h = py2 - py1

    strip_h = optimal_strip_height(crop_w, crop_h, max_px)

    if strip_h is None:
        # No subdivision needed — single crop at current dimensions
        img.close()
        crop_region(image_path, region, padding)
        return [region]

    # Calculate strip positions with overlap
    step = strip_h - overlap
    if step <= 0:
        step = strip_h

    strips = []
    strip_y = py1
    strip_idx = 0
    while strip_y < py2:
        strip_y2 = min(strip_y + strip_h, py2)

        # Absorb tiny trailing strips (< 20% of target) into the last strip
        if strip_y2 - strip_y < strip_h * 0.2 and strips:
            last = strips[-1]
            lx1, ly1, lx2, _ = last.bbox
            strips[-1] = Region(
                bbox=(lx1, ly1, lx2, py2),
                region_type=region.region_type,
                confidence=region.confidence,
            )
            # Re-crop the extended last strip
            _save_strip_crop(img, strips[-1], image_path, len(strips) - 1)
            break

        strip_region = Region(
            bbox=(px1, strip_y, px2, strip_y2),
            region_type=region.region_type,
            confidence=region.confidence,
        )
        _save_strip_crop(img, strip_region, image_path, strip_idx)
        strips.append(strip_region)

        strip_y += step
        strip_idx += 1

    img.close()
    return strips


def _save_strip_crop(img: Image.Image, region: Region, image_path: str, idx: int):
    """Crop a strip from the open image and save to temp file."""
    px1, py1, px2, py2 = region.bbox
    crop = img.crop((px1, py1, px2, py2))

    stem = Path(image_path).stem
    suffix = f"_strip{idx}_{px1}_{py1}_{px2}_{py2}.jpg"
    tmp_path = Path(tempfile.gettempdir()) / (stem + suffix)

    if crop.mode == "RGBA":
        crop = crop.convert("RGB")
    crop.save(str(tmp_path), format="JPEG", quality=95)
    crop.close()

    region.crop_path = ensure_under_size_limit(str(tmp_path))


def detect_and_crop(
    model, image_path: str, model_name: str | None = None,
) -> list[Region]:
    """Full layout pipeline: detect regions, filter, sort, crop/subdivide.

    For regions taller than the model's vision resolution limit, splits
    into horizontal strips to maximize effective resolution per character.
    Logs resolution metrics for each region.

    Returns sorted list of Region objects with crop_path set.
    Returns empty list if no relevant regions detected (caller should fallback).
    """
    import logging
    logger = logging.getLogger(__name__)

    regions = detect_regions(model, image_path)
    if not regions:
        return []

    # Resolve overlapping regions before processing
    regions = resolve_overlaps(regions)
    if not regions:
        return []

    img = Image.open(image_path)
    image_height = img.height
    img.close()

    regions = sort_reading_order(regions, image_height)

    max_px = get_max_px(model_name) if model_name else DEFAULT_MAX_PX

    # Subdivide large regions and crop small ones
    final_regions = []
    for region in regions:
        strips = subdivide_region(image_path, region, max_px)

        if len(strips) == 1:
            s = strips[0]
            sw, sh = s.bbox[2] - s.bbox[0], s.bbox[3] - s.bbox[1]
            scale = compute_scale(sw, sh, max_px)
            logger.info(
                "    %s %dx%d scale=%.2fx (no split needed)",
                region.region_type, sw, sh, scale,
            )
        else:
            # Compare: full padded region vs first strip (same width, less height)
            # Full padded region = union of all strip bboxes
            full_w = strips[0].bbox[2] - strips[0].bbox[0]
            full_h = strips[-1].bbox[3] - strips[0].bbox[1]
            full_scale = compute_scale(full_w, full_h, max_px)

            s0 = strips[0]
            sw, sh = s0.bbox[2] - s0.bbox[0], s0.bbox[3] - s0.bbox[1]
            strip_scale = compute_scale(sw, sh, max_px)
            logger.info(
                "    %s %dx%d → %d strips of ~%dx%d, scale %.2fx → %.2fx (+%.0f%%)",
                region.region_type, full_w, full_h, len(strips), sw, sh,
                full_scale, strip_scale,
                (strip_scale / full_scale - 1) * 100 if full_scale > 0 else 0,
            )

        final_regions.extend(strips)

    return final_regions
