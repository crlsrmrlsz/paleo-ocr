"""Image utilities for DSPy pipeline — JPEG recompression and preprocessing."""

import base64
import tempfile
from pathlib import Path

import numpy as np

from .config import MAX_B64_BYTES


def ensure_under_size_limit(
    image_path: str, max_b64_bytes: int = MAX_B64_BYTES
) -> str:
    """Return a file path whose base64 encoding fits within the size limit.

    If the original image is within limit, returns the same path.
    If over limit, saves a recompressed JPEG to a temp file and returns that path.
    """
    img_bytes = Path(image_path).read_bytes()
    b64_size = len(base64.standard_b64encode(img_bytes))

    if b64_size <= max_b64_bytes:
        return image_path

    from io import BytesIO

    from PIL import Image

    img = Image.open(BytesIO(img_bytes))
    if img.mode == "RGBA":
        img = img.convert("RGB")

    for quality in (85, 75, 65, 50):
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        b64_size = len(base64.standard_b64encode(buf.getvalue()))
        if b64_size <= max_b64_bytes:
            # Save to temp file alongside the original
            suffix = f"_q{quality}.jpg"
            tmp_path = Path(tempfile.gettempdir()) / (Path(image_path).stem + suffix)
            tmp_path.write_bytes(buf.getvalue())
            return str(tmp_path)

    # Last resort: save at quality 50 even if still over limit
    tmp_path = Path(tempfile.gettempdir()) / (Path(image_path).stem + "_q50.jpg")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=50)
    tmp_path.write_bytes(buf.getvalue())
    return str(tmp_path)


# ---------------------------------------------------------------------------
# Image preprocessing for strip enhancement
# ---------------------------------------------------------------------------


def preprocess_clahe(image_path: str, clip_limit: float = 2.0, tile_size: int = 8) -> str:
    """Apply CLAHE (Contrast-Limited Adaptive Histogram Equalization).

    Enhances local contrast — makes faint ink more visible without
    destroying color information. Applied per-channel on color images.
    """
    import cv2

    img = cv2.imread(image_path)
    if img is None:
        return image_path

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))

    if len(img.shape) == 2:
        # Grayscale
        result = clahe.apply(img)
    else:
        # Color: apply CLAHE to L channel in LAB space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    tmp = Path(tempfile.gettempdir()) / (Path(image_path).stem + "_clahe.jpg")
    cv2.imwrite(str(tmp), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return str(tmp)


def preprocess_sharpen(image_path: str, radius: int = 2, percent: int = 150, threshold: int = 3) -> str:
    """Apply unsharp mask sharpening.

    Enhances edge definition of letterforms. Counteracts the blur
    introduced by Claude's internal downscaling.
    """
    from PIL import Image, ImageFilter

    img = Image.open(image_path)
    if img.mode == "RGBA":
        img = img.convert("RGB")

    sharpened = img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))

    tmp = Path(tempfile.gettempdir()) / (Path(image_path).stem + "_sharp.jpg")
    sharpened.save(str(tmp), format="JPEG", quality=95)
    sharpened.close()
    img.close()
    return str(tmp)


def preprocess_denoise(image_path: str, d: int = 9, sigma_color: int = 75, sigma_space: int = 75) -> str:
    """Apply bilateral filtering for edge-preserving denoising.

    Removes noise (stains, foxing) while preserving ink edges.
    """
    import cv2

    img = cv2.imread(image_path)
    if img is None:
        return image_path

    result = cv2.bilateralFilter(img, d, sigma_color, sigma_space)

    tmp = Path(tempfile.gettempdir()) / (Path(image_path).stem + "_denoise.jpg")
    cv2.imwrite(str(tmp), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return str(tmp)
