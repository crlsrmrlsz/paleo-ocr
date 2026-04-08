"""DSPy pipeline configuration."""

from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")
DATA_DIR = PROJECT_ROOT / "data"
CORPORA_DIR = DATA_DIR / "corpora"
OPTIMIZATION_DIR = DATA_DIR / "optimization"
SUBSETS_DIR = DATA_DIR / "subsets"
RESULTS_DIR = DATA_DIR / "results"

DEFAULT_MODEL = "openrouter/anthropic/claude-opus-4.6"
DEFAULT_MAX_TOKENS = 8192

MAX_BOOTSTRAPPED_DEMOS = 3
MAX_LABELED_DEMOS = 3

# Layout detection (AABB, not OBB — biglam model is axis-aligned)
YOLO_REPO = "biglam/medieval-manuscript-yolov11"
YOLO_WEIGHTS = "medieval-yolov11l.pt"
YOLO_CONFIDENCE = 0.25
YOLO_PADDING = 0.05  # 5% padding on crops
YOLO_MIN_AREA_RATIO = 0.01  # skip regions < 1% of page area

# SegmOnto class → pipeline region type
REGION_MAP = {
    "MainZone": "main_text",
    "MarginTextZone": "margin",
    "GraphicZone": "graphic",
    "StampZone": "stamp",
}

# Anthropic 5MB base64 image limit
MAX_B64_BYTES = 5_000_000

# Model vision resolution limits (longest side in px)
# Source: docs/research/image_processing_reference.md
MODEL_MAX_PX = {
    "openrouter/anthropic/claude-opus-4.6": 1568,
    "openrouter/openai/gpt-5.4": 6000,       # with detail: "original"
    "openrouter/google/gemini-3.1-pro": 3072,  # with media_resolution: ultra_high
    "openrouter/mistralai/mistral-large-3": 1568,  # conservative estimate
}
DEFAULT_MAX_PX = 1568  # conservative fallback

# Strip splitting for large regions
STRIP_OVERLAP_PX = 200  # overlap between adjacent strips to avoid cutting mid-line

# RLM post-processor
RLM_MAX_ITERATIONS = 10
RLM_MAX_LLM_CALLS = 20
