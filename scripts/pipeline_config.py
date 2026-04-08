"""Shared pipeline configuration.

Canonical definitions of project-wide constants and utilities used
across preparation, evaluation, review, and transcription scripts.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CORPORA_DIR = DATA_DIR / "corpora"
SUBSETS_DIR = DATA_DIR / "subsets"

# Valid subset/edition combinations
VALID_COMBOS = {
    "codea": ("paleographic", "critical"),
    "toledo": ("editorial",),
}

# Pages excluded from evaluation (kept in dataset but not counted in metrics)
EXCLUDED_PAGES = {
    # CODEA-3623: ink from reverse page too visible, makes transcription unreliable
    "codea_CODEA-3623_1r",
    "codea_CODEA-3623_3r",
    "codea_CODEA-3623_6r",
    # CODEA-2027_1v: GT only covers first paragraph, incomplete reference
    "codea_CODEA-2027_1v",
}

# Romein 4-tier normalization labels
ROMEIN_TIERS = ("T1_raw", "T2_nospace", "T3_lowercase", "T4_alnum")

# Metric tier groups for --skip-semantic flag
METRIC_TIERS = {
    "standard": ["cer", "cer_n", "nls", "wer", "ser"],
    "diagnostic": ["romein", "decomposition"],
    "order_independent": ["boc", "delta_cer"],
    "semantic": ["semantic_similarity", "ner"],
}

# NER fuzzy matching threshold (NLS > this value counts as a match)
NER_MATCH_THRESHOLD = 0.7

# Bootstrap resamples for confidence intervals
BOOTSTRAP_N_RESAMPLES = 10_000

# Sentence-transformer model for semantic similarity
SEMANTIC_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# spaCy model for NER
SPACY_MODEL_NAME = "es_core_news_lg"


def load_manifest(subset: str = "all") -> list[dict]:
    """Load evaluation manifest(s).

    Args:
        subset: "codea", "toledo", or "all" (concatenates both).
    """
    if subset == "all":
        datasets = ["codea", "toledo"]
    else:
        datasets = [subset]

    entries = []
    for ds in datasets:
        manifest_path = SUBSETS_DIR / ds / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                entries.extend(json.load(f))

    return entries
