"""Romein 4-tier diagnostic CER (cumulative normalization).

Each tier applies increasingly aggressive normalization to BOTH reference
and hypothesis before computing CER. The drop between tiers isolates
specific error categories:

  T1 raw      → Full accuracy (NFC + basic whitespace cleanup)
  T2 nospace  → + remove all Unicode separators → isolates word segmentation
  T3 lowercase→ + casefold → isolates case confusion
  T4 alnum    → + keep only Letter/Number/Mark categories → core content
"""

import re
import unicodedata

from .core import compute_cer


def normalize_t1(text: str) -> str:
    """T1 raw: NFC + soft-hyphen removal + tab→space + dedup spaces."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00ad", "")  # soft hyphen
    text = text.replace("\t", " ")
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def normalize_t2(text: str) -> str:
    """T2 nospace: T1 + remove all Unicode category Z (separators)."""
    text = normalize_t1(text)
    # Remove all characters in Unicode category Z (Zs, Zl, Zp) plus newlines
    text = re.sub(r"[\s]", "", text)
    return text


def normalize_t3(text: str) -> str:
    """T3 lowercase: T2 + casefold."""
    return normalize_t2(text).casefold()


def normalize_t4(text: str) -> str:
    """T4 alnum: T3 + keep only Unicode categories L, N, M (letters, numbers, marks)."""
    text = normalize_t3(text)
    return "".join(c for c in text if unicodedata.category(c)[0] in ("L", "N", "M"))


TIER_NORMALIZERS = {
    "T1_raw": normalize_t1,
    "T2_nospace": normalize_t2,
    "T3_lowercase": normalize_t3,
    "T4_alnum": normalize_t4,
}


def compute_romein_tiers(reference: str, hypothesis: str) -> dict:
    """Compute CER at all 4 Romein tiers.

    All tiers are normalized by the T1 reference length (not each tier's own
    reference length). This ensures monotonicity: T1 >= T2 >= T3 >= T4,
    because each successive normalization can only eliminate error sources,
    never introduce them. Using a fixed denominator prevents the CER from
    increasing when space-removal shrinks the reference.

    Returns dict with keys T1_raw, T2_nospace, T3_lowercase, T4_alnum.
    """
    import editdistance

    ref_t1 = normalize_t1(reference)
    t1_len = len(ref_t1)
    if t1_len == 0:
        # Empty reference: 1.0 if any hypothesis, else 0.0
        has_any = any(len(normalizer(hypothesis)) > 0 for normalizer in TIER_NORMALIZERS.values())
        return {tier: (1.0 if has_any else 0.0) for tier in TIER_NORMALIZERS}

    results = {}
    for tier_name, normalizer in TIER_NORMALIZERS.items():
        ref_norm = normalizer(reference)
        hyp_norm = normalizer(hypothesis)
        dist = editdistance.eval(ref_norm, hyp_norm)
        results[tier_name] = round(dist / t1_len, 6)
    return results
