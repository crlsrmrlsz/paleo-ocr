"""Reading-order-independent metrics: BOC and delta_CER.

BOC (Bag of Characters) ignores character order entirely — it measures
pure recognition quality. delta_CER = CER - BOC quantifies how much
error comes from reading order vs. character recognition.
"""

from collections import Counter


def compute_boc(reference: str, hypothesis: str) -> float:
    """Bag of Characters distance = symmetric_diff / max(len_ref, len_hyp).

    Returns 0.0 for identical character distributions, 1.0 for completely
    disjoint. Bounded [0, 1].
    """
    if not reference and not hypothesis:
        return 0.0
    max_len = max(len(reference), len(hypothesis))
    if max_len == 0:
        return 0.0

    ref_counts = Counter(reference)
    hyp_counts = Counter(hypothesis)

    # Symmetric difference: characters in one but not the other
    all_chars = set(ref_counts) | set(hyp_counts)
    diff = sum(abs(ref_counts.get(c, 0) - hyp_counts.get(c, 0)) for c in all_chars)

    return diff / max_len


def compute_order_independent(reference: str, hypothesis: str, cer: float) -> dict:
    """Compute BOC and delta_CER.

    Args:
        reference: Reference text
        hypothesis: Hypothesis text
        cer: Pre-computed CER value (to avoid recomputing)

    Returns:
        dict with boc and delta_cer values
    """
    boc = compute_boc(reference, hypothesis)
    delta_cer = cer - boc  # positive = reading order contributes to error

    return {
        "boc": round(boc, 6),
        "delta_cer": round(delta_cer, 6),
    }
