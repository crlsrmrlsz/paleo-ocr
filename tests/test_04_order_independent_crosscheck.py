"""Cross-check BOC and delta_CER against manual Counter computation.

Strategy:
  - Manual Counter-based BOC computation vs our compute_boc()
  - Order invariance property
  - Document known BOC bound bug (max=2.0, not 1.0)
  - delta_CER = CER - BOC consistency
"""

from collections import Counter

import pytest

from metrics.core import compute_cer
from metrics.order_independent import compute_boc, compute_order_independent


PAIRS = [
    ("identical", "abc", "abc"),
    ("empty_both", "", ""),
    ("one_extra", "abc", "abcd"),
    ("freq_diff", "aab", "abb"),
    ("reversed", "abc", "cba"),
    ("completely_different", "abc", "xyz"),
    ("repeated_different", "aaaa", "bbbb"),
    ("single_char", "a", "b"),
    ("long_text",
     "Lorem ipsum dolor sit amet consetetur sadipscing elitr",
     "Lorem Ipsum dolor sit armet consetetur sadipscing elitr"),
    ("empty_ref", "", "hello"),
    ("empty_hyp", "hello", ""),
    ("unicode", "café résumé", "cafe resume"),
    ("spanish", "Don Juan, anno 1542", "don jvan. anno 1542"),
    ("pure_insertion", "abc", "aXbYcZ"),
    ("pure_deletion", "abcde", "ace"),
]


# ═══════════════════════════════════════════════
# BOC: manual Counter cross-check
# ═══════════════════════════════════════════════

def _manual_boc(ref: str, hyp: str) -> float:
    """Independent BOC computation using Counter."""
    if not ref and not hyp:
        return 0.0
    max_len = max(len(ref), len(hyp))
    if max_len == 0:
        return 0.0
    ref_counts = Counter(ref)
    hyp_counts = Counter(hyp)
    all_chars = set(ref_counts) | set(hyp_counts)
    diff = sum(abs(ref_counts.get(c, 0) - hyp_counts.get(c, 0)) for c in all_chars)
    return diff / max_len


class TestBOCCrosscheck:
    """Cross-check BOC against manual Counter computation."""

    @pytest.mark.crosscheck
    @pytest.mark.parametrize("desc, ref, hyp", PAIRS,
                             ids=[t[0] for t in PAIRS])
    def test_boc_vs_manual(self, desc, ref, hyp):
        """Our compute_boc() must match manual Counter-based computation."""
        ours = compute_boc(ref, hyp)
        manual = _manual_boc(ref, hyp)
        assert abs(ours - manual) < 1e-9, f"BOC mismatch: ours={ours}, manual={manual}"

    @pytest.mark.parametrize("desc, ref, hyp", PAIRS,
                             ids=[t[0] for t in PAIRS])
    def test_boc_symmetry(self, desc, ref, hyp):
        """BOC(a,b) == BOC(b,a) — symmetric metric."""
        assert abs(compute_boc(ref, hyp) - compute_boc(hyp, ref)) < 1e-9

    def test_boc_order_invariance(self):
        """BOC ignores character order — reversed string has same BOC."""
        assert compute_boc("abc", "cba") == 0.0
        assert compute_boc("abc", "bac") == 0.0
        assert compute_boc("abc", "cab") == 0.0
        assert compute_boc("hello", "ohlel") == 0.0

    def test_boc_identical(self):
        assert compute_boc("abc", "abc") == 0.0
        assert compute_boc("", "") == 0.0

    def test_boc_nonnegative(self):
        """BOC must be non-negative for all pairs."""
        for desc, ref, hyp in PAIRS:
            assert compute_boc(ref, hyp) >= 0.0


class TestBOCKnownValues:
    """Known values from evaluation_verification.md."""

    def test_b1_identical(self):
        assert compute_boc("abc", "abc") == 0.0

    def test_b2_empty(self):
        assert compute_boc("", "") == 0.0

    def test_b3_one_extra(self):
        assert compute_boc("abc", "abcd") == 0.25

    def test_b4_freq_diff(self):
        assert abs(compute_boc("aab", "abb") - 2 / 3) < 1e-9

    def test_b5_order_invariant(self):
        assert compute_boc("abc", "cba") == 0.0

    def test_b6_known_bug(self):
        """KNOWN BUG: BOC exceeds [0,1] bound for completely disjoint strings."""
        assert compute_boc("abc", "xyz") == 2.0

    def test_b7_known_bug(self):
        """KNOWN BUG: maximum BOC is 2.0, not 1.0."""
        assert compute_boc("aaaa", "bbbb") == 2.0


# ═══════════════════════════════════════════════
# BOC ≤ CER property (for order-preserving OCR)
# ═══════════════════════════════════════════════

class TestBOCBoundProperty:
    """BOC vs CER relationship.

    NOTE: BOC ≤ CER does NOT hold universally, even for order-preserving OCR.
    A single substitution "a"→"b" gives CER=1/1=1.0 but BOC=2/1=2.0 because
    BOC counts both the missing 'a' and the extra 'b' as differences.
    BOC ≤ CER only holds for pure insertions or pure deletions.
    """

    @pytest.mark.parametrize("desc, ref, hyp", [
        ("pure_insertion", "abc", "aXbYcZ"),
        ("pure_deletion", "abcde", "ace"),
        ("identical", "abc", "abc"),
        ("empty_both", "", ""),
    ], ids=["pure_insertion", "pure_deletion", "identical", "empty_both"])
    def test_boc_leq_cer_for_pure_insert_delete(self, desc, ref, hyp):
        """BOC ≤ CER holds for pure insertions/deletions (no substitutions)."""
        boc = compute_boc(ref, hyp)
        cer = compute_cer(ref, hyp) if ref else 0.0
        assert boc <= cer + 1e-9, f"BOC > CER: BOC={boc}, CER={cer} for {desc}"

    def test_boc_can_exceed_cer_for_substitutions(self):
        """Document: BOC > CER is expected when substitutions dominate.

        Each substitution contributes 1 to CER but 2 to BOC (one missing + one extra).
        """
        boc = compute_boc("a", "b")
        cer = compute_cer("a", "b")
        assert boc == 2.0  # |{a:1}−{b:1}| / max(1,1) = 2/1
        assert cer == 1.0
        assert boc > cer  # this is expected behavior, not a bug


# ═══════════════════════════════════════════════
# delta_CER consistency
# ═══════════════════════════════════════════════

class TestDeltaCERConsistency:
    """delta_CER = CER - BOC must be consistent."""

    @pytest.mark.crosscheck
    @pytest.mark.parametrize("desc, ref, hyp", PAIRS,
                             ids=[t[0] for t in PAIRS])
    def test_delta_cer_formula(self, desc, ref, hyp):
        """delta_CER ≈ CER - BOC (both rounded to 6 decimals).

        Tolerance: round(a-b,6) may differ from round(a,6)-round(b,6) by up to 1e-6.
        """
        cer = compute_cer(ref, hyp)
        oi = compute_order_independent(ref, hyp, cer)
        expected_delta = round(cer, 6) - oi["boc"]
        assert abs(oi["delta_cer"] - expected_delta) < 2e-6, (
            f"delta_CER mismatch: got {oi['delta_cer']}, expected {expected_delta}"
        )

    def test_delta_cer_reversed_string(self):
        """Pure ordering error: reversed string has delta_CER = CER."""
        cer = compute_cer("abc", "cba")
        oi = compute_order_independent("abc", "cba", cer)
        assert oi["boc"] == 0.0
        assert abs(oi["delta_cer"] - round(cer, 6)) < 1e-9

    @pytest.mark.parametrize("desc, ref, hyp",
                             [t for t in PAIRS if t[1]],
                             ids=[t[0] for t in PAIRS if t[1]])
    def test_delta_cer_for_identical(self, desc, ref, hyp):
        """Identical strings must have delta_CER = 0."""
        if ref != hyp:
            pytest.skip("Only for identical pairs")
        cer = compute_cer(ref, hyp)
        oi = compute_order_independent(ref, hyp, cer)
        assert oi["boc"] == 0.0
        assert oi["delta_cer"] == 0.0
