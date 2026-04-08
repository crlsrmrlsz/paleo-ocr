"""Cross-check core metrics against independent library implementations.

Strategy:
  CER   → our compute_cer() vs jiwer.cer()
  CER_n → our compute_cer_n() vs derived from jiwer.process_characters()
  NLS   → our compute_nls() vs rapidfuzz.Levenshtein.normalized_similarity()
  WER   → our compute_wer() vs editdistance at word level (fastwer unavailable on 3.13)
  SER   → trivial metric, unit tests only (no independent library)
"""

import pytest
import editdistance as ed
import jiwer
from rapidfuzz.distance import Levenshtein

from metrics.core import compute_cer, compute_cer_n, compute_nls, compute_wer, compute_ser


# ---------------------------------------------------------------------------
# Test pairs (inline for self-contained module; conftest pairs also used via fixture)
# ---------------------------------------------------------------------------
PAIRS = [
    ("identical", "hello world", "hello world"),
    ("single_sub", "kitten", "sitting"),
    ("single_char", "a", "b"),
    ("foo_bar", "Foo", "Bar"),
    ("pure_deletion", "abcde", "ace"),
    ("pure_insertion", "abc", "aXbYcZ"),
    ("empty_both", "", ""),
    ("empty_hyp", "some text here", ""),
    ("completely_different", "abc", "xyz"),
    ("longer_hyp", "X", "X X Y Y"),
    ("unicode_diacritics", "café résumé naïve", "cafe resume naive"),
    ("spanish", "Don Juan, anno 1542", "don jvan. anno 1542"),
    ("long_lorem",
     "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam "
     "nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam "
     "erat, sed diam voluptua. At vero eos et accusam et justo duo "
     "dolores et ea rebum. Stet clita kasd gubergren, no sea takimata "
     "sanctus est Lorem ipsum dolor sit amet.",
     "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam "
     "nonu yy eirmod tempor invidunt ut labore et dolore magna aliquyam "
     "erat, sed diam voluptua. At Vero eos et accusam et justo duo "
     "dolores et ea rebum. Stet clita kasd gubergren, no sea iakimata "
     "sanctus est Lorem Ipsum dolor sit amet."),
    ("fraktur", "das Verſproene glei den Augenblick",
     "das Verfproene glei den Augemblick"),
    ("repeated_chars", "aaaa", "bbbb"),
    ("single_delete", "Muell", "Mull"),
    ("single_insert", "Foo", "Food"),
    ("multiword", "this is a test", "this a test"),
]

PAIRS_NONEMPTY_REF = [(d, r, h) for d, r, h in PAIRS if r]


# ═══════════════════════════════════════════════
# CER: our compute_cer() vs jiwer.cer()
# ═══════════════════════════════════════════════

class TestCERCrosscheck:
    """CER cross-check: editdistance.eval/len(ref) vs jiwer.cer()."""

    @pytest.mark.crosscheck
    @pytest.mark.parametrize("desc, ref, hyp", PAIRS_NONEMPTY_REF,
                             ids=[t[0] for t in PAIRS_NONEMPTY_REF])
    def test_cer_vs_jiwer(self, desc, ref, hyp):
        """Our CER must match jiwer.cer() for non-empty references."""
        ours = compute_cer(ref, hyp)
        theirs = jiwer.cer(ref, hyp)
        assert abs(ours - theirs) < 1e-9, f"CER mismatch: ours={ours}, jiwer={theirs}"

    def test_cer_empty_both(self):
        assert compute_cer("", "") == 0.0

    def test_cer_empty_ref(self):
        """Our convention: empty ref → 1.0 (dinglehopper returns inf)."""
        assert compute_cer("", "Foo") == 1.0

    def test_cer_known_values(self):
        """Exact values from evaluation_verification.md."""
        assert compute_cer("a", "a") == 0.0
        assert compute_cer("a", "b") == 1.0
        assert compute_cer("Foo", "Bar") == 1.0
        assert compute_cer("Foo", "") == 1.0
        assert abs(compute_cer("Foo", "Food") - 1 / 3) < 1e-9
        assert abs(compute_cer("Fnord", "Food") - 0.4) < 1e-9
        assert abs(compute_cer("Muell", "Mull") - 0.2) < 1e-9
        assert abs(compute_cer("X", "X X Y Y") - 6.0) < 1e-9
        assert abs(compute_cer("kitten", "sitting") - 0.5) < 1e-9


# ═══════════════════════════════════════════════
# CER_n: our compute_cer_n() vs jiwer.process_characters()
# ═══════════════════════════════════════════════

class TestCERnCrosscheck:
    """CER_n cross-check: (S+D+I)/(S+D+I+C) vs jiwer decomposition.

    NOTE: Our CER_n uses max(ref_span, hyp_span) for "replace" opcodes,
    while jiwer counts substitutions 1:1.  For equal-length replaces they
    agree; for unequal replaces they may differ.  We test both cases.
    """

    @pytest.mark.crosscheck
    @pytest.mark.parametrize("desc, ref, hyp", PAIRS_NONEMPTY_REF,
                             ids=[t[0] for t in PAIRS_NONEMPTY_REF])
    def test_cer_n_vs_jiwer_characters(self, desc, ref, hyp):
        """Cross-check against jiwer.process_characters() derived CER_n."""
        ours = compute_cer_n(ref, hyp)

        result = jiwer.process_characters([ref], [hyp])
        s = result.substitutions
        d = result.deletions
        i = result.insertions
        c = result.hits
        total = s + d + i + c
        jiwer_cer_n = (s + d + i) / total if total > 0 else 0.0

        # For equal-length replaces the two formulas agree exactly.
        # For unequal replaces, allow a tolerance since the accounting differs.
        # Both must be in [0, 1].
        assert 0.0 <= ours <= 1.0, f"CER_n out of bounds: {ours}"
        assert 0.0 <= jiwer_cer_n <= 1.0, f"jiwer CER_n out of bounds: {jiwer_cer_n}"

        # Relaxed tolerance for unequal-replace cases
        assert abs(ours - jiwer_cer_n) < 0.15, (
            f"CER_n divergence too large: ours={ours:.6f}, jiwer={jiwer_cer_n:.6f}"
        )

    def test_cer_n_known_values(self):
        """Exact values from evaluation_verification.md."""
        assert compute_cer_n("abc", "abc") == 0.0
        assert compute_cer_n("", "") == 0.0
        assert compute_cer_n("", "abc") == 1.0
        assert compute_cer_n("abc", "xyz") == 1.0
        assert abs(compute_cer_n("Foo", "Food") - 0.25) < 1e-9
        assert abs(compute_cer_n("kitten", "sitting") - 3 / 7) < 1e-9
        assert abs(compute_cer_n("Muell", "Mull") - 0.2) < 1e-9
        assert abs(compute_cer_n("ab", "xyz") - 1.0) < 1e-9

    def test_cer_n_bounded(self):
        """CER_n must always be in [0, 1]."""
        for desc, ref, hyp in PAIRS:
            val = compute_cer_n(ref, hyp)
            assert 0.0 <= val <= 1.0, f"CER_n out of [0,1] for {desc}: {val}"

    def test_cer_n_leq_cer_when_cer_leq_1(self):
        """When CER ≤ 1, CER_n ≤ CER (normalization can only reduce)."""
        for desc, ref, hyp in PAIRS_NONEMPTY_REF:
            cer = compute_cer(ref, hyp)
            if cer <= 1.0:
                cer_n = compute_cer_n(ref, hyp)
                assert cer_n <= cer + 1e-9, (
                    f"CER_n > CER for {desc}: CER_n={cer_n}, CER={cer}"
                )


# ═══════════════════════════════════════════════
# NLS: our compute_nls() vs rapidfuzz normalized_similarity()
# ═══════════════════════════════════════════════

class TestNLSCrosscheck:
    """NLS cross-check: our formula vs rapidfuzz C++ implementation."""

    @pytest.mark.crosscheck
    @pytest.mark.parametrize("desc, ref, hyp", PAIRS,
                             ids=[t[0] for t in PAIRS])
    def test_nls_vs_rapidfuzz(self, desc, ref, hyp):
        """Our NLS must match rapidfuzz.Levenshtein.normalized_similarity()."""
        ours = compute_nls(ref, hyp)
        theirs = Levenshtein.normalized_similarity(ref, hyp)
        assert abs(ours - theirs) < 1e-9, f"NLS mismatch: ours={ours}, rf={theirs}"

    def test_nls_known_values(self):
        """Exact values from evaluation_verification.md."""
        assert compute_nls("", "") == 1.0
        assert compute_nls("abc", "abc") == 1.0
        assert compute_nls("abc", "") == 0.0
        assert compute_nls("", "abc") == 0.0
        assert abs(compute_nls("kitten", "sitting") - 4 / 7) < 1e-9
        assert abs(compute_nls("lewenstein", "levenshtein") - 9 / 11) < 1e-9
        assert abs(compute_nls("Foo", "Food") - 0.75) < 1e-9
        assert abs(compute_nls("Fnord", "Food") - 0.6) < 1e-9

    def test_nls_bounded(self):
        """NLS must always be in [0, 1]."""
        for desc, ref, hyp in PAIRS:
            val = compute_nls(ref, hyp)
            assert 0.0 <= val <= 1.0, f"NLS out of [0,1] for {desc}: {val}"

    def test_nls_symmetry(self):
        """NLS(a,b) == NLS(b,a) — symmetric metric."""
        for desc, ref, hyp in PAIRS:
            assert abs(compute_nls(ref, hyp) - compute_nls(hyp, ref)) < 1e-9


# ═══════════════════════════════════════════════
# WER: our compute_wer() vs independent word-level editdistance
# ═══════════════════════════════════════════════

def _independent_wer(ref: str, hyp: str) -> float:
    """Compute WER independently using editdistance at word level."""
    ref_words = ref.strip().split()
    hyp_words = hyp.strip().split()
    if not ref_words:
        return 1.0 if hyp_words else 0.0
    return ed.eval(ref_words, hyp_words) / len(ref_words)


class TestWERCrosscheck:
    """WER cross-check: jiwer wrapper vs independent word-level Levenshtein."""

    @pytest.mark.crosscheck
    @pytest.mark.parametrize("desc, ref, hyp", PAIRS_NONEMPTY_REF,
                             ids=[t[0] for t in PAIRS_NONEMPTY_REF])
    def test_wer_vs_independent(self, desc, ref, hyp):
        """Our WER must match independent word-level editdistance / len(ref_words)."""
        if not ref.strip():
            pytest.skip("Whitespace-only reference handled by convention")
        ours = compute_wer(ref, hyp)
        independent = _independent_wer(ref, hyp)
        assert abs(ours - independent) < 1e-6, (
            f"WER mismatch: ours={ours}, independent={independent}"
        )

    def test_wer_known_values(self):
        """Exact values from evaluation_verification.md."""
        assert compute_wer("X", "X") == 0.0
        assert compute_wer("X", "Y") == 1.0
        assert compute_wer("X", "X X Y Y") == 3.0
        assert abs(compute_wer("X Y X", "X Z") - 2 / 3) < 1e-6
        assert compute_wer("X", "Y Z") == 2.0
        assert compute_wer("this is a test", "this is a test") == 0.0
        assert compute_wer("this is a test", "this a test") == 0.25
        assert compute_wer("", "") == 0.0
        assert compute_wer("  ", "hello") == 1.0


# ═══════════════════════════════════════════════
# SER: unit tests (no independent library — trivial metric)
# ═══════════════════════════════════════════════

class TestSERUnit:
    """SER unit tests from evaluation_verification.md."""

    def test_ser_known_values(self):
        assert compute_ser(["hello", "world"], ["hello", "world"]) == 0.0
        assert compute_ser(["hello", "world"], ["hello", "World"]) == 0.5
        assert abs(compute_ser(["a", "b", "c"], ["a", "b"]) - 1 / 3) < 1e-9
        assert abs(compute_ser(["a"], ["a", "b", "c"]) - 2 / 3) < 1e-9
        assert compute_ser([], ["hello"]) == 1.0
        assert compute_ser([], []) == 0.0
        assert compute_ser(["x"], ["y"]) == 1.0

    def test_ser_symmetry_of_length(self):
        """Extra lines in either direction count as errors."""
        # 3 ref lines vs 1 hyp line → 2/3 error
        assert abs(compute_ser(["a", "b", "c"], ["a"]) - 2 / 3) < 1e-9
        # 1 ref line vs 3 hyp lines → 2/3 error
        assert abs(compute_ser(["a"], ["a", "b", "c"]) - 2 / 3) < 1e-9


# ═══════════════════════════════════════════════
# Cross-metric consistency
# ═══════════════════════════════════════════════

class TestCoreConsistency:
    """Cross-metric consistency checks."""

    @pytest.mark.parametrize("desc, ref, hyp", PAIRS,
                             ids=[t[0] for t in PAIRS])
    def test_nls_plus_cer_consistency(self, desc, ref, hyp):
        """For identical-length strings: NLS ≈ 1 - CER (approximate)."""
        if not ref or not hyp or len(ref) != len(hyp):
            pytest.skip("Only applies to same-length non-empty pairs")
        cer = compute_cer(ref, hyp)
        nls = compute_nls(ref, hyp)
        # When len(ref) == len(hyp), NLS = 1 - dist/len = 1 - CER
        assert abs(nls - (1 - cer)) < 1e-9, f"NLS + CER != 1 for equal-length: NLS={nls}, CER={cer}"

    @pytest.mark.parametrize("desc, ref, hyp", PAIRS_NONEMPTY_REF,
                             ids=[t[0] for t in PAIRS_NONEMPTY_REF])
    def test_cer_nonnegative(self, desc, ref, hyp):
        """CER must be non-negative."""
        assert compute_cer(ref, hyp) >= 0.0

    @pytest.mark.parametrize("desc, ref, hyp", PAIRS,
                             ids=[t[0] for t in PAIRS])
    def test_identical_perfect_scores(self, desc, ref, hyp):
        """Identical strings must give perfect scores."""
        if ref != hyp:
            pytest.skip("Only for identical pairs")
        assert compute_cer(ref, hyp) == 0.0
        assert compute_cer_n(ref, hyp) == 0.0
        assert compute_nls(ref, hyp) == 1.0
        if ref.strip():
            assert compute_wer(ref, hyp) == 0.0
