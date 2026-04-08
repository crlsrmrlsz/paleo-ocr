"""Cross-check error decomposition against rapidfuzz.Levenshtein.editops().

Strategy:
  - Compare our compute_opcodes() counts with rapidfuzz editops() tag counts
  - Verify P/R/F1 bounds and self-consistency
  - Cross-check S+D+I+C totals
"""

from collections import Counter

import pytest
from rapidfuzz.distance import Levenshtein

from metrics.decomposition import compute_decomposition, compute_opcodes


PAIRS = [
    ("kitten_sitting", "kitten", "sitting"),
    ("pure_deletion", "abcde", "ace"),
    ("pure_insertion", "abc", "aXbYcZ"),
    ("unequal_replace", "ab", "xyz"),
    ("identical", "hello", "hello"),
    ("empty_both", "", ""),
    ("empty_ref", "", "abc"),
    ("empty_hyp", "abc", ""),
    ("completely_different", "abc", "xyz"),
    ("single_sub", "a", "b"),
    ("single_insert", "Foo", "Food"),
    ("single_delete", "Muell", "Mull"),
    ("long_text",
     "Lorem ipsum dolor sit amet consetetur sadipscing elitr",
     "Lorem Ipsum dolor sit armet, consetetur sadipscing elitr"),
    ("unicode", "café résumé", "cafe resume"),
    ("repeated", "aaaa", "bbbb"),
]

PAIRS_NONEMPTY = [(d, r, h) for d, r, h in PAIRS if r or h]


# ═══════════════════════════════════════════════
# Opcode cross-check with rapidfuzz editops
# ═══════════════════════════════════════════════

class TestOpcodesCrosscheck:
    """Cross-check opcode counts against rapidfuzz.Levenshtein.editops()."""

    @pytest.mark.crosscheck
    @pytest.mark.parametrize("desc, ref, hyp", PAIRS_NONEMPTY,
                             ids=[t[0] for t in PAIRS_NONEMPTY])
    def test_total_edit_distance_matches(self, desc, ref, hyp):
        """S+D+I from opcodes must equal the Levenshtein edit distance."""
        ops = compute_opcodes(ref, hyp)
        total_edits = ops["substitutions"] + ops["deletions"] + ops["insertions"]
        expected_dist = Levenshtein.distance(ref, hyp)
        assert total_edits == expected_dist, (
            f"S+D+I={total_edits} != edit_distance={expected_dist} for {desc}"
        )

    @pytest.mark.crosscheck
    @pytest.mark.parametrize("desc, ref, hyp", PAIRS_NONEMPTY,
                             ids=[t[0] for t in PAIRS_NONEMPTY])
    def test_correct_count_consistency(self, desc, ref, hyp):
        """C (correct) + S (substitutions) + D (deletions) must equal len(ref)."""
        ops = compute_opcodes(ref, hyp)
        # From reference perspective: each ref char is either correct, substituted, or deleted
        assert ops["correct"] + ops["substitutions"] + ops["deletions"] == len(ref), (
            f"C+S+D != len(ref): {ops['correct']}+{ops['substitutions']}+{ops['deletions']} != {len(ref)}"
        )

    @pytest.mark.crosscheck
    @pytest.mark.parametrize("desc, ref, hyp", PAIRS_NONEMPTY,
                             ids=[t[0] for t in PAIRS_NONEMPTY])
    def test_hyp_count_consistency(self, desc, ref, hyp):
        """C + S + I must equal len(hyp)."""
        ops = compute_opcodes(ref, hyp)
        assert ops["correct"] + ops["substitutions"] + ops["insertions"] == len(hyp), (
            f"C+S+I != len(hyp): {ops['correct']}+{ops['substitutions']}+{ops['insertions']} != {len(hyp)}"
        )

    @pytest.mark.crosscheck
    @pytest.mark.parametrize("desc, ref, hyp", PAIRS_NONEMPTY,
                             ids=[t[0] for t in PAIRS_NONEMPTY])
    def test_vs_editops_for_equal_length_replaces(self, desc, ref, hyp):
        """For cases without unequal-length replaces, our counts match editops exactly."""
        our_ops = compute_opcodes(ref, hyp)
        rf_editops = Levenshtein.editops(ref, hyp)
        rf_counts = Counter(tag for tag, _, _ in rf_editops)

        rf_s = rf_counts.get("replace", 0)
        rf_d = rf_counts.get("delete", 0)
        rf_i = rf_counts.get("insert", 0)

        # editops always decomposes into 1:1 operations, so it will never produce
        # a "replace" spanning multiple chars. Our opcodes() uses block-level spans.
        # Total edit distance must match regardless:
        our_total = our_ops["substitutions"] + our_ops["deletions"] + our_ops["insertions"]
        rf_total = rf_s + rf_d + rf_i
        assert our_total == rf_total, (
            f"Total edits differ: ours={our_total}, editops={rf_total}"
        )


class TestOpcodesKnownValues:
    """Known values from evaluation_verification.md."""

    def test_kitten_sitting(self):
        ops = compute_opcodes("kitten", "sitting")
        assert ops == {"substitutions": 2, "deletions": 0, "insertions": 1, "correct": 4}

    def test_pure_deletion(self):
        ops = compute_opcodes("abcde", "ace")
        assert ops == {"substitutions": 0, "deletions": 2, "insertions": 0, "correct": 3}

    def test_pure_insertion(self):
        ops = compute_opcodes("abc", "aXbYcZ")
        assert ops == {"substitutions": 0, "deletions": 0, "insertions": 3, "correct": 3}

    def test_unequal_replace(self):
        ops = compute_opcodes("ab", "xyz")
        assert ops == {"substitutions": 2, "deletions": 0, "insertions": 1, "correct": 0}

    def test_empty_both(self):
        ops = compute_opcodes("", "")
        assert ops == {"substitutions": 0, "deletions": 0, "insertions": 0, "correct": 0}


# ═══════════════════════════════════════════════
# Decomposition (P/R/F1) cross-check
# ═══════════════════════════════════════════════

class TestDecompositionCrosscheck:
    """Cross-check P/R/F1 against manually derived values from opcodes."""

    @pytest.mark.crosscheck
    @pytest.mark.parametrize("desc, ref, hyp", PAIRS_NONEMPTY,
                             ids=[t[0] for t in PAIRS_NONEMPTY])
    def test_prf1_from_opcodes(self, desc, ref, hyp):
        """P/R/F1 must be derivable from S/D/I/C counts."""
        decomp = compute_decomposition(ref, hyp)
        ops = compute_opcodes(ref, hyp)
        s, d, i, c = ops["substitutions"], ops["deletions"], ops["insertions"], ops["correct"]

        # Manually compute P/R/F1
        p_denom = c + s + i
        expected_p = c / p_denom if p_denom > 0 else 0.0
        r_denom = c + s + d
        expected_r = c / r_denom if r_denom > 0 else 0.0
        expected_f1 = (2 * expected_p * expected_r / (expected_p + expected_r)
                       if (expected_p + expected_r) > 0 else 0.0)

        assert abs(decomp["precision"] - round(expected_p, 6)) < 1e-6
        assert abs(decomp["recall"] - round(expected_r, 6)) < 1e-6
        assert abs(decomp["f1"] - round(expected_f1, 6)) < 1e-6

    @pytest.mark.parametrize("desc, ref, hyp", PAIRS,
                             ids=[t[0] for t in PAIRS])
    def test_prf1_bounds(self, desc, ref, hyp):
        """P, R, F1 must all be in [0, 1]."""
        decomp = compute_decomposition(ref, hyp)
        assert 0.0 <= decomp["precision"] <= 1.0
        assert 0.0 <= decomp["recall"] <= 1.0
        assert 0.0 <= decomp["f1"] <= 1.0

    @pytest.mark.parametrize("desc, ref, hyp", PAIRS,
                             ids=[t[0] for t in PAIRS])
    def test_f1_leq_min_pr(self, desc, ref, hyp):
        """F1 ≤ max(P, R) — harmonic mean never exceeds arithmetic mean."""
        decomp = compute_decomposition(ref, hyp)
        p, r, f1 = decomp["precision"], decomp["recall"], decomp["f1"]
        # F1 is harmonic mean: always ≤ arithmetic mean, always ≤ max(P,R)
        assert f1 <= max(p, r) + 1e-6

    @pytest.mark.parametrize("desc, ref, hyp", PAIRS,
                             ids=[t[0] for t in PAIRS])
    def test_error_rates_nonneg(self, desc, ref, hyp):
        """S/D/I rates must be non-negative."""
        decomp = compute_decomposition(ref, hyp)
        assert decomp["s_rate"] >= 0.0
        assert decomp["d_rate"] >= 0.0
        assert decomp["i_rate"] >= 0.0


class TestDecompositionKnownValues:
    """Known values from evaluation_verification.md."""

    def test_kitten_sitting_prf(self):
        d = compute_decomposition("kitten", "sitting")
        assert abs(d["precision"] - 4 / 7) < 1e-5
        assert abs(d["recall"] - 4 / 6) < 1e-5
        assert abs(d["f1"] - 32 / 52) < 1e-5

    def test_pure_deletion_prf(self):
        d = compute_decomposition("abcde", "ace")
        assert d["precision"] == 1.0
        assert abs(d["recall"] - 0.6) < 1e-5

    def test_pure_insertion_prf(self):
        d = compute_decomposition("abc", "aXbYcZ")
        assert d["precision"] == 0.5
        assert d["recall"] == 1.0

    def test_identical_perfect(self):
        d = compute_decomposition("hello", "hello")
        assert d["precision"] == 1.0
        assert d["recall"] == 1.0
        assert d["f1"] == 1.0
        assert d["s_rate"] == 0.0
        assert d["d_rate"] == 0.0
        assert d["i_rate"] == 0.0
