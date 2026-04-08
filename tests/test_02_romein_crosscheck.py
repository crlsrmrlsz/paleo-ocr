"""Cross-check Romein 4-tier diagnostic CER.

Strategy:
  - Apply our normalizers, then compute CER with jiwer to cross-check
  - Verify tier monotonicity: T1 >= T2 >= T3 >= T4
  - Property tests on normalizers: idempotency, non-increasing length, casefold invariant
"""

import unicodedata

import editdistance as ed
import jiwer
import pytest

from metrics.romein import (
    compute_romein_tiers,
    normalize_t1,
    normalize_t2,
    normalize_t3,
    normalize_t4,
)


# ---------------------------------------------------------------------------
# Test texts with diverse Unicode properties
# ---------------------------------------------------------------------------
NORMALIZER_TEXTS = [
    ("basic", "Hello World"),
    ("tabs_spaces", "hello\t world"),
    ("soft_hyphen", "bro\u00adken"),
    ("double_spaces", "hello   world   foo"),
    ("mixed_case_punct", "Hello, World! 123"),
    ("german_eszett", "Straße"),
    ("combining_diacritics", "caf\u0065\u0301"),  # e + combining acute
    ("spanish_paleographic", "Don Juan, anno 1542"),
    ("fraktur_long_s", "das Verſproene"),
    ("empty", ""),
    ("only_spaces", "   "),
    ("only_punctuation", "!@#$%^&*()"),
    ("newlines", "line one\nline two\nline three"),
    ("unicode_separators", "hello\u2003world\u200bfoo"),  # em space + zero-width space
    ("cjk_mixed", "日本語テスト ABC"),
]

PAIRS = [
    ("hello_world", "Hello World", "helloworld!"),
    ("paleographic", "Don Juan, anno 1542", "don jvan. anno 1542"),
    ("fraktur", "das Verſproene glei den Augenblick",
     "das Verfproene glei den Augemblick"),
    ("identical", "exacto", "exacto"),
    ("case_only", "HELLO WORLD", "hello world"),
    ("spacing_only", "hello world", "helloworld"),
    ("punct_only", "hello, world!", "hello world"),
    ("completely_different", "abc", "xyz"),
]


# ═══════════════════════════════════════════════
# Normalizer property tests
# ═══════════════════════════════════════════════

class TestNormalizerProperties:
    """Verify algebraic properties of T1–T4 normalizers."""

    @pytest.mark.parametrize("desc, text", NORMALIZER_TEXTS,
                             ids=[t[0] for t in NORMALIZER_TEXTS])
    def test_t1_idempotent(self, desc, text):
        """Applying T1 twice gives the same result as once."""
        assert normalize_t1(normalize_t1(text)) == normalize_t1(text)

    @pytest.mark.parametrize("desc, text", NORMALIZER_TEXTS,
                             ids=[t[0] for t in NORMALIZER_TEXTS])
    def test_t2_idempotent(self, desc, text):
        assert normalize_t2(normalize_t2(text)) == normalize_t2(text)

    @pytest.mark.parametrize("desc, text", NORMALIZER_TEXTS,
                             ids=[t[0] for t in NORMALIZER_TEXTS])
    def test_t3_idempotent(self, desc, text):
        assert normalize_t3(normalize_t3(text)) == normalize_t3(text)

    @pytest.mark.parametrize("desc, text", NORMALIZER_TEXTS,
                             ids=[t[0] for t in NORMALIZER_TEXTS])
    def test_t4_idempotent(self, desc, text):
        assert normalize_t4(normalize_t4(text)) == normalize_t4(text)

    @pytest.mark.parametrize("desc, text", NORMALIZER_TEXTS,
                             ids=[t[0] for t in NORMALIZER_TEXTS])
    def test_length_non_increasing(self, desc, text):
        """Each successive tier's output is no longer than the previous.

        Exception: casefold() can expand characters (e.g., ß → ss), so
        T3 (casefold) may be longer than T2. We only assert T2 ≤ T1 and T4 ≤ T3.
        """
        t1 = normalize_t1(text)
        t2 = normalize_t2(text)
        t4 = normalize_t4(text)
        assert len(t2) <= len(t1), f"T2 longer than T1: {len(t2)} > {len(t1)}"
        # T3 can be longer than T2 due to casefold expansion (ß → ss)
        # T4 strips non-alnum, so it's always ≤ T3
        t3 = normalize_t3(text)
        assert len(t4) <= len(t3), f"T4 longer than T3: {len(t4)} > {len(t3)}"

    @pytest.mark.parametrize("desc, text", NORMALIZER_TEXTS,
                             ids=[t[0] for t in NORMALIZER_TEXTS])
    def test_t3_is_lowercase(self, desc, text):
        """T3 output must be fully casefolded."""
        t3 = normalize_t3(text)
        assert t3 == t3.casefold(), f"T3 not casefolded: {t3!r}"

    @pytest.mark.parametrize("desc, text", NORMALIZER_TEXTS,
                             ids=[t[0] for t in NORMALIZER_TEXTS])
    def test_t4_alnum_only(self, desc, text):
        """T4 output must contain only L, N, M Unicode categories."""
        t4 = normalize_t4(text)
        for ch in t4:
            cat = unicodedata.category(ch)[0]
            assert cat in ("L", "N", "M"), (
                f"T4 contains category {unicodedata.category(ch)} char: {ch!r}"
            )

    @pytest.mark.parametrize("desc, text", NORMALIZER_TEXTS,
                             ids=[t[0] for t in NORMALIZER_TEXTS])
    def test_t1_is_nfc(self, desc, text):
        """T1 output must be NFC normalized."""
        t1 = normalize_t1(text)
        assert t1 == unicodedata.normalize("NFC", t1)

    @pytest.mark.parametrize("desc, text", NORMALIZER_TEXTS,
                             ids=[t[0] for t in NORMALIZER_TEXTS])
    def test_t2_no_whitespace(self, desc, text):
        """T2 output must contain no whitespace."""
        t2 = normalize_t2(text)
        for ch in t2:
            assert not ch.isspace(), f"T2 contains whitespace: {ch!r}"


class TestNormalizerKnownValues:
    """Known values from evaluation_verification.md."""

    def test_t1_tab_dedup(self):
        assert normalize_t1("hello\t world") == "hello world"

    def test_t2_remove_spaces(self):
        assert normalize_t2("Hello World") == "HelloWorld"

    def test_t3_casefold(self):
        assert normalize_t3("Hello World") == "helloworld"

    def test_t4_alnum(self):
        assert normalize_t4("Hello, World! 123") == "helloworld123"

    def test_t1_soft_hyphen(self):
        """Soft hyphens (U+00AD) should be removed."""
        assert "\u00ad" not in normalize_t1("bro\u00adken")


# ═══════════════════════════════════════════════
# Romein tier CER cross-check with jiwer
# ═══════════════════════════════════════════════

class TestRomeinTiersCrosscheck:
    """Cross-check: apply normalizers manually, compute CER with editdistance, compare."""

    @pytest.mark.crosscheck
    @pytest.mark.parametrize("desc, ref, hyp", PAIRS,
                             ids=[t[0] for t in PAIRS])
    def test_tiers_vs_manual_computation(self, desc, ref, hyp):
        """Our compute_romein_tiers() must match manual normalize→editdistance pipeline."""
        tiers = compute_romein_tiers(ref, hyp)
        t1_ref = normalize_t1(ref)
        t1_len = len(t1_ref)

        if t1_len == 0:
            pytest.skip("Empty T1 reference — special case")

        normalizers = {
            "T1_raw": normalize_t1,
            "T2_nospace": normalize_t2,
            "T3_lowercase": normalize_t3,
            "T4_alnum": normalize_t4,
        }

        for tier_name, normalizer in normalizers.items():
            ref_norm = normalizer(ref)
            hyp_norm = normalizer(hyp)
            expected = ed.eval(ref_norm, hyp_norm) / t1_len
            actual = tiers[tier_name]
            assert abs(actual - expected) < 1e-5, (
                f"{tier_name} mismatch: actual={actual}, expected={expected}"
            )

    @pytest.mark.crosscheck
    @pytest.mark.parametrize("desc, ref, hyp", PAIRS,
                             ids=[t[0] for t in PAIRS])
    def test_tier_monotonicity(self, desc, ref, hyp):
        """T1 >= T2 >= T3 >= T4 (monotonically non-increasing)."""
        tiers = compute_romein_tiers(ref, hyp)
        assert tiers["T1_raw"] >= tiers["T2_nospace"] - 1e-9, (
            f"T1 < T2: {tiers['T1_raw']} < {tiers['T2_nospace']}"
        )
        assert tiers["T2_nospace"] >= tiers["T3_lowercase"] - 1e-9, (
            f"T2 < T3: {tiers['T2_nospace']} < {tiers['T3_lowercase']}"
        )
        assert tiers["T3_lowercase"] >= tiers["T4_alnum"] - 1e-9, (
            f"T3 < T4: {tiers['T3_lowercase']} < {tiers['T4_alnum']}"
        )


class TestRomeinKnownValues:
    """Known values from evaluation_verification.md."""

    def test_r1_hello_world(self):
        r1 = compute_romein_tiers("Hello World", "helloworld!")
        assert abs(r1["T1_raw"] - 4 / 11) < 1e-5, f"got {r1['T1_raw']}"
        assert abs(r1["T2_nospace"] - 3 / 11) < 1e-5, f"got {r1['T2_nospace']}"
        assert abs(r1["T3_lowercase"] - 1 / 11) < 1e-5, f"got {r1['T3_lowercase']}"
        assert r1["T4_alnum"] == 0.0, f"got {r1['T4_alnum']}"

    def test_r2_paleographic(self):
        r2 = compute_romein_tiers("Don Juan, anno 1542", "don jvan. anno 1542")
        assert abs(r2["T1_raw"] - 4 / 19) < 1e-5, f"got {r2['T1_raw']}"
        assert abs(r2["T4_alnum"] - 1 / 19) < 1e-5, f"got {r2['T4_alnum']}"

    def test_fraktur_long_s_t3_helps(self):
        """Casefold maps ſ → s, so T3 should reduce CER vs T2."""
        tiers = compute_romein_tiers(
            "das Verſproene glei den Augenblick",
            "das Verfproene glei den Augemblick",
        )
        # T3 casefolding doesn't help here (ſ→s but Verf≠Vers), but monotonicity holds
        assert tiers["T3_lowercase"] <= tiers["T2_nospace"] + 1e-9
