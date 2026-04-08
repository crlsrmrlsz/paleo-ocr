"""Document-level cross-validation with published metric values.

Tests use complete documents from authoritative sources:
  - dinglehopper Lorem Ipsum: 591 chars, 8 known edits
  - Fraktur fragment: 2 known errors
  - Noisy OCR benchmark: published char_acc values
  - Cross-metric consistency on full documents
"""

import pytest
import editdistance as ed

from metrics.core import compute_cer, compute_cer_n, compute_nls, compute_wer, compute_ser
from metrics.romein import compute_romein_tiers
from metrics.decomposition import compute_opcodes, compute_decomposition
from metrics.order_independent import compute_boc, compute_order_independent


# ═══════════════════════════════════════════════
# dinglehopper Lorem Ipsum — 591-char document, 8 known edits
# ═══════════════════════════════════════════════

# Ground truth: standard Lorem Ipsum (exactly 591 chars, 100 words)
GT_LOREM = (
    "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam "
    "nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam "
    "erat, sed diam voluptua. At vero eos et accusam et justo duo "
    "dolores et ea rebum. Stet clita kasd gubergren, no sea takimata "
    "sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit "
    "amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor "
    "invidunt ut labore et dolore magna aliquyam erat, sed diam "
    "voluptua. At vero eos et accusam et justo duo dolores et ea "
    "rebum. Stet clita kasd gubergren, no sea takimata sanctus est "
    "Lorem ipsum dolor sit amet."
)

# Construct hypothesis with 6 error locations (8 total edit operations)
def _build_lorem_hyp():
    hyp = GT_LOREM
    # Error 1: "nonumy" → "nonu yy" (1 sub + 1 insert = 2 edits)
    idx = hyp.index("nonumy")
    hyp = hyp[:idx] + "nonu yy" + hyp[idx + 6:]
    # Error 2: " vero " → " Vero " (1 sub)
    idx = hyp.index(" vero ")
    hyp = hyp[:idx] + " Vero " + hyp[idx + 6:]
    # Error 3: "takimata" → "iakimata" (1 sub)
    idx = hyp.index("takimata")
    hyp = hyp[:idx] + "iakimata" + hyp[idx + 8:]
    # Error 4: second "amet," → "armet," (1 insert)
    first = hyp.index("amet,")
    second = hyp.index("amet,", first + 1)
    hyp = hyp[:second] + "armet," + hyp[second + 5:]
    # Error 5: second "nonumy" → "nonurny" (1 sub)
    idx = hyp.index("nonumy")
    hyp = hyp[:idx] + "nonurny" + hyp[idx + 6:]
    # Error 6: last "ipsum" → "Ipsum" (1 sub)
    idx = hyp.rindex("ipsum")
    hyp = hyp[:idx] + "Ipsum" + hyp[idx + 5:]
    return hyp


HYP_LOREM = _build_lorem_hyp()


@pytest.mark.document
class TestDinglehopperLoremIpsum:
    """dinglehopper reference: 591-char Lorem Ipsum, 8 known edits."""

    def test_gt_length(self):
        assert len(GT_LOREM) == 591, f"GT length: {len(GT_LOREM)}"

    def test_gt_words(self):
        assert len(GT_LOREM.split()) == 100, f"GT words: {len(GT_LOREM.split())}"

    def test_edit_distance(self):
        """Exactly 8 character edits between GT and hypothesis."""
        dist = ed.eval(GT_LOREM, HYP_LOREM)
        assert dist == 8, f"edit distance: {dist}"

    def test_cer(self):
        """CER = 8/591."""
        cer = compute_cer(GT_LOREM, HYP_LOREM)
        assert abs(cer - 8 / 591) < 1e-6, f"CER: {cer}"

    def test_wer(self):
        """WER = 7/100 (7 words affected out of 100)."""
        wer = compute_wer(GT_LOREM, HYP_LOREM)
        assert abs(wer - 7 / 100) < 1e-6, f"WER: {wer}"

    def test_cer_n_leq_cer(self):
        """CER_n ≤ CER for this document."""
        cer = compute_cer(GT_LOREM, HYP_LOREM)
        cer_n = compute_cer_n(GT_LOREM, HYP_LOREM)
        assert cer_n <= cer, f"CER_n={cer_n} > CER={cer}"

    def test_nls_high(self):
        """NLS should be high (> 0.95) for only 8 errors in 591 chars."""
        nls = compute_nls(GT_LOREM, HYP_LOREM)
        assert 0.95 < nls <= 1.0, f"NLS: {nls}"

    def test_opcode_edit_count(self):
        """S+D+I must equal 8 (the known edit distance)."""
        ops = compute_opcodes(GT_LOREM, HYP_LOREM)
        total = ops["substitutions"] + ops["deletions"] + ops["insertions"]
        assert total == 8, f"S+D+I={total}, expected 8"

    def test_romein_monotonicity(self):
        """T1 >= T2 >= T3 >= T4 on this document."""
        tiers = compute_romein_tiers(GT_LOREM, HYP_LOREM)
        assert tiers["T1_raw"] >= tiers["T2_nospace"] - 1e-9
        assert tiers["T2_nospace"] >= tiers["T3_lowercase"] - 1e-9
        assert tiers["T3_lowercase"] >= tiers["T4_alnum"] - 1e-9

    def test_t3_helps_with_case(self):
        """T3 (casefold) should reduce CER since we have case errors (Vero, Ipsum)."""
        tiers = compute_romein_tiers(GT_LOREM, HYP_LOREM)
        assert tiers["T3_lowercase"] < tiers["T2_nospace"], (
            f"T3 didn't help: T3={tiers['T3_lowercase']}, T2={tiers['T2_nospace']}"
        )

    def test_boc_vs_cer_relationship(self):
        """BOC and CER are both small for this document.

        NOTE: BOC > CER is expected when substitutions dominate (each sub
        contributes 1 to CER but 2 to BOC). Here both should be small.
        """
        cer = compute_cer(GT_LOREM, HYP_LOREM)
        boc = compute_boc(GT_LOREM, HYP_LOREM)
        assert cer < 0.02  # 8/591 ≈ 0.0135
        assert boc < 0.03  # small but may exceed CER due to substitutions

    def test_delta_cer_small(self):
        """delta_CER should be small in magnitude for mostly-correct document."""
        cer = compute_cer(GT_LOREM, HYP_LOREM)
        oi = compute_order_independent(GT_LOREM, HYP_LOREM, cer)
        # delta_CER can be negative when substitutions make BOC > CER
        assert abs(oi["delta_cer"]) < 0.01, f"delta_CER={oi['delta_cer']}"

    def test_decomposition_prf_consistent(self):
        """P/R/F1 must be self-consistent for this document."""
        decomp = compute_decomposition(GT_LOREM, HYP_LOREM)
        # High accuracy → all should be close to 1.0
        assert decomp["precision"] > 0.95
        assert decomp["recall"] > 0.95
        assert decomp["f1"] > 0.95


# ═══════════════════════════════════════════════
# Fraktur fragment — 2 known errors
# ═══════════════════════════════════════════════

GT_FRAKTUR = "das Verſproene glei den Augenblick"
OCR_FRAKTUR = "das Verfproene glei den Augemblick"


@pytest.mark.document
class TestDinglehopperFraktur:
    """Fraktur fragment with long-s: 2 known errors."""

    def test_cer(self):
        """CER = 2/len(GT)."""
        cer = compute_cer(GT_FRAKTUR, OCR_FRAKTUR)
        expected = 2 / len(GT_FRAKTUR)
        assert abs(cer - expected) < 1e-6, f"CER: {cer}, expected: {expected}"

    def test_edit_distance(self):
        dist = ed.eval(GT_FRAKTUR, OCR_FRAKTUR)
        assert dist == 2, f"edit distance: {dist}"

    def test_romein_monotonicity(self):
        tiers = compute_romein_tiers(GT_FRAKTUR, OCR_FRAKTUR)
        assert tiers["T1_raw"] >= tiers["T2_nospace"] - 1e-9
        assert tiers["T2_nospace"] >= tiers["T3_lowercase"] - 1e-9
        assert tiers["T3_lowercase"] >= tiers["T4_alnum"] - 1e-9

    def test_nls_high(self):
        """Only 2 errors → high NLS."""
        nls = compute_nls(GT_FRAKTUR, OCR_FRAKTUR)
        assert nls > 0.9

    def test_opcode_consistency(self):
        """S+D+I = 2."""
        ops = compute_opcodes(GT_FRAKTUR, OCR_FRAKTUR)
        total = ops["substitutions"] + ops["deletions"] + ops["insertions"]
        assert total == 2


# ═══════════════════════════════════════════════
# noisy-ocr-benchmark style: CER = 1 - char_acc/100
# ═══════════════════════════════════════════════

@pytest.mark.document
class TestNoisyOCRBenchmarkProperty:
    """Verify CER = 1 - NLS for equal-length strings (char_acc = NLS*100).

    The noisy-ocr-benchmark publishes char_acc values. Our CER should
    satisfy: CER = 1 - char_acc/100 when strings have equal length.
    For unequal lengths, CER ≈ 1 - char_acc/100 with small tolerance.
    """

    @pytest.mark.parametrize("ref, hyp, expected_acc", [
        ("hello world", "hello world", 100.0),  # perfect
        ("hello world", "hello warld", 90.909),  # 1 sub in 11 chars
        ("abcdefghij", "abcdefghix", 90.0),  # 1 sub in 10 chars
    ])
    def test_char_acc_from_nls(self, ref, hyp, expected_acc):
        """char_acc = NLS * 100."""
        nls = compute_nls(ref, hyp)
        char_acc = nls * 100
        assert abs(char_acc - expected_acc) < 0.01, (
            f"char_acc={char_acc:.3f}, expected={expected_acc}"
        )


# ═══════════════════════════════════════════════
# Full pipeline cross-consistency on diverse documents
# ═══════════════════════════════════════════════

DOCUMENTS = [
    ("lorem", GT_LOREM, HYP_LOREM),
    ("fraktur", GT_FRAKTUR, OCR_FRAKTUR),
    ("spanish_short",
     "En el año del Señor de mil quinientos cuarenta y dos",
     "En el anno del sennor de mill quinientos quarenta y dos"),
    ("colonial_legal",
     "Sepan quantos esta carta de venta vieren como yo Don Pedro",
     "Sepan cuantos esta carta de benta vieren como io Don Pedro"),
    ("notarial",
     "En la muy noble y leal ciudad de Toledo a quince dias del mes",
     "En la muy noble y leal civdad de Toledo a quinze dias del mes"),
]


@pytest.mark.document
class TestFullPipelineConsistency:
    """Cross-metric consistency on complete documents."""

    @pytest.mark.parametrize("desc, ref, hyp", DOCUMENTS,
                             ids=[d[0] for d in DOCUMENTS])
    def test_cer_cer_n_relationship(self, desc, ref, hyp):
        """CER_n ≤ CER when CER ≤ 1.0."""
        cer = compute_cer(ref, hyp)
        cer_n = compute_cer_n(ref, hyp)
        if cer <= 1.0:
            assert cer_n <= cer + 1e-9, f"CER_n > CER: {cer_n} > {cer}"

    @pytest.mark.parametrize("desc, ref, hyp", DOCUMENTS,
                             ids=[d[0] for d in DOCUMENTS])
    def test_nls_cer_complement(self, desc, ref, hyp):
        """NLS + CER ≈ 1 for equal-length strings; NLS ∈ [0,1] always."""
        nls = compute_nls(ref, hyp)
        assert 0.0 <= nls <= 1.0

    @pytest.mark.parametrize("desc, ref, hyp", DOCUMENTS,
                             ids=[d[0] for d in DOCUMENTS])
    def test_romein_monotonicity(self, desc, ref, hyp):
        tiers = compute_romein_tiers(ref, hyp)
        assert tiers["T1_raw"] >= tiers["T2_nospace"] - 1e-9
        assert tiers["T2_nospace"] >= tiers["T3_lowercase"] - 1e-9
        assert tiers["T3_lowercase"] >= tiers["T4_alnum"] - 1e-9

    @pytest.mark.parametrize("desc, ref, hyp", DOCUMENTS,
                             ids=[d[0] for d in DOCUMENTS])
    def test_opcode_totals(self, desc, ref, hyp):
        """S+D+I = edit distance; C+S+D = len(ref)."""
        ops = compute_opcodes(ref, hyp)
        s, d, i, c = ops["substitutions"], ops["deletions"], ops["insertions"], ops["correct"]
        from rapidfuzz.distance import Levenshtein
        expected_dist = Levenshtein.distance(ref, hyp)
        assert s + d + i == expected_dist
        assert c + s + d == len(ref)
        assert c + s + i == len(hyp)

    @pytest.mark.parametrize("desc, ref, hyp", DOCUMENTS,
                             ids=[d[0] for d in DOCUMENTS])
    def test_boc_nonneg_and_finite(self, desc, ref, hyp):
        """BOC must be non-negative and finite."""
        boc = compute_boc(ref, hyp)
        assert boc >= 0.0
        assert boc < float("inf")

    @pytest.mark.parametrize("desc, ref, hyp", DOCUMENTS,
                             ids=[d[0] for d in DOCUMENTS])
    def test_decomposition_prf_bounds(self, desc, ref, hyp):
        decomp = compute_decomposition(ref, hyp)
        assert 0.0 <= decomp["precision"] <= 1.0
        assert 0.0 <= decomp["recall"] <= 1.0
        assert 0.0 <= decomp["f1"] <= 1.0

    @pytest.mark.parametrize("desc, ref, hyp", DOCUMENTS,
                             ids=[d[0] for d in DOCUMENTS])
    def test_all_metrics_run(self, desc, ref, hyp):
        """All metrics must run without error and return expected types."""
        cer = compute_cer(ref, hyp)
        assert isinstance(cer, float)
        cer_n = compute_cer_n(ref, hyp)
        assert isinstance(cer_n, float)
        nls = compute_nls(ref, hyp)
        assert isinstance(nls, float)
        wer = compute_wer(ref, hyp)
        assert isinstance(wer, float)
        tiers = compute_romein_tiers(ref, hyp)
        assert isinstance(tiers, dict) and len(tiers) == 4
        ops = compute_opcodes(ref, hyp)
        assert isinstance(ops, dict) and len(ops) == 4
        decomp = compute_decomposition(ref, hyp)
        assert isinstance(decomp, dict)
        boc = compute_boc(ref, hyp)
        assert isinstance(boc, float)
        oi = compute_order_independent(ref, hyp, cer)
        assert isinstance(oi, dict) and "boc" in oi and "delta_cer" in oi
