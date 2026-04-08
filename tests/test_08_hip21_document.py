"""HIP'21 IMPACT dataset — document-level validation with published CER values.

Uses 5 selected documents from the HIP'21 companion dataset:
  Neudecker, C. et al. (2021). "A survey of OCR evaluation tools and metrics."
  Proc. 6th International Workshop on Historical Document Imaging and Processing
  (HIP'21), ACM. DOI: 10.1145/3476887.3476888

Each document has:
  - Ground truth in PAGE XML (.gt.xml)
  - OCR output in ALTO XML from GT4HistOCR (.gt4hist.xml)
  - Published CER/WER from dinglehopper (.gt4hist.dinglehopper.json)

Validation strategy:
  1. Our CER vs dinglehopper published CER → exact match (proves same formula)
  2. Our CER vs jiwer.cer() on same text    → exact match (independent cross-check)
  3. Romein monotonicity T1 ≥ T2 ≥ T3 ≥ T4  → verified on all 5 documents
  4. Opcode invariants S+D+I=ed, C+S+D=len(ref) → verified on all 5 documents

Text extraction uses dinglehopper's own extractors (page_text, alto_text) because
the IMPACT PAGE XML contains Unicode PUA codepoints for Fraktur ligatures that
require dinglehopper's glyph decomposition to resolve to standard Unicode.
"""

import json
from pathlib import Path

import pytest
import editdistance as ed

from metrics.core import compute_cer, compute_cer_n, compute_nls, compute_wer
from metrics.romein import compute_romein_tiers
from metrics.decomposition import compute_opcodes, compute_decomposition
from metrics.order_independent import compute_boc, compute_order_independent

# ---------------------------------------------------------------------------
# Optional dependencies — skip entire module if unavailable
# ---------------------------------------------------------------------------
try:
    from lxml import etree
    from dinglehopper import page_text, alto_text
    _HAS_DINGLEHOPPER = True
except ImportError:
    _HAS_DINGLEHOPPER = False

try:
    import jiwer
    from jiwer.transforms import Compose, ReduceToListOfListOfChars
    _HAS_JIWER = True
except ImportError:
    _HAS_JIWER = False


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "hip21"

# HIP'21 document metadata: (id, language, description)
HIP21_DOCS = [
    ("00046949", "deu", "German Fraktur — religious text"),
    ("00046910", "deu", "German Fraktur — narrative prose"),
    ("00525436", "eng", "English — 17th century preface"),
    ("00451938", "fra", "French — theological treatise"),
    ("00539327", "nld", "Dutch — political pamphlet"),
]


def _load_hip21_document(doc_id: str):
    """Load GT text, OCR text, and published metrics for a HIP'21 document."""
    gt_path = FIXTURES_DIR / f"{doc_id}.gt.xml"
    ocr_path = FIXTURES_DIR / f"{doc_id}.gt4hist.xml"
    json_path = FIXTURES_DIR / f"{doc_id}.gt4hist.dinglehopper.json"

    if not gt_path.exists():
        pytest.skip(f"HIP'21 fixture {doc_id} not found; run download_fixtures.py")

    gt_text = page_text(etree.parse(str(gt_path)))
    ocr_text = alto_text(etree.parse(str(ocr_path)))

    with open(json_path) as f:
        published = json.load(f)

    return gt_text, ocr_text, published


# ---------------------------------------------------------------------------
# Build parameterized test data
# ---------------------------------------------------------------------------
def _hip21_params():
    """Generate pytest params, skipping if fixtures or dinglehopper unavailable."""
    if not _HAS_DINGLEHOPPER:
        return []
    if not FIXTURES_DIR.exists():
        return []
    return [
        pytest.param(doc_id, lang, desc, id=f"{doc_id}_{lang}")
        for doc_id, lang, desc in HIP21_DOCS
    ]


HIP21_PARAMS = _hip21_params()

skip_no_dinglehopper = pytest.mark.skipif(
    not _HAS_DINGLEHOPPER,
    reason="dinglehopper or lxml not installed",
)


# ═══════════════════════════════════════════════
# CER cross-check against published dinglehopper values
# ═══════════════════════════════════════════════

@pytest.mark.document
@skip_no_dinglehopper
class TestHIP21CERPublished:
    """Our CER must match dinglehopper's published CER for each document.

    Both use the same formula: CER = edit_distance(ref, hyp) / len(ref).
    Since we use dinglehopper's own text extraction, we get the same
    (ref, hyp) strings and thus the same edit distance and CER.
    """

    @pytest.mark.parametrize("doc_id, lang, desc", HIP21_PARAMS)
    def test_cer_matches_published(self, doc_id, lang, desc):
        """Our CER matches dinglehopper's published CER (< 0.001 tolerance)."""
        gt, ocr, pub = _load_hip21_document(doc_id)
        our_cer = compute_cer(gt, ocr)
        pub_cer = pub["cer"]
        assert abs(our_cer - pub_cer) < 0.001, (
            f"CER mismatch for {doc_id}: ours={our_cer:.6f}, published={pub_cer:.6f}"
        )

    @pytest.mark.parametrize("doc_id, lang, desc", HIP21_PARAMS)
    def test_gt_length_matches_published(self, doc_id, lang, desc):
        """Extracted GT length matches published n_characters."""
        gt, _, pub = _load_hip21_document(doc_id)
        assert len(gt) == pub["n_characters"], (
            f"GT length {len(gt)} != published {pub['n_characters']}"
        )


# ═══════════════════════════════════════════════
# CER cross-check against jiwer (independent library)
# ═══════════════════════════════════════════════

@pytest.mark.document
@skip_no_dinglehopper
@pytest.mark.skipif(not _HAS_JIWER, reason="jiwer not installed")
class TestHIP21CERJiwer:
    """Our CER must match jiwer.cer() on the SAME extracted text.

    jiwer's default CER applies a Strip transform that removes leading/
    trailing whitespace, changing the edit distance.  We use the raw
    character decomposition (no strip) for exact comparison.
    """

    # jiwer transform without Strip — pure character-level CER
    _NO_STRIP = Compose([ReduceToListOfListOfChars()])

    @pytest.mark.parametrize("doc_id, lang, desc", HIP21_PARAMS)
    def test_cer_matches_jiwer(self, doc_id, lang, desc):
        """Our CER matches jiwer CER (no-strip transform) exactly."""
        gt, ocr, _ = _load_hip21_document(doc_id)
        our_cer = compute_cer(gt, ocr)
        jiwer_cer = jiwer.cer(
            gt, ocr,
            reference_transform=self._NO_STRIP,
            hypothesis_transform=self._NO_STRIP,
        )
        assert abs(our_cer - jiwer_cer) < 1e-9, (
            f"CER vs jiwer for {doc_id}: ours={our_cer:.12f}, jiwer={jiwer_cer:.12f}"
        )


# ═══════════════════════════════════════════════
# Romein 4-tier monotonicity
# ═══════════════════════════════════════════════

@pytest.mark.document
@skip_no_dinglehopper
class TestHIP21RomeinMonotonicity:
    """T1 ≥ T2 ≥ T3 ≥ T4 on all 5 HIP'21 documents."""

    @pytest.mark.parametrize("doc_id, lang, desc", HIP21_PARAMS)
    def test_tier_monotonicity(self, doc_id, lang, desc):
        gt, ocr, _ = _load_hip21_document(doc_id)
        tiers = compute_romein_tiers(gt, ocr)
        assert tiers["T1_raw"] >= tiers["T2_nospace"] - 1e-9, (
            f"T1 < T2: {tiers['T1_raw']:.6f} < {tiers['T2_nospace']:.6f}"
        )
        assert tiers["T2_nospace"] >= tiers["T3_lowercase"] - 1e-9, (
            f"T2 < T3: {tiers['T2_nospace']:.6f} < {tiers['T3_lowercase']:.6f}"
        )
        assert tiers["T3_lowercase"] >= tiers["T4_alnum"] - 1e-9, (
            f"T3 < T4: {tiers['T3_lowercase']:.6f} < {tiers['T4_alnum']:.6f}"
        )


# ═══════════════════════════════════════════════
# Opcode invariants
# ═══════════════════════════════════════════════

@pytest.mark.document
@skip_no_dinglehopper
class TestHIP21OpcodeInvariants:
    """Algebraic invariants must hold on all 5 HIP'21 documents."""

    @pytest.mark.parametrize("doc_id, lang, desc", HIP21_PARAMS)
    def test_sdi_equals_edit_distance(self, doc_id, lang, desc):
        """S + D + I = Levenshtein edit distance."""
        gt, ocr, _ = _load_hip21_document(doc_id)
        ops = compute_opcodes(gt, ocr)
        sdi = ops["substitutions"] + ops["deletions"] + ops["insertions"]
        expected = ed.eval(gt, ocr)
        assert sdi == expected, f"S+D+I={sdi}, edit_distance={expected}"

    @pytest.mark.parametrize("doc_id, lang, desc", HIP21_PARAMS)
    def test_csd_equals_ref_length(self, doc_id, lang, desc):
        """C + S + D = len(reference)."""
        gt, ocr, _ = _load_hip21_document(doc_id)
        ops = compute_opcodes(gt, ocr)
        csd = ops["correct"] + ops["substitutions"] + ops["deletions"]
        assert csd == len(gt), f"C+S+D={csd}, len(ref)={len(gt)}"

    @pytest.mark.parametrize("doc_id, lang, desc", HIP21_PARAMS)
    def test_csi_equals_hyp_length(self, doc_id, lang, desc):
        """C + S + I = len(hypothesis)."""
        gt, ocr, _ = _load_hip21_document(doc_id)
        ops = compute_opcodes(gt, ocr)
        csi = ops["correct"] + ops["substitutions"] + ops["insertions"]
        assert csi == len(ocr), f"C+S+I={csi}, len(hyp)={len(ocr)}"


# ═══════════════════════════════════════════════
# Full metric pipeline on HIP'21 documents
# ═══════════════════════════════════════════════

@pytest.mark.document
@skip_no_dinglehopper
class TestHIP21FullPipeline:
    """All metrics must run without error and satisfy cross-consistency."""

    @pytest.mark.parametrize("doc_id, lang, desc", HIP21_PARAMS)
    def test_all_metrics_run(self, doc_id, lang, desc):
        """Every metric function returns the expected type."""
        gt, ocr, _ = _load_hip21_document(doc_id)
        assert isinstance(compute_cer(gt, ocr), float)
        assert isinstance(compute_cer_n(gt, ocr), float)
        assert isinstance(compute_nls(gt, ocr), float)
        assert isinstance(compute_wer(gt, ocr), float)
        tiers = compute_romein_tiers(gt, ocr)
        assert isinstance(tiers, dict) and len(tiers) == 4
        ops = compute_opcodes(gt, ocr)
        assert isinstance(ops, dict) and len(ops) == 4
        decomp = compute_decomposition(gt, ocr)
        assert isinstance(decomp, dict)
        boc = compute_boc(gt, ocr)
        assert isinstance(boc, float)
        oi = compute_order_independent(gt, ocr, compute_cer(gt, ocr))
        assert isinstance(oi, dict)

    @pytest.mark.parametrize("doc_id, lang, desc", HIP21_PARAMS)
    def test_cer_n_leq_cer(self, doc_id, lang, desc):
        """CER_n ≤ CER when CER ≤ 1.0."""
        gt, ocr, _ = _load_hip21_document(doc_id)
        cer = compute_cer(gt, ocr)
        cer_n = compute_cer_n(gt, ocr)
        if cer <= 1.0:
            assert cer_n <= cer + 1e-9, f"CER_n > CER: {cer_n} > {cer}"

    @pytest.mark.parametrize("doc_id, lang, desc", HIP21_PARAMS)
    def test_nls_bounds(self, doc_id, lang, desc):
        """NLS ∈ [0, 1]."""
        gt, ocr, _ = _load_hip21_document(doc_id)
        nls = compute_nls(gt, ocr)
        assert 0.0 <= nls <= 1.0, f"NLS out of bounds: {nls}"

    @pytest.mark.parametrize("doc_id, lang, desc", HIP21_PARAMS)
    def test_decomposition_prf_bounds(self, doc_id, lang, desc):
        """P, R, F1 ∈ [0, 1]."""
        gt, ocr, _ = _load_hip21_document(doc_id)
        decomp = compute_decomposition(gt, ocr)
        assert 0.0 <= decomp["precision"] <= 1.0
        assert 0.0 <= decomp["recall"] <= 1.0
        assert 0.0 <= decomp["f1"] <= 1.0

    @pytest.mark.parametrize("doc_id, lang, desc", HIP21_PARAMS)
    def test_boc_nonneg(self, doc_id, lang, desc):
        """BOC ≥ 0."""
        gt, ocr, _ = _load_hip21_document(doc_id)
        boc = compute_boc(gt, ocr)
        assert boc >= 0.0, f"BOC negative: {boc}"
