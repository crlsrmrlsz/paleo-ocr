"""Cross-check semantic metrics against sklearn cosine similarity.

Strategy:
  - Cosine similarity: our compute_semantic_similarity() vs sklearn pairwise
  - NER: smoke tests + property validation
  - All tests require sentence-transformers + spaCy models (marked @semantic)
"""

import pytest

# Guard imports — these tests are skipped if models aren't available
try:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
    from sentence_transformers import SentenceTransformer

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

try:
    import spacy

    _HAS_SPACY = True
except ImportError:
    _HAS_SPACY = False


PAIRS = [
    ("identical", "El rey Don Felipe segundo de España.", "El rey Don Felipe segundo de España."),
    ("minor_ocr_error",
     "Don Juan de la Mancha, anno Domini 1542",
     "don jvan de la mancha. anno domini 1542"),
    ("moderate_error",
     "En el nombre de Dios todopoderoso amen",
     "En el nonbre de dios todo poderoso amen"),
    ("completely_different",
     "El rey mandó construir un castillo",
     "Las flores crecen en el jardín"),
    ("english_short", "The quick brown fox", "The quik brown fox"),
]


# ═══════════════════════════════════════════════
# Semantic similarity cross-check with sklearn
# ═══════════════════════════════════════════════

@pytest.mark.semantic
class TestSemanticSimilarityCrosscheck:
    """Cross-check our cosine similarity against sklearn."""

    @pytest.fixture(autouse=True)
    def _check_deps(self):
        if not _HAS_SKLEARN:
            pytest.skip("sklearn not available")

    @pytest.fixture(scope="class")
    def model(self):
        try:
            return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        except Exception:
            pytest.skip("sentence-transformers model not available")

    @pytest.mark.crosscheck
    @pytest.mark.parametrize("desc, ref, hyp", PAIRS,
                             ids=[t[0] for t in PAIRS])
    def test_cosine_vs_sklearn(self, desc, ref, hyp, model):
        """Our cosine similarity must match sklearn pairwise computation."""
        from metrics.semantic import compute_semantic_similarity

        our_sim = compute_semantic_similarity(ref, hyp)

        # Independent computation via sklearn
        embeddings = model.encode([ref, hyp], convert_to_numpy=True)
        sklearn_sim = sklearn_cosine([embeddings[0]], [embeddings[1]])[0][0]

        assert abs(our_sim - float(sklearn_sim)) < 1e-4, (
            f"Cosine mismatch: ours={our_sim:.6f}, sklearn={sklearn_sim:.6f}"
        )

    def test_identity_similarity(self, model):
        """Identical text must have similarity ≈ 1.0."""
        from metrics.semantic import compute_semantic_similarity

        sim = compute_semantic_similarity(
            "El rey Don Felipe segundo de España.",
            "El rey Don Felipe segundo de España.",
        )
        assert sim > 0.99, f"Identity similarity too low: {sim}"

    def test_empty_both(self, model):
        """Both empty → 1.0 by convention."""
        from metrics.semantic import compute_semantic_similarity
        assert compute_semantic_similarity("", "") == 1.0

    def test_empty_one(self, model):
        """One empty → 0.0 by convention."""
        from metrics.semantic import compute_semantic_similarity
        assert compute_semantic_similarity("", "hello") == 0.0
        assert compute_semantic_similarity("hello", "") == 0.0

    @pytest.mark.parametrize("desc, ref, hyp", PAIRS,
                             ids=[t[0] for t in PAIRS])
    def test_similarity_bounded(self, desc, ref, hyp, model):
        """Similarity should be in [-1, 1], typically [0, 1] for similar texts."""
        from metrics.semantic import compute_semantic_similarity

        sim = compute_semantic_similarity(ref, hyp)
        assert -1.0 <= sim <= 1.0 + 1e-6, f"Similarity out of bounds: {sim}"

    def test_ocr_error_high_similarity(self, model):
        """Minor OCR errors should still yield high similarity (> 0.8)."""
        from metrics.semantic import compute_semantic_similarity

        sim = compute_semantic_similarity(
            "Don Juan de la Mancha, anno Domini 1542",
            "don jvan de la mancha. anno domini 1542",
        )
        assert sim > 0.8, f"OCR error similarity too low: {sim}"


# ═══════════════════════════════════════════════
# NER metrics smoke tests
# ═══════════════════════════════════════════════

@pytest.mark.semantic
class TestNERMetrics:
    """Smoke tests for NER extraction and fuzzy matching."""

    @pytest.fixture(autouse=True)
    def _check_deps(self):
        if not _HAS_SPACY:
            pytest.skip("spaCy not available")
        try:
            spacy.load("es_core_news_lg")
        except OSError:
            pytest.skip("es_core_news_lg model not available")

    def test_ner_empty_both(self):
        from metrics.semantic import compute_ner_metrics
        result = compute_ner_metrics("", "")
        assert result["ner_precision"] == 1.0
        assert result["ner_recall"] == 1.0
        assert result["ner_f1"] == 1.0

    def test_ner_prf_bounds(self):
        """NER P/R/F1 must be in [0, 1]."""
        from metrics.semantic import compute_ner_metrics

        for desc, ref, hyp in PAIRS:
            result = compute_ner_metrics(ref, hyp)
            assert 0.0 <= result["ner_precision"] <= 1.0
            assert 0.0 <= result["ner_recall"] <= 1.0
            assert 0.0 <= result["ner_f1"] <= 1.0

    def test_ner_counts_nonneg(self):
        """Entity counts must be non-negative integers."""
        from metrics.semantic import compute_ner_metrics

        result = compute_ner_metrics(
            "Don Juan de la Mancha viajó a Sevilla en 1542",
            "don jvan de la mancha viajo a sevilla en 1542",
        )
        assert result["ref_entities"] >= 0
        assert result["hyp_entities"] >= 0
        assert result["matched_entities"] >= 0
        assert result["matched_entities"] <= min(result["ref_entities"], result["hyp_entities"])

    def test_ner_identical_text(self):
        """Identical text should have perfect or near-perfect NER scores."""
        from metrics.semantic import compute_ner_metrics

        text = "El rey Felipe II ordenó la construcción del Escorial en Madrid."
        result = compute_ner_metrics(text, text)
        assert result["ner_precision"] >= 0.99
        assert result["ner_recall"] >= 0.99

    def test_all_semantic_keys(self):
        """compute_all_semantic must return all expected keys."""
        from metrics.semantic import compute_all_semantic

        result = compute_all_semantic(
            "Don Juan de la Mancha", "don jvan de la mancha"
        )
        expected_keys = {
            "semantic_similarity",
            "ner_precision", "ner_recall", "ner_f1",
            "ref_entities", "hyp_entities", "matched_entities",
        }
        assert set(result.keys()) == expected_keys
