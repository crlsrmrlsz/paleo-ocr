"""Shared fixtures, path setup, and markers for the paleo-ocr test suite."""

import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path setup: ensure scripts/evaluation and scripts are importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
EVAL_DIR = SCRIPTS_DIR / "evaluation"

for p in (str(EVAL_DIR), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------
def pytest_configure(config):
    config.addinivalue_line("markers", "crosscheck: library cross-check tests")
    config.addinivalue_line("markers", "document: document-level cross-validation tests")
    config.addinivalue_line("markers", "network: tests that download external data")
    config.addinivalue_line("markers", "semantic: tests requiring sentence-transformers/spacy models")


# ---------------------------------------------------------------------------
# Shared text pairs for cross-check tests
# ---------------------------------------------------------------------------
TEXT_PAIRS = [
    # (description, reference, hypothesis)
    ("identical", "hello world", "hello world"),
    ("single_sub", "kitten", "sitting"),
    ("single_char", "a", "b"),
    ("pure_deletion", "abcde", "ace"),
    ("pure_insertion", "abc", "aXbYcZ"),
    ("empty_both", "", ""),
    ("empty_ref", "", "some text"),
    ("empty_hyp", "some text", ""),
    ("completely_different", "abc", "xyz"),
    ("longer_hyp", "X", "X X Y Y"),
    ("unicode_diacritics", "café résumé naïve", "cafe resume naive"),
    ("spanish_paleographic",
     "Don Juan de la Mancha, anno Domini 1542",
     "don jvan de la mancha. anno domini 1542"),
    ("long_text",
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
    ("fraktur_long_s",
     "das Verſproene glei den Augenblick",
     "das Verfproene glei den Augemblick"),
    ("whitespace_variants", "hello\tworld  foo", "hello world foo"),
    ("repeated_chars", "aaaa", "bbbb"),
]


@pytest.fixture(params=TEXT_PAIRS, ids=[t[0] for t in TEXT_PAIRS])
def text_pair(request):
    """Yield (description, reference, hypothesis) tuples."""
    return request.param


# Subset that excludes empty-ref cases (where some metrics diverge by convention)
TEXT_PAIRS_NONEMPTY = [t for t in TEXT_PAIRS if t[1] != ""]


@pytest.fixture(params=TEXT_PAIRS_NONEMPTY, ids=[t[0] for t in TEXT_PAIRS_NONEMPTY])
def text_pair_nonempty(request):
    """Yield text pairs where reference is non-empty."""
    return request.param


# ---------------------------------------------------------------------------
# Fixture availability helpers
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def fixtures_dir():
    """Return the fixtures directory path, skip if not populated."""
    if not FIXTURES_DIR.exists():
        pytest.skip("Fixtures directory not found; run download_fixtures.py first")
    return FIXTURES_DIR


def _check_semantic_models():
    """Check if sentence-transformers and spaCy models are available."""
    try:
        from sentence_transformers import SentenceTransformer
        SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    except Exception:
        return False
    try:
        import spacy
        spacy.load("es_core_news_lg")
    except Exception:
        return False
    return True


@pytest.fixture(scope="session")
def semantic_models_available():
    """Skip test if semantic models aren't downloaded."""
    if not _check_semantic_models():
        pytest.skip("Semantic models not available (sentence-transformers / spaCy)")
    return True
