"""Semantic metrics: embedding similarity and NER capture.

These metrics measure meaning preservation beyond character-level accuracy.
Models and data are loaded lazily on first call for efficiency.
"""

import sys
from pathlib import Path

# Resolve pipeline config
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
from pipeline_config import NER_MATCH_THRESHOLD, SEMANTIC_MODEL_NAME, SPACY_MODEL_NAME

# Lazy-loaded global singletons
_sentence_model = None
_nlp = None


def _get_sentence_model():
    """Lazy-load sentence-transformers model (loaded once, reused)."""
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer
        _sentence_model = SentenceTransformer(SEMANTIC_MODEL_NAME)
    return _sentence_model


def _get_spacy_nlp():
    """Lazy-load spaCy NER model (loaded once, reused)."""
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load(SPACY_MODEL_NAME)
        except OSError:
            print(
                f"spaCy model '{SPACY_MODEL_NAME}' not found. "
                f"Install with: python -m spacy download {SPACY_MODEL_NAME}",
                file=sys.stderr,
            )
            raise
    return _nlp


def compute_semantic_similarity(reference: str, hypothesis: str) -> float:
    """Cosine similarity between sentence embeddings.

    Uses paraphrase-multilingual-MiniLM-L12-v2 for Spanish text.
    Returns a float in [-1, 1], typically [0, 1] for similar texts.
    """
    if not reference and not hypothesis:
        return 1.0
    if not reference or not hypothesis:
        return 0.0

    model = _get_sentence_model()
    embeddings = model.encode([reference, hypothesis], convert_to_numpy=True)

    # Cosine similarity
    from numpy import dot
    from numpy.linalg import norm
    cos_sim = dot(embeddings[0], embeddings[1]) / (norm(embeddings[0]) * norm(embeddings[1]))
    return float(cos_sim)


def _fuzzy_match_entities(
    ref_entities: list[dict],
    hyp_entities: list[dict],
    threshold: float = NER_MATCH_THRESHOLD,
) -> dict:
    """Match NER entities between reference and hypothesis using fuzzy NLS.

    Each reference entity is matched to the best hypothesis entity (greedy,
    without replacement). A match requires NLS > threshold.
    """
    from rapidfuzz.distance import Levenshtein

    matched = 0
    used_hyp = set()

    for ref_ent in ref_entities:
        best_score = 0.0
        best_idx = -1
        ref_text = ref_ent["text"].lower()

        for idx, hyp_ent in enumerate(hyp_entities):
            if idx in used_hyp:
                continue
            # Only match same entity label
            if hyp_ent["label"] != ref_ent["label"]:
                continue

            hyp_text = hyp_ent["text"].lower()
            max_len = max(len(ref_text), len(hyp_text))
            if max_len == 0:
                continue
            nls = 1.0 - Levenshtein.distance(ref_text, hyp_text) / max_len

            if nls > best_score:
                best_score = nls
                best_idx = idx

        if best_score > threshold and best_idx >= 0:
            matched += 1
            used_hyp.add(best_idx)

    return {
        "matched": matched,
        "ref_total": len(ref_entities),
        "hyp_total": len(hyp_entities),
    }


def compute_ner_metrics(reference: str, hypothesis: str) -> dict:
    """Extract named entities and compute fuzzy-matched P/R/F1.

    Uses spaCy es_core_news_lg for NER, then fuzzy-matches entities
    between ref and hyp with NLS > NER_MATCH_THRESHOLD.
    """
    if not reference and not hypothesis:
        return {
            "ner_precision": 1.0, "ner_recall": 1.0, "ner_f1": 1.0,
            "ref_entities": 0, "hyp_entities": 0, "matched_entities": 0,
        }

    nlp = _get_spacy_nlp()

    ref_doc = nlp(reference)
    hyp_doc = nlp(hypothesis)

    ref_entities = [{"text": ent.text, "label": ent.label_} for ent in ref_doc.ents]
    hyp_entities = [{"text": ent.text, "label": ent.label_} for ent in hyp_doc.ents]

    if not ref_entities and not hyp_entities:
        return {
            "ner_precision": 1.0, "ner_recall": 1.0, "ner_f1": 1.0,
            "ref_entities": 0, "hyp_entities": 0, "matched_entities": 0,
        }

    match_result = _fuzzy_match_entities(ref_entities, hyp_entities)
    matched = match_result["matched"]

    precision = matched / len(hyp_entities) if hyp_entities else 0.0
    recall = matched / len(ref_entities) if ref_entities else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "ner_precision": round(precision, 6),
        "ner_recall": round(recall, 6),
        "ner_f1": round(f1, 6),
        "ref_entities": len(ref_entities),
        "hyp_entities": len(hyp_entities),
        "matched_entities": matched,
    }


def compute_all_semantic(reference: str, hypothesis: str) -> dict:
    """Compute all semantic metrics."""
    sim = compute_semantic_similarity(reference, hypothesis)
    ner = compute_ner_metrics(reference, hypothesis)
    return {
        "semantic_similarity": round(sim, 6),
        **ner,
    }
