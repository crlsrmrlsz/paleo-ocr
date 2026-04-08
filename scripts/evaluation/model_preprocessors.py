"""Model-specific output preprocessing.

Each HTR model produces output in a different format. This module provides
a registry of preprocessing functions that extract the transcription text
from model-specific wrappers, metadata, and formatting before generic
normalization is applied.

To add a new model:
  1. Write a preprocess_<model>(text) function
  2. Add it to MODEL_PREPROCESSORS dict
"""

import re


def preprocess_churro(text: str) -> str:
    """Strip CHURRO-3B's metadata header.

    CHURRO outputs XML with metadata elements (language, direction, material,
    description, script). The notebook strips XML tags, but the metadata text
    content remains as a 5-line header:
        Spanish / ltr / [material] / [description] / Latin
    """
    lines = text.split("\n")
    if not lines or lines[0].strip() != "Spanish":
        return text
    # Find "Latin" line within the first 10 lines (header is typically 5 lines)
    for i in range(2, min(10, len(lines))):
        if lines[i].strip() == "Latin":
            return "\n".join(lines[i + 1 :])
    return text


def strip_xml_tags(text: str) -> str:
    """Remove XML/HTML tags from text. Generic fallback for all models."""
    return re.sub(r"<[^>]+>", "", text)


# --- Registry ---
MODEL_PREPROCESSORS: dict[str, callable] = {
    "churro_3b": preprocess_churro,
}


def preprocess(text: str, model_name: str) -> str:
    """Dispatch to model-specific preprocessor, then apply generic cleaners."""
    fn = MODEL_PREPROCESSORS.get(model_name)
    if fn:
        text = fn(text)
    # Generic: strip any remaining XML/HTML tags (safe no-op if none)
    text = strip_xml_tags(text)
    return text
