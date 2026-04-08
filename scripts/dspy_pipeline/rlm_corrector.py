"""RLM abbreviation lookup for per-strip correction.

Uses dspy.RLM to programmatically search a flat abbreviation dictionary
and expand abbreviated forms found in the strip transcription.
The abbreviation dict is a REPL variable (model sees 500-char preview,
searches via Python code). The transcription is also a REPL variable.
"""

import dspy

from .config import RLM_MAX_ITERATIONS, RLM_MAX_LLM_CALLS
from .domain_knowledge import build_flat_abbreviations


def create_rlm_corrector():
    """Create RLM module pre-loaded with the flat abbreviation dict.

    Returns a callable: corrector(strip_transcription) -> Prediction
    """
    abbreviations = build_flat_abbreviations()

    rlm = dspy.RLM(
        "strip_transcription, abbreviations -> corrected_transcription",
        max_iterations=RLM_MAX_ITERATIONS,
        max_llm_calls=RLM_MAX_LLM_CALLS,
    )

    class BoundRLMCorrector:
        def __init__(self, rlm_module, abbrev_data):
            self.rlm = rlm_module
            self.abbrev_data = abbrev_data

        def __call__(self, strip_transcription):
            return self.rlm(
                strip_transcription=strip_transcription,
                abbreviations=self.abbrev_data,
            )

    return BoundRLMCorrector(rlm, abbreviations)
