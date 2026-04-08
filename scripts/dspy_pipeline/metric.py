"""DSPy-compatible CER metric for HTR evaluation.

Supports both MIPROv2 (3-arg) and GEPA (5-arg) calling conventions.
When called with pred_name (GEPA), returns ScoreWithFeedback with
textual feedback for the reflection LM.
"""

import re

import editdistance


def _normalize_whitespace(text: str) -> str:
    """Collapse all whitespace (including newlines) to single spaces.

    Matches the normalization in scripts/evaluation/compute_metrics.py.
    CER should measure transcription accuracy, not line segmentation.
    """
    return re.sub(r"\s+", " ", text).strip()


def cer_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """1 - CER (higher = better). DSPy maximizes metrics, so we invert CER.

    GEPA calls with 5 args and expects ScoreWithFeedback when pred_name
    is provided, giving the reflection LM actionable feedback.
    """
    gt = _normalize_whitespace(example.transcription)
    pred = _normalize_whitespace(prediction.transcription)
    if not gt:
        score = 1.0 if not pred else 0.0
    else:
        cer = editdistance.eval(pred, gt) / len(gt)
        score = max(0.0, 1.0 - cer)

    # GEPA reflection: return score + textual feedback
    if pred_name is not None:
        from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

        cer_pct = (1.0 - score) * 100
        if score >= 0.8:
            feedback = f"CER {cer_pct:.0f}%. Good transcription, minor errors."
        elif score >= 0.5:
            feedback = (
                f"CER {cer_pct:.0f}%. Moderate errors — check abbreviation "
                "expansion, character forms (u/v, i/y, ç/z), and spelling "
                "modernization (preserve original orthography)."
            )
        else:
            feedback = (
                f"CER {cer_pct:.0f}%. High error rate — likely misreading "
                "letterforms or hallucinating content not on the page. "
                "Transcribe only what is visible."
            )
        return ScoreWithFeedback(score=score, feedback=feedback)

    return score
