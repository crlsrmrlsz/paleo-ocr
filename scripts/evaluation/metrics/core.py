"""Core HTR evaluation metrics: CER, CER_n, NLS, WER, SER."""

import editdistance
from jiwer import wer as jiwer_wer
from rapidfuzz.distance import Levenshtein


def compute_cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate = edit_distance / len(reference).

    Unbounded (can exceed 1.0 if hypothesis is much longer than reference).
    """
    if not reference:
        return 1.0 if hypothesis else 0.0
    return editdistance.eval(reference, hypothesis) / len(reference)


def compute_cer_n(reference: str, hypothesis: str) -> float:
    """Bounded CER (OCR-D convention) = (S+D+I) / (S+D+I+C).

    Always in [0, 1]. Uses rapidfuzz opcodes for the edit script.
    """
    if not reference and not hypothesis:
        return 0.0
    if not reference:
        return 1.0

    opcodes = Levenshtein.opcodes(reference, hypothesis)
    s = d = i = c = 0
    for tag, ref_start, ref_end, hyp_start, hyp_end in opcodes:
        if tag == "equal":
            c += ref_end - ref_start
        elif tag == "replace":
            s += max(ref_end - ref_start, hyp_end - hyp_start)
        elif tag == "delete":
            d += ref_end - ref_start
        elif tag == "insert":
            i += hyp_end - hyp_start

    total = s + d + i + c
    if total == 0:
        return 0.0
    return (s + d + i) / total


def compute_nls(reference: str, hypothesis: str) -> float:
    """Normalized Levenshtein Similarity = 1 - dist/max(len_ref, len_hyp).

    Bounded [0, 1]. Used in the CHURRO benchmark.
    """
    if not reference and not hypothesis:
        return 1.0
    max_len = max(len(reference), len(hypothesis))
    if max_len == 0:
        return 1.0
    dist = Levenshtein.distance(reference, hypothesis)
    return 1.0 - dist / max_len


def compute_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate using jiwer."""
    if not reference.strip():
        return 1.0 if hypothesis.strip() else 0.0
    return jiwer_wer(reference, hypothesis)


def compute_ser(reference_lines: list[str], hypothesis_lines: list[str]) -> float:
    """Sentence Error Rate = fraction of lines that differ.

    Compares line-by-line; extra/missing lines count as errors.
    """
    if not reference_lines:
        return 1.0 if hypothesis_lines else 0.0

    max_lines = max(len(reference_lines), len(hypothesis_lines))
    errors = 0
    for idx in range(max_lines):
        ref_line = reference_lines[idx] if idx < len(reference_lines) else ""
        hyp_line = hypothesis_lines[idx] if idx < len(hypothesis_lines) else ""
        if ref_line != hyp_line:
            errors += 1

    return errors / max_lines


def compute_all_core(reference: str, hypothesis: str) -> dict:
    """Compute all core metrics in one call."""
    ref_lines = reference.split("\n") if reference else []
    hyp_lines = hypothesis.split("\n") if hypothesis else []

    return {
        "cer": round(compute_cer(reference, hypothesis), 6),
        "cer_n": round(compute_cer_n(reference, hypothesis), 6),
        "nls": round(compute_nls(reference, hypothesis), 6),
        "wer": round(compute_wer(reference, hypothesis), 6),
        "ser": round(compute_ser(ref_lines, hyp_lines), 6),
    }
