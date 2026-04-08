"""Error decomposition from edit opcodes: Precision, Recall, F1, S/D/I rates."""

from rapidfuzz.distance import Levenshtein


def compute_opcodes(reference: str, hypothesis: str) -> dict:
    """Extract substitution, deletion, insertion, and correct counts from opcodes."""
    if not reference and not hypothesis:
        return {"substitutions": 0, "deletions": 0, "insertions": 0, "correct": 0}

    opcodes = Levenshtein.opcodes(reference, hypothesis)
    s = d = i = c = 0
    for tag, ref_start, ref_end, hyp_start, hyp_end in opcodes:
        if tag == "equal":
            c += ref_end - ref_start
        elif tag == "replace":
            # Substitutions = min of the two spans; excess is I or D
            ref_len = ref_end - ref_start
            hyp_len = hyp_end - hyp_start
            s += min(ref_len, hyp_len)
            if ref_len > hyp_len:
                d += ref_len - hyp_len
            else:
                i += hyp_len - ref_len
        elif tag == "delete":
            d += ref_end - ref_start
        elif tag == "insert":
            i += hyp_end - hyp_start

    return {"substitutions": s, "deletions": d, "insertions": i, "correct": c}


def compute_decomposition(reference: str, hypothesis: str) -> dict:
    """Full error decomposition: P/R/F1 and per-operation error rates.

    Precision = C/(C+S+I) — penalizes hallucinated/wrong chars
    Recall    = C/(C+S+D) — penalizes missed reference chars
    F1        = harmonic mean
    S_rate, D_rate, I_rate — each normalized by N_ref
    """
    ops = compute_opcodes(reference, hypothesis)
    s, d, i, c = ops["substitutions"], ops["deletions"], ops["insertions"], ops["correct"]

    n_ref = len(reference) if reference else 1

    # Precision: of all characters the model produced, how many are correct?
    precision_denom = c + s + i
    precision = c / precision_denom if precision_denom > 0 else 0.0

    # Recall: of all reference characters, how many were correctly produced?
    recall_denom = c + s + d
    recall = c / recall_denom if recall_denom > 0 else 0.0

    # F1
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "s_rate": round(s / n_ref, 6),
        "d_rate": round(d / n_ref, 6),
        "i_rate": round(i / n_ref, 6),
        "substitutions": s,
        "deletions": d,
        "insertions": i,
        "correct": c,
    }
