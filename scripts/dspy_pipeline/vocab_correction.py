"""Vocabulary-guided post-correction for HTR transcriptions.

Provides tools to flag unknown words in a transcription and suggest
corrections from a combined historical-Spanish vocabulary.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from scripts.dspy_pipeline.config import DATA_DIR

# ---------------------------------------------------------------------------
# Module-level caches
# ---------------------------------------------------------------------------
_COMBINED_VOCAB: Optional[set[str]] = None
_ABBREVIATIONS: Optional[dict[str, str]] = None
_PREFIX_INDEX: Optional[dict[str, list[str]]] = None

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_combined_vocab() -> set[str]:
    """Load and merge word forms from all three vocabulary JSON files.

    Files used:
    - data/vocabularies/impact_es_lexicon.json   -- word_forms: list[str]
    - data/vocabularies/codea_paleographic_vocab.json -- word_forms: list[str]
    - data/vocabularies/charta_vocab.json         -- word_forms: dict (keys are forms)

    Returns a lowercase set of all word forms, cached after first call.
    """
    global _COMBINED_VOCAB
    if _COMBINED_VOCAB is not None:
        return _COMBINED_VOCAB

    vocab_dir = DATA_DIR / "vocabularies"
    combined: set[str] = set()

    # impact_es_lexicon — list
    impact_path = vocab_dir / "impact_es_lexicon.json"
    with open(impact_path, encoding="utf-8") as fh:
        data = json.load(fh)
    for w in data["word_forms"]:
        combined.add(w.lower())

    # codea_paleographic_vocab — list
    codea_path = vocab_dir / "codea_paleographic_vocab.json"
    with open(codea_path, encoding="utf-8") as fh:
        data = json.load(fh)
    for w in data["word_forms"]:
        combined.add(w.lower())

    # charta_vocab — dict (keys are word forms)
    charta_path = vocab_dir / "charta_vocab.json"
    with open(charta_path, encoding="utf-8") as fh:
        data = json.load(fh)
    for w in data["word_forms"]:
        combined.add(w.lower())

    _COMBINED_VOCAB = combined
    return _COMBINED_VOCAB


def load_abbreviations() -> dict[str, str]:
    """Load abbreviation expansions from academic_abbreviations.json.

    Reads data/abbreviations/academic_abbreviations.json. The file has an
    ``abbreviations`` key containing a list of dicts with ``abbreviated`` and
    ``expanded`` keys.  When ``expanded`` contains ``/`` (multiple options),
    the first option is used.

    Returns a cached dict mapping abbreviated form → expanded form.
    """
    global _ABBREVIATIONS
    if _ABBREVIATIONS is not None:
        return _ABBREVIATIONS

    abbrev_path = DATA_DIR / "abbreviations" / "academic_abbreviations.json"
    with open(abbrev_path, encoding="utf-8") as fh:
        data = json.load(fh)

    abbrevs: dict[str, str] = {}
    for entry in data["abbreviations"]:
        abbr = entry["abbreviated"].lower()
        expanded = entry["expanded"]
        # Take first option when multiple are listed (e.g. "qual/cual" → "qual")
        first = expanded.split("/")[0].strip().lower()
        abbrevs[abbr] = first

    _ABBREVIATIONS = abbrevs
    return _ABBREVIATIONS

# ---------------------------------------------------------------------------
# Edit distance (pure Python, no external dependency)
# ---------------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    """Standard Levenshtein distance between strings *a* and *b*."""
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la

    # Two-row DP
    prev = list(range(lb + 1))
    curr = [0] * (lb + 1)
    for i in range(1, la + 1):
        curr[0] = i
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,       # deletion
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + cost # substitution
            )
        prev, curr = curr, prev

    return prev[lb]

# ---------------------------------------------------------------------------
# Prefix index
# ---------------------------------------------------------------------------

def _get_prefix_index(vocab: set[str]) -> dict[str, list[str]]:
    """Build (and cache) a dict mapping 2-char prefix → list of vocab words."""
    global _PREFIX_INDEX
    if _PREFIX_INDEX is not None:
        return _PREFIX_INDEX

    index: dict[str, list[str]] = {}
    for word in vocab:
        if len(word) >= 2:
            key = word[:2]
        else:
            key = word
        index.setdefault(key, []).append(word)

    _PREFIX_INDEX = index
    return _PREFIX_INDEX

# ---------------------------------------------------------------------------
# Candidate finding
# ---------------------------------------------------------------------------

_ALPHABET = "abcdefghijklmnopqrstuvwxyzáéíóúñçü"


def find_candidates(
    word: str,
    vocab: set[str],
    max_dist: int = 3,
    top_n: int = 3,
) -> list[tuple[str, int]]:
    """Find the closest vocabulary words to *word* within *max_dist* edits.

    Uses the prefix index to narrow the candidate set: gathers words whose
    first 2 characters are within edit distance 1 of the word's own prefix,
    then filters by length and computes full Levenshtein distance.

    Returns a list of ``(vocab_word, distance)`` tuples sorted by
    ``(distance, word)``, limited to *top_n* entries.
    """
    word_lower = word.lower()
    index = _get_prefix_index(vocab)

    # Collect candidate prefixes: exact + single-char variations of prefix
    candidate_words: set[str] = set()

    if len(word_lower) >= 2:
        prefix = word_lower[:2]
    else:
        prefix = word_lower

    # Exact prefix
    if prefix in index:
        candidate_words.update(index[prefix])

    # Swap first two chars
    if len(prefix) == 2:
        swapped = prefix[1] + prefix[0]
        if swapped in index:
            candidate_words.update(index[swapped])

    # Vary first char
    for ch in _ALPHABET:
        alt = ch + prefix[1] if len(prefix) == 2 else ch
        if alt in index:
            candidate_words.update(index[alt])

    # Vary second char
    if len(prefix) == 2:
        for ch in _ALPHABET:
            alt = prefix[0] + ch
            if alt in index:
                candidate_words.update(index[alt])

    # Filter by length: only keep words within max_dist of the query length
    word_len = len(word_lower)
    length_filtered = [
        w for w in candidate_words
        if abs(len(w) - word_len) <= max_dist
    ]

    # Compute edit distances and filter
    results: list[tuple[str, int]] = []
    for w in length_filtered:
        d = _edit_distance(word_lower, w)
        if d <= max_dist:
            results.append((w, d))

    # Sort by (distance, word) and return top_n
    results.sort(key=lambda x: (x[1], x[0]))
    return results[:top_n]


# ---------------------------------------------------------------------------
# Compound split
# ---------------------------------------------------------------------------

def find_compound_split(
    word: str,
    vocab: set[str],
    min_part_len: int = 2,
    max_parts: int = 3,
) -> list[str] | None:
    """Try to split *word* into 2 or 3 known vocabulary parts.

    Returns the first successful split (all parts in *vocab* with length ≥
    *min_part_len*), or ``None`` if no valid split is found.
    """
    word_lower = word.lower()
    n = len(word_lower)

    # 2-part splits
    for i in range(min_part_len, n - min_part_len + 1):
        left = word_lower[:i]
        right = word_lower[i:]
        if left in vocab and right in vocab:
            return [left, right]

    if max_parts >= 3:
        # 3-part splits
        for i in range(min_part_len, n - 2 * min_part_len + 1):
            for j in range(i + min_part_len, n - min_part_len + 1):
                left = word_lower[:i]
                mid = word_lower[i:j]
                right = word_lower[j:]
                if left in vocab and mid in vocab and right in vocab:
                    return [left, mid, right]

    return None


# ---------------------------------------------------------------------------
# Abbreviation expansion
# ---------------------------------------------------------------------------

def find_abbrev_expansion(
    word: str,
    vocab: set[str],
    abbrevs: dict[str, str],
) -> list[str] | None:
    """Expand *word* using the abbreviations table.

    1. Exact match: if *word* is in *abbrevs* and its expansion is in *vocab*,
       return ``[expansion]``.
    2. Partial match: for each known abbreviation that appears as a substring
       of *word*, replace it with its expansion and check if the result is in
       *vocab*.

    Returns a list of valid expansions, or ``None`` if none found.
    """
    word_lower = word.lower()
    results: list[str] = []

    # 1. Exact match
    if word_lower in abbrevs:
        expanded = abbrevs[word_lower]
        if expanded in vocab:
            results.append(expanded)

    # 2. Partial match: abbreviation is a substring of word
    for abbr, expanded in abbrevs.items():
        if abbr != word_lower and abbr in word_lower:
            candidate = word_lower.replace(abbr, expanded, 1)
            if candidate in vocab and candidate not in results:
                results.append(candidate)

    return results if results else None


# ---------------------------------------------------------------------------
# Flag unknown words
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-záéíóúñçü]+", re.IGNORECASE)


def flag_unknown_words(
    text: str,
    vocab: set[str],
    min_word_len: int = 3,
) -> list[dict]:
    """Find words in *text* that are not in *vocab*.

    Tokenises with ``[a-záéíóúñçü]+``, lowercases each token, and flags
    those absent from *vocab* whose length ≥ *min_word_len*.

    Returns a list of dicts with keys:
    - ``word``     — lowercased token
    - ``original`` — original token as it appears in the text
    - ``start``    — character offset of the match start
    - ``end``      — character offset of the match end
    """
    flagged: list[dict] = []
    for m in _WORD_RE.finditer(text):
        original = m.group()
        word = original.lower()
        if len(word) < min_word_len:
            continue
        if word not in vocab:
            flagged.append(
                {
                    "word": word,
                    "original": original,
                    "start": m.start(),
                    "end": m.end(),
                }
            )
    return flagged


# ---------------------------------------------------------------------------
# Full pipeline helpers
# ---------------------------------------------------------------------------

def build_candidates_table(
    text: str,
    vocab: set[str],
    abbrevs: dict[str, str],
    max_dist: int = 3,
    top_n: int = 3,
    compound_split: bool = True,
    abbrev_partial: bool = True,
    min_word_len: int = 3,
) -> list[dict]:
    """Flag unknown words and attach correction candidates.

    Augments each flagged-word dict with:
    - ``edit_candidates`` — list of ``(word, dist)`` from :func:`find_candidates`
    - ``compound``        — result of :func:`find_compound_split` or ``None``
    - ``abbrev``          — result of :func:`find_abbrev_expansion` or ``None``
    """
    flagged = flag_unknown_words(text, vocab, min_word_len=min_word_len)
    for entry in flagged:
        word = entry["word"]
        entry["edit_candidates"] = find_candidates(word, vocab, max_dist=max_dist, top_n=top_n)
        entry["compound"] = find_compound_split(word, vocab) if compound_split else None
        entry["abbrev"] = find_abbrev_expansion(word, vocab, abbrevs) if abbrev_partial else None
    return flagged


def format_correction_prompt(text: str, flagged: list[dict]) -> str:
    """Build an LLM prompt listing flagged words with their candidates.

    Returns a string containing the original transcription followed by a
    numbered table of unknown words and suggested replacements.
    """
    lines = [
        "You are correcting an HTR transcription of a historical Spanish manuscript.",
        "For each numbered unknown word below, reply with the line number followed by",
        "a colon and either the best correction or KEEP (if the word is correct).",
        "",
        "=== TRANSCRIPTION ===",
        text,
        "",
        "=== UNKNOWN WORDS ===",
    ]
    for idx, entry in enumerate(flagged, start=1):
        word = entry["word"]
        edit_cands = [f"{w} (d={d})" for w, d in entry.get("edit_candidates", [])]
        compound = entry.get("compound")
        abbrev = entry.get("abbrev")

        suggestions: list[str] = []
        if edit_cands:
            suggestions.append("edit: " + ", ".join(edit_cands))
        if compound:
            suggestions.append("split: " + " + ".join(compound))
        if abbrev:
            suggestions.append("abbrev: " + ", ".join(abbrev))

        suggestion_str = " | ".join(suggestions) if suggestions else "no suggestions"
        lines.append(f"{idx}. '{word}' [{suggestion_str}]")

    return "\n".join(lines)


def parse_corrections(response: str, flagged: list[dict]) -> dict[int, str]:
    """Parse LLM response lines of the form ``N: candidate`` or ``N: KEEP``.

    Returns a dict mapping 1-based index → corrected word (or original word
    for KEEP responses).
    """
    corrections: dict[int, str] = {}
    pattern = re.compile(r"^\s*(\d+)\s*:\s*(.+)\s*$")
    for line in response.splitlines():
        m = pattern.match(line)
        if m:
            idx = int(m.group(1))
            candidate = m.group(2).strip()
            if 1 <= idx <= len(flagged):
                if candidate.upper() == "KEEP":
                    corrections[idx] = flagged[idx - 1]["original"]
                else:
                    corrections[idx] = candidate
    return corrections


def apply_corrections(
    text: str,
    flagged: list[dict],
    corrections: dict[int, str],
) -> str:
    """Apply corrections to *text* in reverse position order.

    Iterates flagged words in reverse order (by ``start`` offset) so that
    earlier replacements don't invalidate the offsets of later ones.
    """
    # Build list of (start, end, replacement) for entries that have a correction
    replacements: list[tuple[int, int, str]] = []
    for idx, entry in enumerate(flagged, start=1):
        if idx in corrections:
            replacements.append((entry["start"], entry["end"], corrections[idx]))

    # Apply in reverse order
    for start, end, replacement in sorted(replacements, key=lambda x: x[0], reverse=True):
        text = text[:start] + replacement + text[end:]

    return text
