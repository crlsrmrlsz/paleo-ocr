"""Abbreviation reference and orthographic rules for 16th-century Spanish manuscripts.

Cross-validated between:
  - Empirical: 2,105 abbreviation instances from 64 CODEA GT pages
    (data/abbreviations/codea_abbreviation_frequency.json)
  - Academic: 182 entries from 8 scholarly sources
    (data/abbreviations/academic_abbreviations.json)
  - CHARTA 2013 transcription criteria
    (docs/references/charta_criteria_summary.md)

Used as structured reference data for the DSPy pipeline InputField.
"""

import json

# Top abbreviated words from CODEA GT, cross-validated with academic sources.
# Format: abbreviated_form → expanded_form (period spelling per CHARTA TP rules)
# Only includes entries with ≥5 empirical occurrences AND academic documentation.
ABBREVIATIONS = {
    # --- Highest frequency (>20 occurrences in CODEA GT) ---
    # que: 159 occurrences, 28 pages. Sources: Ueda 2015, Carlin 1999, CHARTA 2013
    "q": "que",
    # dicho/a/s: 254 combined. Sources: Carlin 1999, CODEA empirical
    "dho": "dicho",
    "dha": "dicha",
    "dhos": "dichos",
    # reales: 84 occurrences. Sources: Carlin 1999
    "rs": "reales",
    "rl": "real",
    # maravedis: 29 occurrences. Sources: Carlin 1999, CODEA empirical
    "mrs": "maravedis",
    # para: 26 occurrences. Sources: Carlin 1999, CHARTA 2013
    "pra": "para",
    # nuestro/a: 22+ occurrences. Sources: Carlin 1999, CHARTA 2013
    "nro": "nuestro",
    "nra": "nuestra",
    # --- Medium frequency (7-20 occurrences) ---
    # testigos: 29 occurrences. Sources: Carlin 1999
    "tgos": "testigos",
    "tgo": "testigo",
    # derecho: 18 occurrences. Sources: Carlin 1999
    "dro": "derecho",
    # vuestro/a: academic sources (Carlin 1999, CHARTA 2013)
    "vro": "vuestro",
    "vra": "vuestra",
    # vezino/a: 16 occurrences. Sources: Carlin 1999, CODEA empirical
    "vzo": "vezino",
    "vza": "vezina",
    "vzos": "vezinos",
    # escriuano: 4+ occurrences. Sources: Carlin 1999, CODEA empirical
    "esno": "escriuano",
    # merced: academic sources (Carlin 1999)
    "mrd": "merced",
    # --- Lower frequency but academically important ---
    # por/pre: Sources: CHARTA 2013, Ueda 2015 (consonant+r pattern)
    "pr": "por",
    # qual: 9 occurrences in CODEA
    "ql": "qual",
    # Christographic: Sources: CHARTA 2013, Carlin 1999
    "xpo": "Christo",
    "xpoval": "Christoval",
    # Titles: Sources: Carlin 1999, FamilySearch
    "Dn": "Don",
    "Da": "Doña",
}

# Most frequent expansion patterns from CODEA GT (top 15 by count).
# These are the letter sequences hidden under abbreviation marks.
# Used to understand what the model needs to "see through" in the manuscript.
EXPANSION_PATTERNS = {
    "ic": {"count": 235, "words": ["dicho", "dicha", "dichos"]},
    "n": {"count": 219, "words": ["anno", "sennor", "con", "en"]},
    "ue": {"count": 217, "words": ["que", "qual", "quando"]},
    "e": {"count": 201, "words": ["testigos", "del", "dela"]},
    "eales": {"count": 72, "words": ["reales"]},
    "er": {"count": 49, "words": ["clerigo", "bachiller", "muger"]},
    "aravedis": {"count": 29, "words": ["maravedis"]},
    "eal": {"count": 27, "words": ["real"]},
    "uest": {"count": 22, "words": ["nuestro", "vuestro"]},
    "r": {"count": 21, "words": ["por", "dar"]},
    "ar": {"count": 20, "words": ["para", "parte"]},
    "es": {"count": 19, "words": ["yglesia"]},
    "an": {"count": 18, "words": ["juan", "santa"]},
    "ie": {"count": 17, "words": ["tierras"]},
    "re": {"count": 12, "words": ["presentes", "preçio"]},
}

# Orthographic alternations that are period-standard (NOT errors).
# Grounded in empirical CODEA data with counts.
# Sources: CHARTA 2013 §4.2-4.4, Millares Carlo, empirical CODEA GT
VALID_ALTERNATIONS = {
    "c_cedilla": {
        "count": 483,
        "description": "ç is standard 16th-c. spelling, not an error",
        "examples": ["çibdad", "vezino", "conoçe", "pareçer", "hazer"],
    },
    "x_for_j": {
        "count": 112,
        "description": "x represents /ʃ/ shifting to /x/, standard before 1700",
        "examples": ["dixo", "dexar", "abaxo", "baxo", "cruxifixo"],
    },
    "v_for_u": {
        "count": 56,
        "description": "u/v interchangeable in 16th c., transcribe as written",
        "examples": ["vn", "vna", "vnas", "vnos", "vezino"],
    },
    "j_for_i": {
        "count": 47,
        "description": "j/i interchangeable, especially before consonants",
        "examples": ["mjll", "bachjller", "escriuano", "domjnjo"],
    },
    "y_for_i": {
        "count": 41,
        "description": "y preferred word-initially in many scribal traditions",
        "examples": ["yglesia", "ynquisidor", "ynformaçion", "ynponga"],
    },
    "double_ss": {
        "count": 68,
        "description": "Double s is standard, first s may resemble j",
        "examples": ["alonsso", "anssi", "anssimismo", "possesion"],
    },
    "u_for_b": {
        "count": 7,
        "description": "u/b alternation, especially in escriuano",
        "examples": ["escriuano", "deuemos", "deuido"],
    },
}

# Annotation markers and their frequency in CODEA GT (64 pages)
MARKER_FREQUENCIES = {
    "[tachado:]": 34,
    "[margen:]": 30,
    "[sobrescrito:]": 28,
    "[rúbrica]": 16,
    "[interlineado:]": 12,
    "[firma:]": 9,
    "[cruz]": 7,
    "[mano N:]": 6,
    "[lat.:]": 4,
    "[* * *]": 4,
    "[signo]": 2,
}


def build_reading_reference() -> str:
    """Stage 1 reference: help the VLM read the manuscript image.

    Tells the model what abbreviation marks look like, what common
    abbreviated forms mean, and what period spellings to preserve.
    This is RECOGNITION help — "here's what you're about to see."
    """
    lines = [
        "VISUAL ABBREVIATION GUIDE (16th-century Spanish manuscripts):",
        "",
        "Abbreviation marks you will see and what they mean:",
        "  Tilde/line over a vowel → missing n or m (e.g., cō = con, dō = don)",
        "  Superscript small letters → word is abbreviated (e.g., dho = dicho)",
        "  Crossed p (bar through descender) → por, per, or pre",
        "  q with tilde or stroke → que",
        "",
        "Common abbreviated forms (what the scribe wrote → full word):",
    ]
    for abbr, expanded in sorted(ABBREVIATIONS.items()):
        lines.append(f"  {abbr} → {expanded}")

    lines.append("")
    lines.append("Period spellings to PRESERVE (these are correct, not errors):")
    lines.append("  ç (cedilla): çibdad, conoçe, pareçer — do not change to z")
    lines.append("  x for j: dixo, dexar, abaxo — do not change to j")
    lines.append("  v/u interchangeable: vn, vna, vezino — transcribe as written")
    lines.append("  j/i interchangeable: mjll, bachjller — transcribe as written")
    lines.append("  y word-initial: yglesia, ynquisidor — do not change to i")
    lines.append("  double letters: alonsso, anssi, annos — preserve as written")

    return "\n".join(lines)


def build_correction_reference() -> str:
    """Stage 2 reference: help a post-processor correct transcription text.

    Provides expansion patterns with frequency data for identifying
    and fixing abbreviation-related errors in already-transcribed text.
    This is CORRECTION help — "here's how to fix what was already read."
    """
    lines = [
        "ABBREVIATION EXPANSION PATTERNS (from 2,105 instances in 64 CODEA pages):",
        "",
        "Most frequent hidden-letter patterns (letters under abbreviation marks):",
    ]
    for pattern, info in EXPANSION_PATTERNS.items():
        words = ", ".join(info["words"])
        lines.append(f"  [{pattern}] appears {info['count']}x in: {words}")

    lines.append("")
    lines.append("Abbreviated form → expanded word (for text-level correction):")
    for abbr, expanded in sorted(ABBREVIATIONS.items()):
        lines.append(f"  {abbr} → {expanded}")

    lines.append("")
    lines.append("Context-dependent expansions (need surrounding text to resolve):")
    lines.append("  p with bar → por (most common), per, pre — check context")
    lines.append("  tilde over vowel → n (most common) or m before p/b")

    return "\n".join(lines)


# Pre-built stage-specific reference texts
READING_REFERENCE = build_reading_reference()
CORRECTION_REFERENCE = build_correction_reference()

# JSON export for programmatic access
ABBREVIATION_DICT_JSON = json.dumps(ABBREVIATIONS, ensure_ascii=False, indent=2)
