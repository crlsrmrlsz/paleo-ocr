"""Build domain knowledge strings from project reference data."""

import json
from pathlib import Path

from .config import PROJECT_ROOT


def build_abbreviation_table() -> str:
    path = PROJECT_ROOT / "data" / "abbreviations" / "academic_abbreviations.json"
    data = json.loads(path.read_text())
    table = {}
    for entry in data["abbreviations"]:
        table[entry["abbreviated"]] = entry["expanded"]
    return json.dumps(table, ensure_ascii=False)


def build_confusion_pairs() -> str:
    return (
        "Procesal script letter confusions (both directions):\n"
        "- c ↔ e: c has sharper angles; e has a horizontal midstroke\n"
        "- n ↔ u: nearly identical humps in procesal; use context\n"
        "- a ↔ o: a has a tag stroke on top; o is fully closed\n"
        "- m ↔ n: m has 3 humps, n has 2 (rapid procesal may reduce m to 2)\n"
        "- r ↔ long s: long s descends below baseline; r has shorter stroke\n"
        "- f ↔ long s: f has a crossbar; long s does not\n"
        "- d ↔ cl: round d can look like cl in procesal\n"
        "- h ↔ li: looped h may resemble li\n"
        "- ss first s ↔ j: first s in 'ss' often resembles j without dot"
    )


def build_valid_alternations() -> str:
    from .abbreviations import VALID_ALTERNATIONS

    lines = ["Period-standard spellings (XVI-XVII century, preserve as written):"]
    for key, data in VALID_ALTERNATIONS.items():
        examples = ", ".join(data["examples"][:3])
        lines.append(f"- {key}: {data['count']} occurrences — {examples}")
    return "\n".join(lines)


def build_rlm_abbreviation_data() -> str:
    # Academic entries
    acad_path = PROJECT_ROOT / "data" / "abbreviations" / "academic_abbreviations.json"
    acad_data = json.loads(acad_path.read_text())
    combined = {}
    for entry in acad_data["abbreviations"]:
        combined[entry["abbreviated"]] = {
            "expanded": entry["expanded"],
            "source": "academic",
            "category": entry.get("category", ""),
        }

    # Empirical expansion patterns
    freq_path = PROJECT_ROOT / "data" / "abbreviations" / "codea_abbreviation_frequency.json"
    freq_data = json.loads(freq_path.read_text())
    for pattern, data in freq_data.get("abbreviation_patterns", {}).items():
        if pattern not in combined:
            combined[pattern] = {
                "expansion_pattern": pattern,
                "example_words": data.get("example_words", [])[:5],
                "count": data.get("count", 0),
                "source": "empirical",
            }

    return json.dumps(combined, ensure_ascii=False)


def build_flat_abbreviations() -> str:
    """Build flat abbreviation dict for RLM: 'abbreviated' -> 'expanded'.

    Simple string-to-string mapping. No nested objects.
    RLM model sees 500-char preview and searches via code.
    """
    path = PROJECT_ROOT / "data" / "abbreviations" / "academic_abbreviations.json"
    data = json.loads(path.read_text())
    flat = {}
    for entry in data["abbreviations"]:
        flat[entry["abbreviated"]] = entry["expanded"]
    return json.dumps(flat, ensure_ascii=False, indent=2)


def load_domain_knowledge(fields: list[str] | None = None) -> dict:
    builders = {
        "abbreviation_table": build_abbreviation_table,
        "confusion_pairs": build_confusion_pairs,
        "valid_alternations": build_valid_alternations,
    }
    if fields is None:
        fields = list(builders.keys())
    unknown = set(fields) - set(builders.keys())
    if unknown:
        raise ValueError(f"Unknown domain knowledge fields: {unknown}. Valid: {set(builders.keys())}")
    return {name: builders[name]() for name in fields}
