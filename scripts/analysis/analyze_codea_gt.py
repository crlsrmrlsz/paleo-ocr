#!/usr/bin/env python3
"""
Analyze CODEA paleographic ground truth files to extract empirical abbreviation
patterns, annotation markers, and orthographic features.

Reads all 64 optimization dataset pages and produces:
- data/abbreviations/codea_abbreviation_frequency.json
- docs/references/codea_gt_analysis.md

CODEA GT format:
- Lines contain text with line numbers on their own lines (e.g., "1", "2")
- Folio headers like "{h. 1r}"
- Abbreviation expansions: expanded letters on their own line between word fragments
  E.g., "d\nic\nho" = "dicho" where "ic" is the expansion
- Annotations in brackets can span multiple lines: "[\nmargen\n: text]"
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPORA_DIR = PROJECT_ROOT / "data" / "corpora" / "codea" / "documents"
OPTIMIZATION_FILE = PROJECT_ROOT / "data" / "optimization" / "dataset_selection.json"
OUTPUT_JSON = PROJECT_ROOT / "data" / "abbreviations" / "codea_abbreviation_frequency.json"
OUTPUT_MD = PROJECT_ROOT / "docs" / "references" / "codea_gt_analysis.md"


def load_page_ids():
    with open(OPTIMIZATION_FILE) as f:
        return json.load(f)


def get_gt_path(page_id: str) -> Path:
    """Convert page_id like 'CODEA-0177_1r' to GT file path."""
    doc_id, page = page_id.rsplit("_", 1)
    return CORPORA_DIR / doc_id / "ground_truth" / f"paleographic_{page}.txt"


def is_expansion_line(s: str) -> bool:
    """Check if a stripped line is an abbreviation expansion.

    Expansion lines contain ONLY lowercase letters (a-z, ç, ñ, á-ú),
    are 1-10 chars, and are NOT standalone line numbers.
    """
    if not s:
        return False
    if re.match(r'^\d+$', s):
        return False
    return bool(re.match(r'^[a-záéíóúçñ]+$', s)) and 1 <= len(s) <= 10


def parse_abbreviations(lines: list[str]) -> list[dict]:
    """
    Parse abbreviation expansions from GT lines.

    Strategy: find sequences where a line ends with letters, the next line is
    an expansion (short lowercase-only), and optionally followed by more expansions
    and/or a suffix.

    IMPORTANT: After finding an abbreviation, we mark expansion lines as consumed
    so they are not re-detected as prefixes (avoiding cascading false positives like
    "reales" -> "eales" -> "ales" -> ...).
    """
    abbreviations = []
    consumed = set()  # Line indices that are expansion lines (already processed)
    i = 0

    while i < len(lines):
        # Skip lines already consumed as expansion parts
        if i in consumed:
            i += 1
            continue

        line = lines[i]

        if i >= len(lines) - 1:
            i += 1
            continue

        # Find if the line ends with letter characters
        end_match = re.search(r'([a-záéíóúçñA-ZÁÉÍÓÚÇÑ]+)\s*$', line)
        if not end_match:
            i += 1
            continue

        prefix_full = end_match.group(1)

        # Check if next line (not consumed) is an expansion
        next_idx = i + 1
        if next_idx in consumed:
            i += 1
            continue
        next_stripped = lines[next_idx].strip() if next_idx < len(lines) else ""
        if not is_expansion_line(next_stripped):
            i += 1
            continue

        # Gather all consecutive expansion lines
        expansion_parts = [next_stripped]
        expansion_indices = [next_idx]
        j = next_idx + 1
        while j < len(lines) and j not in consumed:
            following = lines[j].strip()
            if is_expansion_line(following):
                expansion_parts.append(following)
                expansion_indices.append(j)
                j += 1
            else:
                break

        # Check for suffix: the next line may start with the rest of the word.
        # IMPORTANT: If the raw line starts with a space, that's a word boundary -
        # the expansion ends here and the next text is a separate word.
        suffix = ""
        if j < len(lines):
            raw_following = lines[j]
            # Only extract suffix if the line starts immediately with letters
            # (no leading space = word continues)
            if raw_following and not raw_following[0:1].isspace():
                cont_match = re.match(r'^([a-záéíóúçñA-ZÁÉÍÓÚÇÑ]+)', raw_following)
                if cont_match:
                    suffix = cont_match.group(1)

        expansion = "".join(expansion_parts)
        full_word = prefix_full + expansion + suffix

        # Limit suffix to avoid grabbing following words
        if len(suffix) > 20:
            for k in range(1, len(suffix)):
                if suffix[k].isupper():
                    suffix = suffix[:k]
                    full_word = prefix_full + expansion + suffix
                    break

        if len(full_word) >= 2:
            # Mark expansion lines as consumed
            for idx in expansion_indices:
                consumed.add(idx)

            abbreviations.append({
                "prefix": prefix_full.lower(),
                "expansion": expansion,
                "suffix": suffix.lower(),
                "full_word": full_word.lower(),
                "line_idx": i,
            })

        i += 1

    return abbreviations


def reconstruct_text(text: str) -> str:
    """
    Reconstruct readable text from GT by collapsing abbreviation expansions inline
    and removing structural elements (line numbers, folio headers).

    This gives us text suitable for orthographic analysis with proper word boundaries.
    """
    lines = text.split("\n")
    result_parts = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip folio headers
        if re.match(r'^\{h\.\s+\d+[rv]\}$', stripped):
            i += 1
            continue

        # Skip standalone line numbers
        if re.match(r'^\d+$', stripped):
            i += 1
            continue

        # If this line is an expansion line (lowercase only, short), skip it
        # since it's part of an abbreviation expansion already attached to previous word
        if is_expansion_line(stripped):
            i += 1
            continue

        result_parts.append(line)
        i += 1

    # Join and clean up
    text_out = " ".join(result_parts)
    # Remove bracket annotations for cleaner word extraction
    text_out = re.sub(r'\[[^\]]*\]', ' ', text_out)
    # Normalize whitespace
    text_out = re.sub(r'\s+', ' ', text_out).strip()
    return text_out


def count_markers(text: str) -> Counter:
    """Count annotation markers in the text.

    Markers span multiple lines in the GT because abbreviation encoding splits
    words/markers across lines. We collapse newlines to detect them.
    """
    collapsed = text.replace('\n', '')

    marker_patterns = {
        "[margen:]": r'\[\s*margen[^:\]]*:',
        "[firma:]": r'\[\s*firma[^:\]]*:',
        "[tachado:]": r'\[\s*tachado[^:\]]*:',
        "[interlineado:]": r'\[\s*interlineado[^:\]]*:',
        "[sobrescrito:]": r'\[\s*sobrescrito[^:\]]*:',
        "[ilegible]": r'\[\s*ilegible\s*\]',
        "[signo]": r'\[\s*signo\s*\]',
        "[rúbrica]": r'\[\s*r[uú]brica\s*\]',
        "[cruz]": r'\[\s*cruz\s*\]',
        "[blanco]": r'\[\s*blanco\s*\]',
        "[roto]": r'\[\s*roto[^\]]*\]',
        "[mano N:]": r'\[\s*mano\s*\d+[^:\]]*:',
        "[lat.:]": r'\[\s*lat\s*\.\s*[^:\]]*:',
        "[* * *]": r'\[\s*\*\s*\*\s*\*\s*\]',
    }

    counts = Counter()
    for name, pattern in marker_patterns.items():
        matches = re.findall(pattern, collapsed, re.IGNORECASE)
        if matches:
            counts[name] = len(matches)

    return counts


def analyze_orthography(reconstructed_text: str) -> dict:
    """Analyze period-specific orthographic features on reconstructed text."""
    text = reconstructed_text
    # Extract words
    words = re.findall(r'[a-záéíóúçñA-ZÁÉÍÓÚÇÑ]+', text)
    words_lower = [w.lower() for w in words]

    features = {}

    # ç (c cedilla)
    cedilla_count = sum(1 for c in text if c in 'çÇ')
    cedilla_words = sorted(set(w for w in words_lower if 'ç' in w))
    features["c_cedilla"] = {
        "count": cedilla_count,
        "unique_words": len(cedilla_words),
        "examples": cedilla_words[:25],
    }

    # Double letters
    double_patterns = {}
    for dl in ["nn", "ss", "ff", "ll", "rr", "pp", "tt", "cc", "mm", "bb", "dd", "gg"]:
        matching = sorted(set(w for w in words_lower if dl in w))
        if matching:
            count = sum(1 for w in words_lower if dl in w)
            double_patterns[dl] = {
                "count": count,
                "unique_words": len(matching),
                "examples": matching[:15],
            }
    features["double_letters"] = double_patterns

    # x used where modern Spanish uses j
    # Common historical x→j words
    x_patterns = re.compile(
        r'^(?:.*(?:dixo|dixe|dixier|dixeron|dixese|dexar|dexo|dexa|dexan|dexando|'
        r'baxo|baxa|abaxo|abax|traxo|traxe|traxer|mexor|exemplo|exerc|'
        r'examin|fixo|lexo|relox|cax[ao]|quex|consex|cruxifix|'
        r'exebçion|execuçion|exerç|taxo|pux|aparex).*)$',
        re.IGNORECASE
    )
    # Also: any word with intervocalic x (vowel-x-vowel)
    x_words_specific = [w for w in words_lower if x_patterns.match(w)]
    x_intervocalic = [w for w in words_lower if re.search(r'[aeiouáéíóú]x[aeiouáéíóú]', w) and w not in x_words_specific]
    all_x = sorted(set(x_words_specific + x_intervocalic))
    features["x_for_j"] = {
        "count": len(x_words_specific) + len(x_intervocalic),
        "unique_words": len(all_x),
        "examples": all_x[:25],
    }

    # y- for i- word-initial
    y_for_i_pattern = re.compile(
        r'^y(?:gl|nq|mp|nf|nv|nst|gn|nt[ie]|ns[ti]|ndi|tem|npo|nbi|nven|nfor|magi)',
        re.IGNORECASE
    )
    y_words = [w for w in words_lower if y_for_i_pattern.match(w)]
    features["y_for_i"] = {
        "count": len(y_words),
        "unique_words": len(set(y_words)),
        "examples": sorted(set(y_words))[:25],
    }

    # v for u (vn=un, vna=una, vuestra, etc.)
    v_for_u_pattern = re.compile(
        r'^v(?:n$|na$|no$|nos$|nas$|nj|uestr|eçin|eziñ|ecin|ezin)',
        re.IGNORECASE
    )
    v_words = [w for w in words_lower if v_for_u_pattern.match(w)]
    features["v_for_u"] = {
        "count": len(v_words),
        "unique_words": len(set(v_words)),
        "examples": sorted(set(v_words))[:25],
    }

    # j for i (mjll=mill, qujnientos, etc.)
    # j in positions where modern Spanish has i (before consonants)
    j_for_i_pattern = re.compile(
        r'(?:mj[ln]|quj[nt]|escriu|jnqu|jnfor|ljçen|benefjç|domjnj|'
        r'preuil|serujç|jçi[oa]|jçe|jsto|jble|çibj|djfinj|finyquj|'
        r'syguj|bachjll|alquj|jnst|djçi|oblj|escrjp|escjr|rrecjb|'
        r'jdad$|vjll[ae]|njngu)',
        re.IGNORECASE
    )
    j_words = [w for w in words_lower if j_for_i_pattern.search(w)]
    features["j_for_i"] = {
        "count": len(j_words),
        "unique_words": len(set(j_words)),
        "examples": sorted(set(j_words))[:25],
    }

    # u for b (escriuano, etc.)
    u_for_b_pattern = re.compile(
        r'(?:escriu|Reci[uv]|triu|deui|deue|viui|biui|viu[ie]|aviu)',
        re.IGNORECASE
    )
    u_words = [w for w in words_lower if u_for_b_pattern.search(w)]
    features["u_for_b"] = {
        "count": len(u_words),
        "unique_words": len(set(u_words)),
        "examples": sorted(set(u_words))[:25],
    }

    # h- missing or added (e.g., "aver" for "haber", "aze" for "hace")
    # or "h" where modern Spanish doesn't have it
    h_patterns = {
        "h_missing": [w for w in words_lower if re.match(r'^(?:aver|aze|azer|azj|izo|ombre|onrr)', w)],
        "h_added": [w for w in words_lower if re.match(r'^(?:hebr|hedad|hera$|horden)', w)],
    }
    features["h_variation"] = {
        "h_missing_count": len(h_patterns["h_missing"]),
        "h_missing_examples": sorted(set(h_patterns["h_missing"]))[:15],
        "h_added_count": len(h_patterns["h_added"]),
        "h_added_examples": sorted(set(h_patterns["h_added"]))[:15],
    }

    return features


def extract_base_word(prefix: str, expansion: str, suffix: str) -> str:
    """
    Extract the core abbreviated word by stripping common preceding function words
    (articles, prepositions) that are often written without spaces before the
    abbreviated word in the manuscript.

    Examples:
    - "eld" + [ic] + "ho" -> base word "dicho" (stripping "el")
    - "delad" + [ic] + "ha" -> base word "dicha" (stripping "dela")
    - "conlad" + [ic] + "ha" -> base word "dicha" (stripping "conla")
    """
    full = (prefix + expansion + suffix).lower()

    # Common prefixes: articles, prepositions, and combinations
    # Try to strip from longest to shortest
    function_prefixes = [
        "porquelos", "porquelas", "porquela", "porqueel",
        "conlasdichas", "conlosdichos", "conladicha", "coneldicho",
        "delasdichas", "delosdichos", "deladicha", "deldicho",
        "alasdichas", "alosdichos", "aladicha", "aldicho",
        "enlosdichos", "enlasdichas", "enladicha", "eneldicho",
        "porlas", "porlos", "porla", "porel",
        "conlas", "conlos", "conla", "conel", "consu",
        "delas", "delos", "dela", "delo", "del",
        "alas", "alos", "ala", "alo", "al",
        "enlas", "enlos", "enla", "enel", "en",
        "las", "los", "la", "lo", "el",
        "su", "sus",
    ]

    # Only strip if what remains starts with the first letter of the expansion
    # or is a known abbreviated word root
    for fp in function_prefixes:
        if full.startswith(fp) and len(full) > len(fp):
            remainder = full[len(fp):]
            if len(remainder) >= 2:
                return remainder

    return full


def main():
    page_ids = load_page_ids()
    print(f"Loaded {len(page_ids)} page IDs from optimization dataset")

    all_abbreviations = []
    all_markers = Counter()
    all_reconstructed_texts = []
    total_chars = 0
    pages_analyzed = 0
    missing_pages = []
    page_stats = {}

    for page_id in page_ids:
        gt_path = get_gt_path(page_id)
        if not gt_path.exists():
            missing_pages.append(page_id)
            print(f"  WARNING: Missing GT file for {page_id}: {gt_path}")
            continue

        text = gt_path.read_text(encoding="utf-8")
        lines = text.split("\n")
        total_chars += len(text)
        pages_analyzed += 1

        # Parse abbreviations
        abbrevs = parse_abbreviations(lines)
        for a in abbrevs:
            a["page_id"] = page_id
            # Clean the full_word to extract just the abbreviated word
            a["base_word"] = extract_base_word(a["prefix"], a["expansion"], a["suffix"])
        all_abbreviations.extend(abbrevs)

        # Count markers
        markers = count_markers(text)
        all_markers += markers

        # Reconstruct text for orthographic analysis
        reconstructed = reconstruct_text(text)
        all_reconstructed_texts.append(reconstructed)

        page_stats[page_id] = {
            "abbreviation_count": len(abbrevs),
            "marker_count": sum(markers.values()),
            "char_count": len(text),
        }

    print(f"Analyzed {pages_analyzed} pages ({len(missing_pages)} missing)")
    print(f"Total characters: {total_chars:,}")
    print(f"Total abbreviation instances found: {len(all_abbreviations)}")
    print(f"Total marker instances found: {sum(all_markers.values())}")

    # === Aggregate abbreviation patterns ===
    # Group by expansion (the hidden letters)
    expansion_groups = defaultdict(list)
    for a in all_abbreviations:
        expansion_groups[a["expansion"]].append(a)

    # Group by base_word
    word_groups = defaultdict(list)
    for a in all_abbreviations:
        word_groups[a["base_word"]].append(a)

    # Build abbreviation_patterns dict
    abbreviation_patterns = {}
    for expansion, instances in sorted(expansion_groups.items(), key=lambda x: -len(x[1])):
        clean_words = sorted(set(inst["base_word"] for inst in instances))
        pattern_examples = []
        seen = set()
        for inst in instances:
            pe = f'{inst["prefix"].lower()}[{inst["expansion"]}]{inst["suffix"].lower()}'
            if pe not in seen:
                pattern_examples.append(pe)
                seen.add(pe)
            if len(pattern_examples) >= 10:
                break

        abbreviation_patterns[expansion] = {
            "count": len(instances),
            "example_words": clean_words[:15],
            "pattern_examples": pattern_examples[:10],
            "description": f"Expansion of '{expansion}' hidden under abbreviation mark",
        }

    # Top abbreviated words (using base_word)
    top_abbreviated_words = {}
    for word, instances in sorted(word_groups.items(), key=lambda x: -len(x[1]))[:60]:
        if len(word) < 2:
            continue
        expansions = sorted(set(inst["expansion"] for inst in instances))
        pages = sorted(set(inst["page_id"] for inst in instances))
        top_abbreviated_words[word] = {
            "count": len(instances),
            "expansions": expansions,
            "pages_found_in": len(pages),
        }

    # === Orthographic analysis ===
    combined_reconstructed = " ".join(all_reconstructed_texts)
    print(f"Reconstructed text length: {len(combined_reconstructed):,} chars")
    orthographic_features = analyze_orthography(combined_reconstructed)

    # === Build output JSON ===
    output = {
        "total_pages_analyzed": pages_analyzed,
        "total_pages_requested": len(page_ids),
        "missing_pages": missing_pages,
        "total_characters": total_chars,
        "total_abbreviation_instances": len(all_abbreviations),
        "unique_expansion_patterns": len(abbreviation_patterns),
        "unique_abbreviated_word_forms": len(word_groups),
        "abbreviation_patterns": abbreviation_patterns,
        "top_abbreviated_words": top_abbreviated_words,
        "marker_frequencies": dict(all_markers.most_common()),
        "orthographic_features": orthographic_features,
        "page_stats": page_stats,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nJSON output written to: {OUTPUT_JSON}")

    # === Build markdown summary ===
    md = generate_markdown(
        pages_analyzed, len(page_ids), missing_pages, total_chars,
        all_abbreviations, abbreviation_patterns, top_abbreviated_words,
        word_groups, all_markers, orthographic_features
    )

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Markdown output written to: {OUTPUT_MD}")


def generate_markdown(
    pages_analyzed, total_requested, missing_pages, total_chars,
    all_abbreviations, abbreviation_patterns, top_abbreviated_words,
    word_groups, all_markers, orthographic_features
):
    md = []
    md.append("# CODEA Ground Truth Analysis: Abbreviation Patterns and Orthographic Features")
    md.append("")
    md.append("Empirical analysis of the CODEA paleographic ground truth from the optimization dataset.")
    md.append("This data informs the DSPy agentic pipeline about real abbreviation patterns in historical Spanish manuscripts.")
    md.append("")

    md.append("## Summary Statistics")
    md.append("")
    md.append(f"- **Pages analyzed**: {pages_analyzed} / {total_requested}")
    if missing_pages:
        md.append(f"- **Missing pages**: {', '.join(missing_pages)}")
    md.append(f"- **Total characters (raw GT)**: {total_chars:,}")
    md.append(f"- **Total abbreviation instances**: {len(all_abbreviations):,}")
    md.append(f"- **Unique expansion patterns**: {len(abbreviation_patterns)}")
    md.append(f"- **Unique abbreviated word forms**: {len(word_groups)}")
    md.append(f"- **Average abbreviations per page**: {len(all_abbreviations) / pages_analyzed:.1f}")
    md.append("")

    # Abbreviation patterns by frequency
    md.append("## Abbreviation Expansion Patterns (by frequency)")
    md.append("")
    md.append("These are the letter sequences hidden under abbreviation marks (tildes, superscripts, etc.).")
    md.append("In the CODEA paleographic GT, they appear on separate lines between visible word fragments.")
    md.append("")
    md.append("| Rank | Expansion | Count | Example Words | Pattern Examples |")
    md.append("|------|-----------|-------|---------------|-----------------|")
    for rank, (expansion, data) in enumerate(
        sorted(abbreviation_patterns.items(), key=lambda x: -x[1]["count"]), 1
    ):
        examples = ", ".join(data["example_words"][:4])
        patterns = ", ".join(data["pattern_examples"][:3])
        md.append(f"| {rank} | `{expansion}` | {data['count']} | {examples} | {patterns} |")
        if rank >= 50:
            remaining = len(abbreviation_patterns) - 50
            if remaining > 0:
                md.append(f"| ... | *{remaining} more patterns* | | | |")
            break
    md.append("")

    # Top abbreviated words
    md.append("## Most Frequently Abbreviated Words (top 40)")
    md.append("")
    md.append("| Rank | Word | Count | Hidden Letters | Pages |")
    md.append("|------|------|-------|----------------|-------|")
    for rank, (word, data) in enumerate(
        sorted(top_abbreviated_words.items(), key=lambda x: -x[1]["count"]), 1
    ):
        expansions = ", ".join(data["expansions"])
        md.append(f"| {rank} | {word} | {data['count']} | `{expansions}` | {data['pages_found_in']} |")
        if rank >= 40:
            break
    md.append("")

    # Marker frequencies
    md.append("## Annotation Marker Frequencies")
    md.append("")
    if all_markers:
        md.append("| Marker | Count |")
        md.append("|--------|-------|")
        for marker, count in all_markers.most_common():
            md.append(f"| {marker} | {count} |")
    else:
        md.append("*No annotation markers found.*")
    md.append("")

    # Orthographic features
    md.append("## Orthographic Features")
    md.append("")
    md.append("Period-specific spelling patterns found in the reconstructed text.")
    md.append("")

    md.append("### C cedilla (c with cedilla)")
    md.append("")
    feat = orthographic_features["c_cedilla"]
    md.append(f"- **Character occurrences**: {feat['count']}")
    md.append(f"- **Unique words with c-cedilla**: {feat['unique_words']}")
    md.append(f"- **Examples**: {', '.join(feat['examples'][:20])}")
    md.append("")

    md.append("### Double Letters")
    md.append("")
    md.append("**Note**: The `nn` count is low because most double-n words (anno, sennor, etc.) are encoded")
    md.append("as abbreviations in the GT (the second `n` is a tilde expansion). See the `n` expansion pattern")
    md.append(f"(count={abbreviation_patterns.get('n', {}).get('count', 0)}) in the abbreviation table above.")
    md.append("")
    md.append("| Pattern | Word Count | Unique Words | Examples |")
    md.append("|---------|------------|--------------|----------|")
    for dl, data in sorted(
        orthographic_features["double_letters"].items(), key=lambda x: -x[1]["count"]
    ):
        examples = ", ".join(data["examples"][:6])
        md.append(f"| {dl} | {data['count']} | {data['unique_words']} | {examples} |")
    md.append("")

    md.append("### X for J (e.g., dixo for dijo)")
    md.append("")
    feat = orthographic_features["x_for_j"]
    md.append(f"- **Occurrences**: {feat['count']}")
    md.append(f"- **Unique words**: {feat['unique_words']}")
    md.append(f"- **Examples**: {', '.join(feat['examples'][:20])}")
    md.append("")

    md.append("### Y for I word-initial (e.g., yglesia for iglesia)")
    md.append("")
    feat = orthographic_features["y_for_i"]
    md.append(f"- **Occurrences**: {feat['count']}")
    md.append(f"- **Unique words**: {feat['unique_words']}")
    md.append(f"- **Examples**: {', '.join(feat['examples'][:20])}")
    md.append("")

    md.append("### V for U (e.g., vn for un)")
    md.append("")
    feat = orthographic_features["v_for_u"]
    md.append(f"- **Occurrences**: {feat['count']}")
    md.append(f"- **Unique words**: {feat['unique_words']}")
    md.append(f"- **Examples**: {', '.join(feat['examples'][:20])}")
    md.append("")

    md.append("### J for I (e.g., mjll for mill)")
    md.append("")
    feat = orthographic_features["j_for_i"]
    md.append(f"- **Occurrences**: {feat['count']}")
    md.append(f"- **Unique words**: {feat['unique_words']}")
    md.append(f"- **Examples**: {', '.join(feat['examples'][:20])}")
    md.append("")

    md.append("### U for B (e.g., escriuano for escribano)")
    md.append("")
    feat = orthographic_features["u_for_b"]
    md.append(f"- **Occurrences**: {feat['count']}")
    md.append(f"- **Unique words**: {feat['unique_words']}")
    md.append(f"- **Examples**: {', '.join(feat['examples'][:20])}")
    md.append("")

    md.append("### H variation")
    md.append("")
    feat = orthographic_features["h_variation"]
    md.append(f"- **H missing (e.g., aver for haber)**: {feat['h_missing_count']} occurrences")
    if feat["h_missing_examples"]:
        md.append(f"  - Examples: {', '.join(feat['h_missing_examples'][:10])}")
    md.append(f"- **H added (e.g., hebrero for febrero)**: {feat['h_added_count']} occurrences")
    if feat["h_added_examples"]:
        md.append(f"  - Examples: {', '.join(feat['h_added_examples'][:10])}")
    md.append("")

    md.append("---")
    md.append("")
    md.append("*Generated by `scripts/analysis/analyze_codea_gt.py` from the CODEA optimization dataset (64 pages).*")
    md.append("")

    return "\n".join(md)


if __name__ == "__main__":
    main()
