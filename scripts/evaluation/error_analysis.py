#!/usr/bin/env python3
"""Systematic character-level error analysis across DSPy pipeline versions.

Produces:
  - Character-level substitution confusion matrix (ref_char → hyp_char)
  - Error type classification (diacritical, paleographic, modernization, etc.)
  - Per-version and cross-version error profiles
  - Identification of persistent vs version-specific errors
"""

import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from rapidfuzz.distance import Levenshtein

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
SUBSETS_DIR = PROJECT_ROOT / "data" / "subsets"
REPORTS_DIR = PROJECT_ROOT / "data" / "evaluation" / "reports"

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from pipeline_config import load_manifest  # noqa: E402

# Pipeline versions to analyze (all Claude Opus)
DSPY_VERSIONS = [
    "pipeline/01_yolo_blocks", "pipeline/02_yolo_model_first",
    "pipeline/03_strips_page_desc", "pipeline/04_strips_split_context",
    "pipeline/04b_strips_split_ctx_rezoom", "pipeline/05_tiles_2d",
    "pipeline/06_strips_no_context", "pipeline/07_yolo_crop_strips",
]

# Content-mode marker stripping (same as compute_metrics.py)
_MARKERS_WITH_TEXT = {
    "firma", "tachado", "margen", "interlineado",
    "sobrescrito", "encabezamiento", "nota", r"mano[^:]*",
}
_MARKERS_EMPTY = {"signo", "rúbrica", "cruz", "ilegible", "blanco", "roto"}


def strip_markers_content(text: str) -> str:
    """Content-mode marker stripping."""
    for m in _MARKERS_WITH_TEXT:
        text = re.sub(rf"\[{m}:\s*(.*?)\]", r"\1", text)
    for m in _MARKERS_EMPTY:
        if m == "blanco":
            text = re.sub(rf"\[{m}\]", " ", text)
        else:
            text = re.sub(rf"\[{m}\]", "", text)
    return text


def normalize_for_eval(text: str) -> str:
    """NFC + whitespace normalization (same as compute_metrics content mode)."""
    text = strip_markers_content(text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─── Error classification categories ───────────────────────────────────────

def classify_substitution(ref_char: str, hyp_char: str) -> str:
    """Classify a character substitution into an error category."""
    r, h = ref_char.lower(), hyp_char.lower()

    # Case-only error
    if r == h:
        return "case"

    # Diacritical errors (ç↔c, ñ↔n, á↔a, etc.)
    r_base = unicodedata.normalize("NFD", r)
    h_base = unicodedata.normalize("NFD", h)
    r_stripped = "".join(c for c in r_base if unicodedata.category(c) != "Mn")
    h_stripped = "".join(c for c in h_base if unicodedata.category(c) != "Mn")
    if r_stripped == h_stripped and r_stripped:
        return "diacritical"

    # Cedilla specifically (ç↔c, ç↔z) — very common in procesal
    if {r, h} & {"ç"} and {r_stripped, h_stripped} & {"c", "z"}:
        return "cedilla"

    # Paleographic confusions (period-standard alternations)
    paleo_pairs = {
        frozenset({"u", "v"}), frozenset({"i", "j"}), frozenset({"x", "j"}),
        frozenset({"s", "ſ"}), frozenset({"b", "v"}),
    }
    if frozenset({r, h}) in paleo_pairs:
        return "paleographic_alternation"

    # Modernization indicators (j→i when GT has j, etc.)
    modernization_map = {
        ("j", "i"): "modernization",  # mjsmo → mismo
        ("x", "j"): "modernization",  # dixo → dijo
        ("ç", "c"): "modernization",  # çenso → censo
        ("ç", "z"): "modernization",  # conoçe → conoce
        ("ss", "s"): "modernization",
        ("nn", "ñ"): "modernization",
        ("q", "c"): "modernization",  # quanto → cuanto
    }
    if (r, h) in modernization_map:
        return "modernization"

    # Ampersand / abbreviation marker
    if r == "&" or h == "&":
        return "abbreviation_marker"

    # Whitespace / word boundary
    if r.isspace() or h.isspace():
        return "word_boundary"

    # Punctuation confusion
    if not r.isalnum() or not h.isalnum():
        return "punctuation"

    # Similar-looking letters (visually confusable in procesal)
    visual_confusions = {
        frozenset({"e", "o"}), frozenset({"a", "o"}), frozenset({"a", "e"}),
        frozenset({"n", "u"}), frozenset({"n", "r"}), frozenset({"m", "n"}),
        frozenset({"c", "e"}), frozenset({"t", "r"}), frozenset({"d", "o"}),
        frozenset({"l", "i"}), frozenset({"l", "t"}), frozenset({"f", "s"}),
        frozenset({"h", "b"}), frozenset({"d", "b"}),
    }
    if frozenset({r, h}) in visual_confusions:
        return "visual_confusion"

    return "other_substitution"


def classify_deletion(ref_chars: str) -> str:
    """Classify deleted characters."""
    if all(c.isspace() for c in ref_chars):
        return "deleted_whitespace"
    if all(c == "&" for c in ref_chars):
        return "deleted_abbreviation"
    return "deleted_text"


def classify_insertion(hyp_chars: str) -> str:
    """Classify inserted characters."""
    if all(c.isspace() for c in hyp_chars):
        return "inserted_whitespace"
    return "inserted_text"


# ─── Core alignment analysis ──────────────────────────────────────────────

def analyze_page(gt_text: str, hyp_text: str) -> dict:
    """Full character-level error analysis for a single page."""
    gt = normalize_for_eval(gt_text)
    hyp = normalize_for_eval(hyp_text)

    opcodes = Levenshtein.opcodes(gt, hyp)

    result = {
        "gt_len": len(gt),
        "hyp_len": len(hyp),
        "substitutions": 0,
        "deletions": 0,
        "insertions": 0,
        "correct": 0,
        # Detailed classification
        "sub_categories": Counter(),
        "sub_pairs": Counter(),  # (ref_char, hyp_char) → count
        "del_categories": Counter(),
        "ins_categories": Counter(),
        # Contextual errors (word-level)
        "deleted_words": [],
        "inserted_words": [],
        "garbled_segments": [],  # stretches where everything is wrong
    }

    # Track consecutive errors for garbled-segment detection
    consecutive_errors = 0
    garbled_start = None

    for tag, rs, re_, hs, he in opcodes:
        if tag == "equal":
            result["correct"] += re_ - rs
            # Reset garbled tracking
            if consecutive_errors >= 20 and garbled_start is not None:
                result["garbled_segments"].append({
                    "gt": gt[garbled_start:rs],
                    "hyp": hyp[garbled_start - (rs - hs) if rs > hs else garbled_start:he - (re_ - rs) if re_ > rs else hs],
                    "gt_pos": garbled_start,
                    "length": rs - garbled_start,
                })
            consecutive_errors = 0
            garbled_start = None

        elif tag == "replace":
            ref_span = gt[rs:re_]
            hyp_span = hyp[hs:he]
            ref_len = re_ - rs
            hyp_len = he - hs
            min_len = min(ref_len, hyp_len)

            # Character-by-character substitution classification
            for k in range(min_len):
                cat = classify_substitution(ref_span[k], hyp_span[k])
                result["sub_categories"][cat] += 1
                pair = (ref_span[k], hyp_span[k])
                result["sub_pairs"][pair] += 1
            result["substitutions"] += min_len

            # Excess as deletion or insertion
            if ref_len > hyp_len:
                excess = ref_span[min_len:]
                result["deletions"] += len(excess)
                result["del_categories"][classify_deletion(excess)] += 1
            elif hyp_len > ref_len:
                excess = hyp_span[min_len:]
                result["insertions"] += len(excess)
                result["ins_categories"][classify_insertion(excess)] += 1

            # Track consecutive errors
            consecutive_errors += max(ref_len, hyp_len)
            if garbled_start is None:
                garbled_start = rs

        elif tag == "delete":
            ref_span = gt[rs:re_]
            result["deletions"] += len(ref_span)
            result["del_categories"][classify_deletion(ref_span)] += 1
            consecutive_errors += re_ - rs
            if garbled_start is None:
                garbled_start = rs

        elif tag == "insert":
            hyp_span = hyp[hs:he]
            result["insertions"] += len(hyp_span)
            result["ins_categories"][classify_insertion(hyp_span)] += 1
            consecutive_errors += he - hs
            if garbled_start is None:
                garbled_start = rs

    return result


def word_level_analysis(gt_text: str, hyp_text: str) -> dict:
    """Word-level error analysis: whole-word insertions, deletions, substitutions."""
    gt = normalize_for_eval(gt_text)
    hyp = normalize_for_eval(hyp_text)

    gt_words = gt.split()
    hyp_words = hyp.split()

    opcodes = Levenshtein.opcodes(gt_words, hyp_words)

    result = {
        "total_gt_words": len(gt_words),
        "total_hyp_words": len(hyp_words),
        "correct_words": 0,
        "substituted_words": 0,
        "deleted_words": 0,
        "inserted_words": 0,
        "word_subs": [],  # (gt_word, hyp_word) pairs
        "word_dels": [],  # deleted gt words
        "word_ins": [],   # inserted hyp words
    }

    for tag, rs, re_, hs, he in opcodes:
        if tag == "equal":
            result["correct_words"] += re_ - rs
        elif tag == "replace":
            for k in range(min(re_ - rs, he - hs)):
                result["word_subs"].append((gt_words[rs + k], hyp_words[hs + k]))
            result["substituted_words"] += min(re_ - rs, he - hs)
            if re_ - rs > he - hs:
                for k in range(he - hs, re_ - rs):
                    result["word_dels"].append(gt_words[rs + k])
                result["deleted_words"] += (re_ - rs) - (he - hs)
            elif he - hs > re_ - rs:
                for k in range(re_ - rs, he - hs):
                    result["word_ins"].append(hyp_words[hs + k])
                result["inserted_words"] += (he - hs) - (re_ - rs)
        elif tag == "delete":
            for k in range(rs, re_):
                result["word_dels"].append(gt_words[k])
            result["deleted_words"] += re_ - rs
        elif tag == "insert":
            for k in range(hs, he):
                result["word_ins"].append(hyp_words[k])
            result["inserted_words"] += he - hs

    return result


# ─── Main analysis ─────────────────────────────────────────────────────────

def run_analysis():
    """Run full error analysis across all DSPy versions on CODEA paleographic."""
    gt_dir = SUBSETS_DIR / "codea" / "ground_truth" / "paleographic"
    gt_files = sorted(gt_dir.glob("*.txt"))

    print(f"Found {len(gt_files)} GT pages")
    print(f"Analyzing {len(DSPY_VERSIONS)} DSPy versions\n")

    # Per-version aggregate results
    version_results = {}

    for version in DSPY_VERSIONS:
        norm_dir = RESULTS_DIR / version / "normalized"
        if not norm_dir.exists():
            print(f"  SKIP {version}: no normalized dir")
            continue

        print(f"═══ {version} ═══")
        agg = {
            "total_gt_chars": 0,
            "total_hyp_chars": 0,
            "total_correct": 0,
            "total_substitutions": 0,
            "total_deletions": 0,
            "total_insertions": 0,
            "sub_categories": Counter(),
            "sub_pairs": Counter(),
            "del_categories": Counter(),
            "ins_categories": Counter(),
            "n_garbled_segments": 0,
            "garbled_chars": 0,
            "pages_analyzed": 0,
            # Word-level
            "word_subs": Counter(),
            "word_dels": Counter(),
            "word_ins": Counter(),
            "total_word_subs": 0,
            "total_word_dels": 0,
            "total_word_ins": 0,
            "total_gt_words": 0,
        }

        for gt_file in gt_files:
            hyp_file = norm_dir / gt_file.name
            if not hyp_file.exists():
                continue

            gt_text = gt_file.read_text(encoding="utf-8")
            hyp_text = hyp_file.read_text(encoding="utf-8")

            # Character-level analysis
            page_result = analyze_page(gt_text, hyp_text)
            agg["total_gt_chars"] += page_result["gt_len"]
            agg["total_hyp_chars"] += page_result["hyp_len"]
            agg["total_correct"] += page_result["correct"]
            agg["total_substitutions"] += page_result["substitutions"]
            agg["total_deletions"] += page_result["deletions"]
            agg["total_insertions"] += page_result["insertions"]
            agg["sub_categories"] += page_result["sub_categories"]
            agg["sub_pairs"] += page_result["sub_pairs"]
            agg["del_categories"] += page_result["del_categories"]
            agg["ins_categories"] += page_result["ins_categories"]
            agg["n_garbled_segments"] += len(page_result["garbled_segments"])
            agg["garbled_chars"] += sum(
                s["length"] for s in page_result["garbled_segments"]
            )
            agg["pages_analyzed"] += 1

            # Word-level analysis
            word_result = word_level_analysis(gt_text, hyp_text)
            agg["total_gt_words"] += word_result["total_gt_words"]
            agg["total_word_subs"] += word_result["substituted_words"]
            agg["total_word_dels"] += word_result["deleted_words"]
            agg["total_word_ins"] += word_result["inserted_words"]
            for gt_w, hyp_w in word_result["word_subs"]:
                agg["word_subs"][(gt_w, hyp_w)] += 1
            for w in word_result["word_dels"]:
                agg["word_dels"][w] += 1
            for w in word_result["word_ins"]:
                agg["word_ins"][w] += 1

        version_results[version] = agg
        _print_version_summary(version, agg)

    # Cross-version analysis
    print("\n" + "═" * 80)
    print("CROSS-VERSION ANALYSIS")
    print("═" * 80)
    _print_cross_version(version_results)

    # Save detailed results
    _save_results(version_results)

    return version_results


def _print_version_summary(version: str, agg: dict):
    """Print summary for one version."""
    total_errors = agg["total_substitutions"] + agg["total_deletions"] + agg["total_insertions"]
    gt_chars = agg["total_gt_chars"]

    print(f"  Pages: {agg['pages_analyzed']}")
    print(f"  GT chars: {gt_chars:,}  |  Hyp chars: {agg['total_hyp_chars']:,}")
    print(f"  CER (approx): {total_errors / gt_chars:.1%}")
    print(f"  Correct: {agg['total_correct']:,} ({agg['total_correct']/gt_chars:.1%})")
    print(f"  Substitutions: {agg['total_substitutions']:,} ({agg['total_substitutions']/gt_chars:.1%})")
    print(f"  Deletions:     {agg['total_deletions']:,} ({agg['total_deletions']/gt_chars:.1%})")
    print(f"  Insertions:    {agg['total_insertions']:,} ({agg['total_insertions']/gt_chars:.1%})")
    print(f"  Garbled segments: {agg['n_garbled_segments']} ({agg['garbled_chars']:,} chars)")

    # Substitution category breakdown
    print(f"\n  Substitution breakdown ({agg['total_substitutions']} total):")
    for cat, count in agg["sub_categories"].most_common(15):
        pct = count / agg["total_substitutions"] * 100 if agg["total_substitutions"] else 0
        print(f"    {cat:30s} {count:6d}  ({pct:5.1f}%)")

    # Top substitution pairs
    print(f"\n  Top 20 character confusions:")
    for (r, h), count in agg["sub_pairs"].most_common(20):
        r_display = repr(r) if not r.isprintable() or r == " " else r
        h_display = repr(h) if not h.isprintable() or h == " " else h
        print(f"    {r_display:>4s} → {h_display:<4s}  {count:5d}")

    # Top word-level substitutions
    print(f"\n  Top 20 word substitutions (GT → Hyp):")
    for (gt_w, hyp_w), count in agg["word_subs"].most_common(20):
        print(f"    {gt_w:>25s} → {hyp_w:<25s}  {count:3d}")

    # Top inserted words (hallucinations)
    print(f"\n  Top 15 inserted words (hallucinations):")
    for w, count in agg["word_ins"].most_common(15):
        print(f"    {w:<30s}  {count:3d}")

    # Top deleted words (omissions)
    print(f"\n  Top 15 deleted words (omissions):")
    for w, count in agg["word_dels"].most_common(15):
        print(f"    {w:<30s}  {count:3d}")

    # Deletion categories
    print(f"\n  Deletion breakdown ({agg['total_deletions']} total):")
    for cat, count in agg["del_categories"].most_common():
        pct = count / max(agg["total_deletions"], 1) * 100
        print(f"    {cat:30s} {count:6d}  ({pct:5.1f}%)")

    # Insertion categories
    print(f"\n  Insertion breakdown ({agg['total_insertions']} total):")
    for cat, count in agg["ins_categories"].most_common():
        pct = count / max(agg["total_insertions"], 1) * 100
        print(f"    {cat:30s} {count:6d}  ({pct:5.1f}%)")

    print()


def _print_cross_version(version_results: dict):
    """Cross-version comparison and persistent patterns."""

    # 1. Error profile comparison table
    print("\n┌─ Error Profile Comparison (% of GT chars) ────────────────────────────┐")
    print(f"{'Version':<20s} {'CER':>7s} {'S_rate':>7s} {'D_rate':>7s} {'I_rate':>7s} {'Garbled':>8s}")
    print("─" * 70)
    for version in DSPY_VERSIONS:
        if version not in version_results:
            continue
        agg = version_results[version]
        gt = agg["total_gt_chars"]
        total_err = agg["total_substitutions"] + agg["total_deletions"] + agg["total_insertions"]
        print(
            f"{version:<20s} "
            f"{total_err/gt:>6.1%} "
            f"{agg['total_substitutions']/gt:>6.1%} "
            f"{agg['total_deletions']/gt:>6.1%} "
            f"{agg['total_insertions']/gt:>6.1%} "
            f"{agg['garbled_chars']:>7,d}"
        )
    print("└" + "─" * 69 + "┘")

    # 2. Substitution categories comparison
    all_categories = set()
    for agg in version_results.values():
        all_categories |= set(agg["sub_categories"].keys())

    print("\n┌─ Substitution Category Comparison (% of subs) ──────────────────────┐")
    cats_sorted = sorted(all_categories)
    header = f"{'Category':<28s}" + "".join(f"{v.replace('dspy_',''):>10s}" for v in DSPY_VERSIONS if v in version_results)
    print(header)
    print("─" * len(header))
    for cat in cats_sorted:
        row = f"{cat:<28s}"
        for version in DSPY_VERSIONS:
            if version not in version_results:
                continue
            agg = version_results[version]
            total_subs = max(agg["total_substitutions"], 1)
            count = agg["sub_categories"].get(cat, 0)
            row += f"{count/total_subs:>9.1%} "
        print(row)
    print("└" + "─" * 69 + "┘")

    # 3. Persistent character confusions (appear in ALL versions)
    print("\n┌─ Persistent Character Confusions (present in all versions) ──────────┐")
    # Get pairs that appear in every version
    all_pairs = None
    for agg in version_results.values():
        pairs = set(agg["sub_pairs"].keys())
        if all_pairs is None:
            all_pairs = pairs
        else:
            all_pairs &= pairs

    if all_pairs:
        # Sum across versions
        combined = Counter()
        for agg in version_results.values():
            for pair in all_pairs:
                combined[pair] += agg["sub_pairs"][pair]

        for (r, h), total_count in combined.most_common(30):
            r_display = repr(r) if not r.isprintable() or r == " " else r
            h_display = repr(h) if not h.isprintable() or h == " " else h
            # Show per-version counts
            per_v = []
            for v in DSPY_VERSIONS:
                if v in version_results:
                    per_v.append(str(version_results[v]["sub_pairs"].get((r, h), 0)))
            print(f"  {r_display:>4s} → {h_display:<4s}  total={total_count:5d}  [{', '.join(per_v)}]")
    print("└" + "─" * 69 + "┘")

    # 4. Version-specific patterns
    print("\n┌─ Version-Specific Error Patterns ──────────────────────────────────────┐")
    for version in DSPY_VERSIONS:
        if version not in version_results:
            continue
        agg = version_results[version]
        gt = agg["total_gt_chars"]
        total_err = agg["total_substitutions"] + agg["total_deletions"] + agg["total_insertions"]

        # Find what's unusual about this version
        mean_s_rate = sum(
            v["total_substitutions"] / v["total_gt_chars"]
            for v in version_results.values()
        ) / len(version_results)
        mean_d_rate = sum(
            v["total_deletions"] / v["total_gt_chars"]
            for v in version_results.values()
        ) / len(version_results)
        mean_i_rate = sum(
            v["total_insertions"] / v["total_gt_chars"]
            for v in version_results.values()
        ) / len(version_results)

        s_rate = agg["total_substitutions"] / gt
        d_rate = agg["total_deletions"] / gt
        i_rate = agg["total_insertions"] / gt

        deviations = []
        if abs(s_rate - mean_s_rate) / mean_s_rate > 0.15:
            direction = "HIGH" if s_rate > mean_s_rate else "LOW"
            deviations.append(f"S_rate {direction} ({s_rate:.1%} vs avg {mean_s_rate:.1%})")
        if abs(d_rate - mean_d_rate) / mean_d_rate > 0.15:
            direction = "HIGH" if d_rate > mean_d_rate else "LOW"
            deviations.append(f"D_rate {direction} ({d_rate:.1%} vs avg {mean_d_rate:.1%})")
        if abs(i_rate - mean_i_rate) / mean_i_rate > 0.15:
            direction = "HIGH" if i_rate > mean_i_rate else "LOW"
            deviations.append(f"I_rate {direction} ({i_rate:.1%} vs avg {mean_i_rate:.1%})")

        if deviations:
            print(f"\n  {version}:")
            for d in deviations:
                print(f"    → {d}")
    print("└" + "─" * 69 + "┘")


def _save_results(version_results: dict):
    """Save results as JSON for downstream use."""
    output = {}
    for version, agg in version_results.items():
        gt = agg["total_gt_chars"]
        total_err = agg["total_substitutions"] + agg["total_deletions"] + agg["total_insertions"]
        output[version] = {
            "pages": agg["pages_analyzed"],
            "gt_chars": gt,
            "hyp_chars": agg["total_hyp_chars"],
            "cer_approx": round(total_err / gt, 6),
            "correct": agg["total_correct"],
            "substitutions": agg["total_substitutions"],
            "deletions": agg["total_deletions"],
            "insertions": agg["total_insertions"],
            "s_rate": round(agg["total_substitutions"] / gt, 6),
            "d_rate": round(agg["total_deletions"] / gt, 6),
            "i_rate": round(agg["total_insertions"] / gt, 6),
            "garbled_segments": agg["n_garbled_segments"],
            "garbled_chars": agg["garbled_chars"],
            "sub_categories": dict(agg["sub_categories"].most_common()),
            "top_50_confusions": [
                {"ref": r, "hyp": h, "count": c}
                for (r, h), c in agg["sub_pairs"].most_common(50)
            ],
            "top_30_word_subs": [
                {"gt": g, "hyp": h, "count": c}
                for (g, h), c in agg["word_subs"].most_common(30)
            ],
            "top_20_word_insertions": [
                {"word": w, "count": c}
                for w, c in agg["word_ins"].most_common(20)
            ],
            "top_20_word_deletions": [
                {"word": w, "count": c}
                for w, c in agg["word_dels"].most_common(20)
            ],
        }

    out_path = REPORTS_DIR / "error_analysis_dspy_versions.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved detailed results to {out_path}")


if __name__ == "__main__":
    run_analysis()
