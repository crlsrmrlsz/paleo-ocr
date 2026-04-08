#!/usr/bin/env python3
"""Verify all evaluation metrics against authoritative reference values.

Test cases sourced from: dinglehopper, jiwer, rapidfuzz, SciPy docs,
Holm (1979), and hand computation cross-referenced with these tools.

See docs/evaluation_verification.md for full documentation of each test case.
"""

import sys
sys.path.insert(0, "scripts/evaluation")
sys.path.insert(0, "scripts")

passed = 0
failed = 0
errors = []


def check(name, condition, detail=""):
    global passed, failed, errors
    if condition:
        passed += 1
    else:
        failed += 1
        errors.append(f"  FAIL: {name} — {detail}")


# ═══════════════════════════════════════════════
# TIER 1: Core Metrics
# ═══════════════════════════════════════════════

from metrics.core import compute_cer, compute_cer_n, compute_nls, compute_wer, compute_ser

# --- CER ---
check("CER-C1", compute_cer("a", "a") == 0.0)
check("CER-C2", compute_cer("a", "b") == 1.0)
check("CER-C3", compute_cer("Foo", "Bar") == 1.0)
check("CER-C4", compute_cer("Foo", "") == 1.0)
check("CER-C5", compute_cer("", "") == 0.0)
check("CER-C6", compute_cer("", "Foo") == 1.0, "our convention (dinglehopper returns inf)")
check("CER-C7", abs(compute_cer("Foo", "Food") - 1/3) < 1e-9)
check("CER-C8", abs(compute_cer("Fnord", "Food") - 0.4) < 1e-9)
check("CER-C9", abs(compute_cer("Muell", "Mull") - 0.2) < 1e-9, f"got {compute_cer('Muell', 'Mull')}")
check("CER-C10", abs(compute_cer("X", "X X Y Y") - 6.0) < 1e-9)
check("CER-C11", abs(compute_cer("kitten", "sitting") - 0.5) < 1e-9)

# --- CER_n ---
check("CER_n-N1", compute_cer_n("abc", "abc") == 0.0)
check("CER_n-N2", compute_cer_n("", "") == 0.0)
check("CER_n-N3", compute_cer_n("", "abc") == 1.0)
check("CER_n-N4", compute_cer_n("abc", "xyz") == 1.0)
check("CER_n-N5", abs(compute_cer_n("Foo", "Food") - 0.25) < 1e-9)
check("CER_n-N6", abs(compute_cer_n("kitten", "sitting") - 3/7) < 1e-9)
check("CER_n-N7", abs(compute_cer_n("Muell", "Mull") - 0.2) < 1e-9, f"got {compute_cer_n('Muell', 'Mull')}")
check("CER_n-N8", abs(compute_cer_n("ab", "xyz") - 1.0) < 1e-9)

# --- NLS ---
check("NLS-L1", compute_nls("", "") == 1.0)
check("NLS-L2", compute_nls("abc", "abc") == 1.0)
check("NLS-L3", compute_nls("abc", "") == 0.0)
check("NLS-L4", compute_nls("", "abc") == 0.0)
check("NLS-L5", abs(compute_nls("kitten", "sitting") - 4/7) < 1e-9)
check("NLS-L6", abs(compute_nls("lewenstein", "levenshtein") - 9/11) < 1e-9)
check("NLS-L7", abs(compute_nls("Foo", "Food") - 0.75) < 1e-9)
check("NLS-L8", abs(compute_nls("Fnord", "Food") - 0.6) < 1e-9)

# --- WER ---
check("WER-W1", compute_wer("X", "X") == 0.0)
check("WER-W2", compute_wer("X", "Y") == 1.0)
check("WER-W3", compute_wer("X", "X X Y Y") == 3.0)
check("WER-W4", abs(compute_wer("X Y X", "X Z") - 2/3) < 1e-6)
check("WER-W5", compute_wer("X", "Y Z") == 2.0)
check("WER-W6", compute_wer("this is a test", "this is a test") == 0.0)
check("WER-W7", compute_wer("this is a test", "this a test") == 0.25)
check("WER-W8", compute_wer("", "") == 0.0)
check("WER-W9", compute_wer("  ", "hello") == 1.0)

# --- SER ---
check("SER-S1", compute_ser(["hello", "world"], ["hello", "world"]) == 0.0)
check("SER-S2", compute_ser(["hello", "world"], ["hello", "World"]) == 0.5)
check("SER-S3", abs(compute_ser(["a", "b", "c"], ["a", "b"]) - 1/3) < 1e-9)
check("SER-S4", abs(compute_ser(["a"], ["a", "b", "c"]) - 2/3) < 1e-9)
check("SER-S5", compute_ser([], ["hello"]) == 1.0)
check("SER-S6", compute_ser([], []) == 0.0)
check("SER-S7", compute_ser(["x"], ["y"]) == 1.0)


# ═══════════════════════════════════════════════
# TIER 2: Diagnostic Metrics
# ═══════════════════════════════════════════════

from metrics.romein import compute_romein_tiers, normalize_t1, normalize_t2, normalize_t3, normalize_t4

# --- Romein Tiers ---
# R1: Hello World vs helloworld!
r1 = compute_romein_tiers("Hello World", "helloworld!")
check("Romein-R1-T1", abs(r1["T1_raw"] - 4/11) < 1e-5, f"got {r1['T1_raw']}")
check("Romein-R1-T2", abs(r1["T2_nospace"] - 3/11) < 1e-5, f"got {r1['T2_nospace']}")
check("Romein-R1-T3", abs(r1["T3_lowercase"] - 1/11) < 1e-5, f"got {r1['T3_lowercase']}")
check("Romein-R1-T4", r1["T4_alnum"] == 0.0, f"got {r1['T4_alnum']}")
check("Romein-R1-mono", r1["T1_raw"] >= r1["T2_nospace"] >= r1["T3_lowercase"] >= r1["T4_alnum"])

# R2: paleographic example
r2 = compute_romein_tiers("Don Juan, anno 1542", "don jvan. anno 1542")
check("Romein-R2-T1", abs(r2["T1_raw"] - 4/19) < 1e-5, f"got {r2['T1_raw']}")
check("Romein-R2-T4", abs(r2["T4_alnum"] - 1/19) < 1e-5, f"got {r2['T4_alnum']}")
check("Romein-R2-mono", r2["T1_raw"] >= r2["T2_nospace"] >= r2["T3_lowercase"] >= r2["T4_alnum"])

# Normalization functions
check("Norm-T1", normalize_t1("hello\t world") == "hello world")
check("Norm-T2", normalize_t2("Hello World") == "HelloWorld")
check("Norm-T3", normalize_t3("Hello World") == "helloworld")
check("Norm-T4", normalize_t4("Hello, World! 123") == "helloworld123")

# --- Decomposition ---
from metrics.decomposition import compute_opcodes, compute_decomposition

# D1: kitten → sitting
ops1 = compute_opcodes("kitten", "sitting")
check("Decomp-D1-ops", ops1 == {"substitutions": 2, "deletions": 0, "insertions": 1, "correct": 4},
      f"got {ops1}")
d1 = compute_decomposition("kitten", "sitting")
check("Decomp-D1-P", abs(d1["precision"] - 4/7) < 1e-5)
check("Decomp-D1-R", abs(d1["recall"] - 4/6) < 1e-5)
check("Decomp-D1-F1", abs(d1["f1"] - 32/52) < 1e-5)

# D2: abcde → ace (pure deletions)
ops2 = compute_opcodes("abcde", "ace")
check("Decomp-D2-ops", ops2 == {"substitutions": 0, "deletions": 2, "insertions": 0, "correct": 3},
      f"got {ops2}")
d2 = compute_decomposition("abcde", "ace")
check("Decomp-D2-P", d2["precision"] == 1.0)
check("Decomp-D2-R", abs(d2["recall"] - 0.6) < 1e-5)

# D3: abc → aXbYcZ (pure insertions)
ops3 = compute_opcodes("abc", "aXbYcZ")
check("Decomp-D3-ops", ops3 == {"substitutions": 0, "deletions": 0, "insertions": 3, "correct": 3},
      f"got {ops3}")
d3 = compute_decomposition("abc", "aXbYcZ")
check("Decomp-D3-P", d3["precision"] == 0.5)
check("Decomp-D3-R", d3["recall"] == 1.0)

# D4: ab → xyz (unequal replace)
ops4 = compute_opcodes("ab", "xyz")
check("Decomp-D4-ops", ops4 == {"substitutions": 2, "deletions": 0, "insertions": 1, "correct": 0},
      f"got {ops4}")


# ═══════════════════════════════════════════════
# TIER 3: Order-Independent Metrics
# ═══════════════════════════════════════════════

from metrics.order_independent import compute_boc, compute_order_independent

check("BOC-B1", compute_boc("abc", "abc") == 0.0)
check("BOC-B2", compute_boc("", "") == 0.0)
check("BOC-B3", compute_boc("abc", "abcd") == 0.25)
check("BOC-B4", abs(compute_boc("aab", "abb") - 2/3) < 1e-9)
check("BOC-B5", compute_boc("abc", "cba") == 0.0, "order-invariant property")
check("BOC-B6-BUG", compute_boc("abc", "xyz") == 2.0, "KNOWN BUG: exceeds [0,1] bound")
check("BOC-B7-BUG", compute_boc("aaaa", "bbbb") == 2.0, "KNOWN BUG: maximum is 2.0")

# delta_CER
cer_rev = compute_cer("abc", "cba")
oi_rev = compute_order_independent("abc", "cba", cer_rev)
check("delta-DC1", oi_rev["boc"] == 0.0)
check("delta-DC1-val", abs(oi_rev["delta_cer"] - round(cer_rev, 6)) < 1e-9, "pure ordering error (rounded)")


# ═══════════════════════════════════════════════
# STATISTICAL LAYER
# ═══════════════════════════════════════════════

from metrics.statistical import bootstrap_ci, wilcoxon_test, holm_bonferroni

# --- Bootstrap CI ---
ci1 = bootstrap_ci([0.5]*5, n_resamples=1000)
check("Boot-BS1-point", ci1["point_estimate"] == 0.5)
check("Boot-BS1-width", ci1["ci_width"] == 0.0)

ci3 = bootstrap_ci([0.1, 0.2, 0.3, 0.4, 0.5], n_resamples=10000)
check("Boot-BS3-point", ci3["point_estimate"] == 0.3)
check("Boot-BS3-CI", ci3["ci_lower"] < 0.3 < ci3["ci_upper"])

ci4a = bootstrap_ci([0.1, 0.2, 0.3, 0.4, 0.5], n_resamples=10000)
ci4b = bootstrap_ci([0.1, 0.2, 0.3, 0.4, 0.5], n_resamples=10000)
check("Boot-BS4-determinism", ci4a == ci4b, "fixed seed must give identical results")

# --- Wilcoxon ---
a_darwin = [106, 108, 114, 116, 123, 124, 128, 129, 141, 52, 149, 156, 160, 33, 175]
b_darwin = [100] * 15
wt1 = wilcoxon_test(a_darwin, b_darwin)
check("Wilcoxon-WT1-stat", abs(wt1["statistic"] - 24.0) < 1e-6, f"got {wt1['statistic']}")
check("Wilcoxon-WT1-p", abs(wt1["p_value"] - 0.041260) < 1e-3, f"got {wt1['p_value']}")
check("Wilcoxon-WT1-sig", wt1["significant"] == True)

wt2 = wilcoxon_test([0.5]*3, [0.5]*3)
check("Wilcoxon-WT2-zero", wt2["p_value"] == 1.0)

# --- Holm-Bonferroni ---
hb1 = holm_bonferroni([0.01, 0.04, 0.03, 0.005])
check("HB1-H1", hb1[0] == {"raw_p": 0.01,  "adjusted_p": 0.03,  "significant": True},  f"got {hb1[0]}")
check("HB1-H2", hb1[1] == {"raw_p": 0.04,  "adjusted_p": 0.06,  "significant": False}, f"got {hb1[1]}")
check("HB1-H3", hb1[2] == {"raw_p": 0.03,  "adjusted_p": 0.06,  "significant": False}, f"got {hb1[2]}")
check("HB1-H4", hb1[3] == {"raw_p": 0.005, "adjusted_p": 0.02,  "significant": True},  f"got {hb1[3]}")

hb2 = holm_bonferroni([0.001, 0.002, 0.003])
check("HB2-all-sig", all(r["significant"] for r in hb2))

hb3 = holm_bonferroni([0.10, 0.20, 0.30])
check("HB3-none-sig", not any(r["significant"] for r in hb3))

hb4 = holm_bonferroni([0.50, 0.60])
check("HB4-cap", hb4[0]["adjusted_p"] == 1.0 and hb4[1]["adjusted_p"] == 1.0)


# ═══════════════════════════════════════════════
# DOCUMENT-LEVEL TEST CASES
# ═══════════════════════════════════════════════

# Lorem Ipsum Bad (dinglehopper): 591-char document, 8 known edits
gt_lorem = ("Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam "
            "nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam "
            "erat, sed diam voluptua. At vero eos et accusam et justo duo "
            "dolores et ea rebum. Stet clita kasd gubergren, no sea takimata "
            "sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit "
            "amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor "
            "invidunt ut labore et dolore magna aliquyam erat, sed diam "
            "voluptua. At vero eos et accusam et justo duo dolores et ea "
            "rebum. Stet clita kasd gubergren, no sea takimata sanctus est "
            "Lorem ipsum dolor sit amet.")
check("DOC-lorem-gt-len", len(gt_lorem) == 591, f"GT length: {len(gt_lorem)}")
check("DOC-lorem-gt-words", len(gt_lorem.split()) == 100, f"GT words: {len(gt_lorem.split())}")

# Construct hypothesis by applying 6 known errors (8 total edit ops)
hyp_lorem = gt_lorem
idx = hyp_lorem.index("nonumy");        hyp_lorem = hyp_lorem[:idx] + "nonu yy" + hyp_lorem[idx+6:]
idx = hyp_lorem.index(" vero ");        hyp_lorem = hyp_lorem[:idx] + " Vero " + hyp_lorem[idx+6:]
idx = hyp_lorem.index("takimata");      hyp_lorem = hyp_lorem[:idx] + "iakimata" + hyp_lorem[idx+8:]
first = hyp_lorem.index("amet,"); second = hyp_lorem.index("amet,", first+1)
hyp_lorem = hyp_lorem[:second] + "armet," + hyp_lorem[second+5:]
idx = hyp_lorem.index("nonumy");        hyp_lorem = hyp_lorem[:idx] + "nonurny" + hyp_lorem[idx+6:]
idx = hyp_lorem.rindex("ipsum");        hyp_lorem = hyp_lorem[:idx] + "Ipsum" + hyp_lorem[idx+5:]

import editdistance as _ed
check("DOC-lorem-edit-dist", _ed.eval(gt_lorem, hyp_lorem) == 8,
      f"edit distance: {_ed.eval(gt_lorem, hyp_lorem)}")
check("DOC-lorem-CER", abs(compute_cer(gt_lorem, hyp_lorem) - 8/591) < 1e-6,
      f"CER: {compute_cer(gt_lorem, hyp_lorem)}")
check("DOC-lorem-WER", abs(compute_wer(gt_lorem, hyp_lorem) - 7/100) < 1e-6,
      f"WER: {compute_wer(gt_lorem, hyp_lorem)}")
check("DOC-lorem-CER_n<=CER",
      compute_cer_n(gt_lorem, hyp_lorem) <= compute_cer(gt_lorem, hyp_lorem))
check("DOC-lorem-NLS", 0.95 < compute_nls(gt_lorem, hyp_lorem) <= 1.0)

# Fraktur fragment: 2 known errors in document context
gt_frak = "das Verſproene glei den Augenblick"
ocr_frak = "das Verfproene glei den Augemblick"
check("DOC-fraktur-CER", abs(compute_cer(gt_frak, ocr_frak) - 2/len(gt_frak)) < 1e-6,
      f"CER: {compute_cer(gt_frak, ocr_frak)}")


# ═══════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"Verification complete: {passed} passed, {failed} failed")
print(f"{'='*60}")

if errors:
    print("\nFailures:")
    for e in errors:
        print(e)
    sys.exit(1)
else:
    print("\nAll checks passed")
