#!/usr/bin/env python3
"""Compute multi-tier HTR evaluation metrics for all model outputs against GT.

Supports per-subset/edition evaluation:
  --subset {codea,toledo,all}   — which dataset to evaluate
  --edition {paleographic,critical,editorial,all} — which GT edition

Marker-aware evaluation modes:
  --mode {strict,content,lenient}

Semantic metrics:
  --skip-semantic   — skip semantic similarity and NER (faster, no model downloads)

Output (per subset/edition):
  - data/evaluation/reports/{subset}_{edition}_metrics.json
  - data/evaluation/reports/{subset}_{edition}_summary.json
  - data/evaluation/reports/{subset}_{edition}_ranking.json
  - data/evaluation/reports/{subset}_{edition}_statistical.json  (if >1 model)

When --subset=all --edition=all:
  - Runs all valid combos: (codea,paleographic), (codea,critical), (toledo,editorial)
  - Generates combined_summary.json with cross-subset comparison
"""

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
SUBSETS_DIR = PROJECT_ROOT / "data" / "subsets"
REPORTS_DIR = PROJECT_ROOT / "data" / "evaluation" / "reports"

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from pipeline_config import BOOTSTRAP_N_RESAMPLES, VALID_COMBOS, load_manifest  # noqa: E402

from evaluation.metrics import (  # noqa: E402
    compute_all_core,
    compute_decomposition,
    compute_order_independent,
    compute_romein_tiers,
    pairwise_significance,
)

VALID_MODES = ("strict", "content", "lenient")

# Markers with embedded text content
_MARKERS_WITH_TEXT = {
    "firma", "tachado", "margen", "interlineado",
    "sobrescrito", "encabezamiento", "nota", r"mano[^:]*",
}

# Markers without text content (visual-only symbols)
_MARKERS_EMPTY = {"signo", "rúbrica", "cruz", "ilegible", "blanco", "roto"}


def get_gt_dir(subset: str, edition: str) -> Path:
    """Resolve ground truth directory for a given subset/edition."""
    return SUBSETS_DIR / subset / "ground_truth" / edition


def strip_markers(text: str, mode: str) -> str:
    """Process GT markers for evaluation.

    Modes:
      "strict"  — keep everything as-is (markers are part of CER)
      "content" — strip marker wrappers, keep contained text where readable:
                   [firma: text] → text, [signo] → "", [ilegible] → ""
                   [blanco] → " " (space), [interlineado: text] → text
      "lenient" — remove entire marked regions (closest to legacy behavior),
                   except [interlineado/sobrescrito/encabezamiento: text] → text
    """
    if mode == "strict":
        return text

    _KEEP_TEXT_MARKERS = {"interlineado", "sobrescrito", "encabezamiento", "firma", "mano[^:]*"}

    if mode == "content":
        for name in _MARKERS_WITH_TEXT:
            if name in _KEEP_TEXT_MARKERS:
                text = re.sub(
                    rf"\[{name}\s*:\s*(.*?)\]", r"\1", text,
                    flags=re.IGNORECASE | re.DOTALL,
                )
            else:
                text = re.sub(
                    rf"\[{name}\s*:\s*.*?\]", "", text,
                    flags=re.IGNORECASE | re.DOTALL,
                )

        for name in _MARKERS_EMPTY:
            if name == "blanco":
                text = re.sub(r"\[blanco\]", " ", text, flags=re.IGNORECASE)
            else:
                text = re.sub(
                    rf"\[{name}\]", "", text, flags=re.IGNORECASE,
                )

        # Strip editorial letter brackets for fair OCR evaluation: [e]llo → ello
        text = re.sub(r"\[([a-záéíóúñç]{1,5})\]", r"\1", text)

    elif mode == "lenient":
        for name in _MARKERS_WITH_TEXT:
            if name in ("interlineado", "sobrescrito", "encabezamiento"):
                text = re.sub(
                    rf"\[{name}\s*:\s*(.*?)\]", r"\1", text,
                    flags=re.IGNORECASE | re.DOTALL,
                )
            else:
                text = re.sub(
                    rf"\[{name}\s*:\s*.*?\]", "", text,
                    flags=re.IGNORECASE | re.DOTALL,
                )

        for name in _MARKERS_EMPTY:
            text = re.sub(rf"\[{name}\]", "", text, flags=re.IGNORECASE)

        # Strip editorial letter brackets for fair OCR evaluation: [e]llo → ello
        text = re.sub(r"\[([a-záéíóúñç]{1,5})\]", r"\1", text)

    # Clean up extra whitespace introduced by removals
    text = re.sub(r"  +", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def _normalize_whitespace(text: str) -> str:
    """Collapse all whitespace (including newlines) to single spaces for evaluation.

    CER measures transcription accuracy, not line segmentation. Toledo and CODEA
    encode line breaks with different fidelity (Toledo's ``/`` markers are sparse
    in some documents), so comparing newlines inflates error rates.
    """
    return re.sub(r"\s+", " ", text).strip()


def evaluate_model(
    model_name: str,
    manifest: list[dict],
    gt_dir: Path,
    mode: str = "content",
    skip_semantic: bool = False,
) -> list[dict]:
    """Evaluate a single model against ground truth. Returns per-page results.

    Computes all metric tiers: core, Romein diagnostic, decomposition,
    order-independent, and optionally semantic.
    """
    norm_dir = RESULTS_DIR / model_name / "normalized"

    if not norm_dir.exists():
        return []

    # Lazy-import semantic module only when needed
    compute_semantic = None
    if not skip_semantic:
        try:
            from evaluation.metrics.semantic import compute_all_semantic
            compute_semantic = compute_all_semantic
        except ImportError as e:
            print(f"  WARNING: Semantic metrics unavailable ({e}), skipping.",
                  file=sys.stderr)

    from pipeline_config import EXCLUDED_PAGES

    results = []
    for entry in manifest:
        page_id = entry["page_id"]

        if page_id in EXCLUDED_PAGES:
            continue

        gt_path = gt_dir / f"{page_id}.txt"
        hyp_path = norm_dir / f"{page_id}.txt"

        if not gt_path.exists() or not hyp_path.exists():
            continue

        gt_text = gt_path.read_text(encoding="utf-8").strip()
        hyp_text = hyp_path.read_text(encoding="utf-8").strip()

        # Apply marker stripping to both GT and hypothesis
        gt_eval = strip_markers(gt_text, mode)
        hyp_eval = strip_markers(hyp_text, mode)

        # Collapse whitespace FIRST (existing behavior), then compute all metrics.
        # Romein tiers further normalize from this baseline.
        gt_eval = _normalize_whitespace(gt_eval)
        hyp_eval = _normalize_whitespace(hyp_eval)

        # --- Tier 1: Core metrics ---
        core = compute_all_core(gt_eval, hyp_eval)

        # --- Tier 2: Romein diagnostic tiers ---
        romein = compute_romein_tiers(gt_eval, hyp_eval)

        # --- Tier 2: Error decomposition ---
        decomp = compute_decomposition(gt_eval, hyp_eval)

        # --- Tier 3: Order-independent ---
        order_ind = compute_order_independent(gt_eval, hyp_eval, core["cer"])

        # Build result dict — top-level cer/wer for backward compatibility
        result = {
            "model": model_name,
            "page_id": page_id,
            "dataset": entry["dataset"],
            "doc_id": entry["doc_id"],
            "siglo": entry.get("siglo", ""),
            "letra_normalized": entry.get("letra_normalized", ""),
            "copista": entry.get("copista", ""),
            "tipologia": entry.get("tipologia", ""),
            # Backward-compatible top-level keys
            "cer": round(core["cer"], 4),
            "wer": round(core["wer"], 4),
            "gt_chars": len(gt_eval),
            "hyp_chars": len(hyp_eval),
            # Extended metrics nested by tier
            "core": core,
            "romein": romein,
            "decomposition": decomp,
            "order_independent": order_ind,
        }

        # --- Tier 4: Semantic metrics (optional) ---
        if compute_semantic is not None:
            result["semantic"] = compute_semantic(gt_eval, hyp_eval)

        results.append(result)

    return results


def _safe_mean(values: list[float]) -> float:
    """Mean that handles empty lists."""
    return statistics.mean(values) if values else 0.0


def _safe_stdev(values: list[float]) -> float:
    """Stdev that handles < 2 values."""
    return statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate_results(all_results: list[dict]) -> dict:
    """Compute aggregate statistics from per-page results."""
    by_model = defaultdict(list)
    for r in all_results:
        by_model[r["model"]].append(r)

    summary = {}
    for model, results in sorted(by_model.items()):
        cers = [r["cer"] for r in results]
        wers = [r["wer"] for r in results]

        model_summary = {
            "n_pages": len(results),
            # Core metrics (backward-compatible)
            "cer_mean": round(_safe_mean(cers), 4),
            "cer_median": round(statistics.median(cers), 4),
            "cer_std": round(_safe_stdev(cers), 4),
            "cer_min": round(min(cers), 4),
            "cer_max": round(max(cers), 4),
            "wer_mean": round(_safe_mean(wers), 4),
            "wer_median": round(statistics.median(wers), 4),
        }

        # Extended core metrics
        for key in ("cer_n", "nls", "ser"):
            vals = [r["core"][key] for r in results if "core" in r]
            if vals:
                model_summary[f"{key}_mean"] = round(_safe_mean(vals), 4)

        # Romein tiers
        romein_agg = {}
        for tier in ("T1_raw", "T2_nospace", "T3_lowercase", "T4_alnum"):
            vals = [r["romein"][tier] for r in results if "romein" in r]
            if vals:
                romein_agg[tier] = round(_safe_mean(vals), 4)
        if romein_agg:
            model_summary["romein_mean"] = romein_agg

        # Decomposition
        for key in ("precision", "recall", "f1", "s_rate", "d_rate", "i_rate"):
            vals = [r["decomposition"][key] for r in results if "decomposition" in r]
            if vals:
                model_summary[f"decomp_{key}_mean"] = round(_safe_mean(vals), 4)

        # Order-independent
        for key in ("boc", "delta_cer"):
            vals = [r["order_independent"][key] for r in results if "order_independent" in r]
            if vals:
                model_summary[f"{key}_mean"] = round(_safe_mean(vals), 4)

        # Semantic (if present)
        sem_keys = ("semantic_similarity", "ner_precision", "ner_recall", "ner_f1")
        for key in sem_keys:
            vals = [r["semantic"][key] for r in results if "semantic" in r]
            if vals:
                model_summary[f"{key}_mean"] = round(_safe_mean(vals), 4)

        # Stratified by century
        by_siglo = defaultdict(list)
        for r in results:
            by_siglo[r["siglo"]].append(r["cer"])
        model_summary["by_siglo"] = {
            s: round(_safe_mean(cers), 4)
            for s, cers in sorted(by_siglo.items())
        }

        # Stratified by letra type
        by_letra = defaultdict(list)
        for r in results:
            by_letra[r["letra_normalized"]].append(r["cer"])
        model_summary["by_letra"] = {
            lt: round(_safe_mean(cers), 4)
            for lt, cers in sorted(by_letra.items())
        }

        summary[model] = model_summary

    return summary


def build_ranking(summary: dict) -> list[dict]:
    """Rank models by mean CER (ascending)."""
    ranking = []
    for model, stats in summary.items():
        ranking.append({
            "rank": 0,
            "model": model,
            "cer_mean": stats["cer_mean"],
            "cer_median": stats["cer_median"],
            "wer_mean": stats["wer_mean"],
            "nls_mean": stats.get("nls_mean", None),
            "n_pages": stats["n_pages"],
        })

    ranking.sort(key=lambda x: x["cer_mean"])
    for i, entry in enumerate(ranking):
        entry["rank"] = i + 1

    return ranking


def run_statistical_tests(
    all_results: list[dict],
    subset: str,
    edition: str,
) -> dict | None:
    """Run pairwise statistical tests if >1 model has results.

    Returns statistical report dict or None.
    """
    # Group page-level CERs by model, aligned by page_id
    by_model = defaultdict(dict)
    for r in all_results:
        by_model[r["model"]][r["page_id"]] = r["cer"]

    models = sorted(by_model.keys())
    if len(models) < 2:
        return None

    # Find pages common to all models
    common_pages = set.intersection(*(set(by_model[m].keys()) for m in models))
    if len(common_pages) < 5:
        print(f"  Statistical tests skipped: only {len(common_pages)} common pages")
        return None

    common_pages = sorted(common_pages)
    model_page_metrics = {
        model: [by_model[model][p] for p in common_pages]
        for model in models
    }

    result = pairwise_significance(
        model_page_metrics,
        n_resamples=BOOTSTRAP_N_RESAMPLES,
    )
    result["common_pages"] = len(common_pages)
    result["subset"] = subset
    result["edition"] = edition

    return result


def run_evaluation(
    subset: str,
    edition: str,
    mode: str,
    model_names: list[str],
    skip_semantic: bool = False,
) -> dict | None:
    """Run evaluation for a specific subset/edition combo.

    Returns dict with results, summary, ranking, or None if no results.
    """
    manifest = load_manifest(subset)
    if not manifest:
        print(f"  No manifest for {subset}")
        return None

    gt_dir = get_gt_dir(subset, edition)
    if not gt_dir.exists():
        print(f"  GT directory not found: {gt_dir}")
        return None

    print(f"\n{'='*70}")
    print(f"Evaluating: subset={subset}, edition={edition}, mode={mode}")
    print(f"  GT dir: {gt_dir.relative_to(PROJECT_ROOT)}")
    print(f"  Manifest: {len(manifest)} pages")

    all_results = []
    for model_name in model_names:
        results = evaluate_model(
            model_name, manifest, gt_dir,
            mode=mode, skip_semantic=skip_semantic,
        )
        all_results.extend(results)
        if results:
            mean_cer = statistics.mean(r["cer"] for r in results)
            print(f"  {model_name}: {len(results)} pages, mean CER = {mean_cer:.4f}")
        else:
            print(f"  {model_name}: no results")

    if not all_results:
        print("  No results to aggregate.")
        return None

    summary = aggregate_results(all_results)
    ranking = build_ranking(summary)

    # Also compute all modes for comparison (core CER only, for speed)
    mode_comparison = {}
    for m in VALID_MODES:
        mode_results = []
        for model_name in model_names:
            mode_results.extend(
                evaluate_model(
                    model_name, manifest, gt_dir,
                    mode=m, skip_semantic=True,
                )
            )
        if mode_results:
            by_model = defaultdict(list)
            for r in mode_results:
                by_model[r["model"]].append(r["cer"])
            mode_comparison[m] = {
                model: round(statistics.mean(cers), 4)
                for model, cers in sorted(by_model.items())
            }

    # Save reports
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = f"{subset}_{edition}"

    metrics_path = REPORTS_DIR / f"{prefix}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "subset": subset,
                "edition": edition,
                "mode": mode,
                "results": all_results,
            },
            f, indent=2, ensure_ascii=False,
        )

    summary_path = REPORTS_DIR / f"{prefix}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "subset": subset,
                "edition": edition,
                "mode": mode,
                "models": summary,
                "mode_comparison": mode_comparison,
            },
            f, indent=2, ensure_ascii=False,
        )

    ranking_path = REPORTS_DIR / f"{prefix}_ranking.json"
    with open(ranking_path, "w") as f:
        json.dump(
            {
                "subset": subset,
                "edition": edition,
                "mode": mode,
                "ranking": ranking,
            },
            f, indent=2, ensure_ascii=False,
        )

    # Statistical tests (if multiple models)
    stat_result = run_statistical_tests(all_results, subset, edition)
    if stat_result:
        stat_path = REPORTS_DIR / f"{prefix}_statistical.json"
        with open(stat_path, "w") as f:
            json.dump(stat_result, f, indent=2, ensure_ascii=False)
        print(f"  Statistical tests: {stat_result['n_comparisons']} pairwise comparisons")

    # Print ranking table
    print(f"\n{'Rank':<6}{'Model':<30}{'CER Mean':<12}{'CER Median':<12}{'NLS Mean':<12}{'Pages'}")
    print("-" * 82)
    for r in ranking:
        nls_str = f"{r['nls_mean']:.4f}" if r.get("nls_mean") is not None else "N/A"
        print(f"{r['rank']:<6}{r['model']:<30}{r['cer_mean']:<12.4f}{r['cer_median']:<12.4f}{nls_str:<12}{r['n_pages']}")
    print("=" * 82)

    return {
        "subset": subset,
        "edition": edition,
        "summary": summary,
        "ranking": ranking,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute multi-tier HTR evaluation metrics",
    )
    parser.add_argument(
        "--mode", choices=VALID_MODES, default="content",
        help="Marker handling mode (default: content)",
    )
    parser.add_argument(
        "--subset", choices=["codea", "toledo", "all"], default="all",
        help="Which dataset subset to evaluate (default: all)",
    )
    parser.add_argument(
        "--edition",
        choices=["paleographic", "critical", "editorial", "all"],
        default="all",
        help="Which GT edition to evaluate against (default: all)",
    )
    parser.add_argument(
        "--skip-semantic", action="store_true",
        help="Skip semantic metrics (faster, no model downloads needed)",
    )
    args = parser.parse_args()

    if not RESULTS_DIR.exists():
        print("No data/results/ directory found.")
        return

    # Find all models with normalized output (supports nested groups)
    model_names = []
    for d in sorted(RESULTS_DIR.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        if (d / "normalized").exists():
            model_names.append(d.name)
        else:
            for sub in sorted(d.iterdir()):
                if sub.is_dir() and (sub / "normalized").exists():
                    model_names.append(f"{d.name}/{sub.name}")

    if not model_names:
        print("No normalized model outputs found. Run normalize_output.py first.")
        return

    print(f"Found {len(model_names)} models with normalized outputs.")
    if not args.skip_semantic:
        print("Semantic metrics enabled (use --skip-semantic to disable)")

    # Determine which subset/edition combos to run
    combos = []
    if args.subset == "all":
        subsets = list(VALID_COMBOS.keys())
    else:
        subsets = [args.subset]

    for subset in subsets:
        valid_editions = VALID_COMBOS.get(subset, ())
        if args.edition == "all":
            editions = valid_editions
        elif args.edition in valid_editions:
            editions = [args.edition]
        else:
            print(f"Skipping {subset}: edition '{args.edition}' not valid (valid: {valid_editions})")
            continue
        for edition in editions:
            combos.append((subset, edition))

    if not combos:
        print("No valid subset/edition combinations to evaluate.")
        return

    print(f"Running {len(combos)} evaluation(s): {combos}")

    # Run all evaluations
    all_eval_results = []
    for subset, edition in combos:
        result = run_evaluation(
            subset, edition, args.mode, model_names,
            skip_semantic=args.skip_semantic,
        )
        if result:
            all_eval_results.append(result)

    # Generate combined summary if multiple evaluations
    if len(all_eval_results) > 1:
        combined = {}
        for result in all_eval_results:
            key = f"{result['subset']}_{result['edition']}"
            for model, stats in result["summary"].items():
                if model not in combined:
                    combined[model] = {}
                combined[model][key] = stats["cer_mean"]

        combined_path = REPORTS_DIR / "combined_summary.json"
        with open(combined_path, "w") as f:
            json.dump(
                {
                    "mode": args.mode,
                    "evaluations": [
                        f"{r['subset']}_{r['edition']}" for r in all_eval_results
                    ],
                    "model_cer_by_evaluation": combined,
                },
                f, indent=2, ensure_ascii=False,
            )

        # Print combined summary table
        eval_keys = [f"{r['subset']}_{r['edition']}" for r in all_eval_results]
        print(f"\n{'='*70}")
        print(f"Combined Summary (Mean CER, mode={args.mode})")
        print("-" * 70)
        header = f"{'Model':<30}"
        for ek in eval_keys:
            header += f"{ek:<20}"
        print(header)
        print("-" * 70)
        for model in model_names:
            row = f"{model:<30}"
            for ek in eval_keys:
                cer = combined.get(model, {}).get(ek, "-")
                if isinstance(cer, float):
                    row += f"{cer:<20.4f}"
                else:
                    row += f"{cer:<20}"
            print(row)
        print("=" * 70)

    print(f"\nReports saved to {REPORTS_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
