#!/usr/bin/env python3
"""Generate visualization reports from evaluation metrics.

Auto-discovers {subset}_{edition}_metrics.json files produced by
compute_metrics.py and generates:

  Existing figures:
    - data/evaluation/figures/cer_boxplot_by_model.png
    - data/evaluation/figures/cer_heatmap_model_x_letra.png
    - data/evaluation/figures/cer_by_century.png

  New figures (from expanded evaluation framework):
    - data/evaluation/figures/romein_tiers_waterfall.png
    - data/evaluation/figures/precision_recall_scatter.png
    - data/evaluation/figures/error_decomposition_stacked.png
    - data/evaluation/figures/bootstrap_ci_forest.png
    - data/evaluation/figures/pairwise_significance_heatmap.png

  Tables:
    - data/evaluation/tables/summary_table.md
    - data/evaluation/tables/summary_table.tex

Usage:
  python generate_reports.py                     # auto-discover all metrics
  python generate_reports.py --metrics FILE ...  # explicit metrics files
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = PROJECT_ROOT / "data" / "evaluation" / "reports"
FIGURES_DIR = PROJECT_ROOT / "data" / "evaluation" / "figures"
TABLES_DIR = PROJECT_ROOT / "data" / "evaluation" / "tables"


def discover_metrics_files() -> list[Path]:
    """Find all *_metrics.json files in the reports directory."""
    if not REPORTS_DIR.exists():
        return []
    return sorted(REPORTS_DIR.glob("*_metrics.json"))


def discover_statistical_files() -> list[Path]:
    """Find all *_statistical.json files in the reports directory."""
    if not REPORTS_DIR.exists():
        return []
    return sorted(REPORTS_DIR.glob("*_statistical.json"))


def load_metrics(paths: list[Path]) -> list[dict]:
    """Load per-page metrics from one or more metrics files.

    Handles both the wrapped format from compute_metrics.py:
        {"subset": ..., "results": [...]}
    and legacy flat list format for backward compatibility.
    """
    all_results = []
    for path in paths:
        with open(path) as f:
            data = json.load(f)

        if isinstance(data, dict) and "results" in data:
            all_results.extend(data["results"])
        elif isinstance(data, list):
            all_results.extend(data)
        else:
            print(f"  WARNING: Unexpected format in {path.name}, skipping",
                  file=sys.stderr)

    return all_results


# ---------------------------------------------------------------------------
# Existing figures (unchanged)
# ---------------------------------------------------------------------------

def plot_cer_boxplot(metrics: list[dict]):
    """Box plot of CER distribution per model."""
    import matplotlib.pyplot as plt

    by_model = defaultdict(list)
    for m in metrics:
        by_model[m["model"]].append(m["cer"])

    # Sort by median CER
    sorted_models = sorted(by_model.keys(), key=lambda k: sorted(by_model[k])[len(by_model[k]) // 2])

    fig, ax = plt.subplots(figsize=(12, max(6, len(sorted_models) * 0.5)))
    data = [by_model[m] for m in sorted_models]

    bp = ax.boxplot(data, vert=False, tick_labels=sorted_models, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#4ECDC4")
        patch.set_alpha(0.7)

    ax.set_xlabel("Character Error Rate (CER)")
    ax.set_title("CER Distribution by Model")
    ax.axvline(x=0.14, color="red", linestyle="--", alpha=0.5, label="SOTA procesal (~14%)")
    ax.legend()
    plt.tight_layout()

    out_path = FIGURES_DIR / "cer_boxplot_by_model.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_cer_heatmap(metrics: list[dict]):
    """Heatmap of mean CER: model x letra type."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Aggregate: model x letra -> mean CER
    grid = defaultdict(lambda: defaultdict(list))
    for m in metrics:
        grid[m["model"]][m["letra_normalized"]].append(m["cer"])

    models = sorted(grid.keys())
    letras = sorted(set(m["letra_normalized"] for m in metrics))

    # Build matrix
    matrix = np.full((len(models), len(letras)), np.nan)
    for i, model in enumerate(models):
        for j, letra in enumerate(letras):
            vals = grid[model][letra]
            if vals:
                matrix[i, j] = sum(vals) / len(vals)

    fig, ax = plt.subplots(figsize=(max(8, len(letras) * 1.5), max(6, len(models) * 0.5)))
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(letras)))
    ax.set_xticklabels(letras, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8)

    # Annotate cells
    for i in range(len(models)):
        for j in range(len(letras)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=7)

    plt.colorbar(im, ax=ax, label="Mean CER")
    ax.set_title("Mean CER: Model x Script Type")
    plt.tight_layout()

    out_path = FIGURES_DIR / "cer_heatmap_model_x_letra.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_cer_by_century(metrics: list[dict]):
    """Grouped bar chart: CER by century for each model."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Aggregate
    grid = defaultdict(lambda: defaultdict(list))
    for m in metrics:
        if m["siglo"]:
            grid[m["model"]][m["siglo"]].append(m["cer"])

    models = sorted(grid.keys())
    centuries = sorted(set(m["siglo"] for m in metrics if m["siglo"]))

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.5), 6))
    x = np.arange(len(models))
    width = 0.8 / len(centuries)
    colors = ["#2C3E50", "#E74C3C", "#3498DB", "#2ECC71"]

    for i, century in enumerate(centuries):
        vals = []
        for model in models:
            cers = grid[model][century]
            vals.append(sum(cers) / len(cers) if cers else 0)
        offset = (i - len(centuries) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=f"Siglo {century}", color=colors[i % len(colors)])

    ax.set_xlabel("Model")
    ax.set_ylabel("Mean CER")
    ax.set_title("Mean CER by Century and Model")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.axhline(y=0.14, color="red", linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_path = FIGURES_DIR / "cer_by_century.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# New figures (expanded evaluation framework)
# ---------------------------------------------------------------------------

def plot_romein_waterfall(metrics: list[dict]):
    """Romein 4-tier waterfall: CER at T1/T2/T3/T4 per model.

    Shows how CER drops with increasing normalization, revealing
    where errors come from (segmentation, case, punctuation).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Check if romein data is available
    has_romein = any("romein" in m for m in metrics)
    if not has_romein:
        print("  Skipped: romein_tiers_waterfall.png (no Romein tier data)")
        return

    tiers = ["T1_raw", "T2_nospace", "T3_lowercase", "T4_alnum"]
    tier_labels = ["T1\nRaw", "T2\nNo-space", "T3\nLowercase", "T4\nAlnum"]

    # Aggregate per model
    by_model = defaultdict(lambda: defaultdict(list))
    for m in metrics:
        if "romein" in m:
            for tier in tiers:
                by_model[m["model"]][tier].append(m["romein"][tier])

    models = sorted(by_model.keys())
    if not models:
        return

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 2), 6))
    x = np.arange(len(tiers))
    width = 0.8 / len(models)
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(models), 3)))

    for i, model in enumerate(models):
        means = [
            sum(by_model[model][t]) / len(by_model[model][t])
            if by_model[model][t] else 0
            for t in tiers
        ]
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, means, width, label=model, color=colors[i], alpha=0.85)

    ax.set_xlabel("Normalization Tier")
    ax.set_ylabel("Mean CER")
    ax.set_title("Romein Diagnostic Tiers: CER Under Increasing Normalization")
    ax.set_xticks(x)
    ax.set_xticklabels(tier_labels)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylim(bottom=0)
    plt.tight_layout()

    out_path = FIGURES_DIR / "romein_tiers_waterfall.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_precision_recall_scatter(metrics: list[dict]):
    """Precision vs Recall scatter: each model as a point.

    Models in the top-right are both precise and complete.
    Top-left = cautious (high precision, low recall = omissions).
    Bottom-right = aggressive (low precision, high recall = hallucinations).
    """
    import matplotlib.pyplot as plt

    has_decomp = any("decomposition" in m for m in metrics)
    if not has_decomp:
        print("  Skipped: precision_recall_scatter.png (no decomposition data)")
        return

    by_model = defaultdict(lambda: {"p": [], "r": []})
    for m in metrics:
        if "decomposition" in m:
            by_model[m["model"]]["p"].append(m["decomposition"]["precision"])
            by_model[m["model"]]["r"].append(m["decomposition"]["recall"])

    models = sorted(by_model.keys())
    if not models:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.Set2(range(len(models)))

    for i, model in enumerate(models):
        p_mean = sum(by_model[model]["p"]) / len(by_model[model]["p"])
        r_mean = sum(by_model[model]["r"]) / len(by_model[model]["r"])
        ax.scatter(r_mean, p_mean, s=120, color=colors[i], zorder=5)
        ax.annotate(model, (r_mean, p_mean), fontsize=7,
                    xytext=(5, 5), textcoords="offset points")

    # F1 iso-lines
    import numpy as np
    for f1_val in [0.5, 0.7, 0.8, 0.9, 0.95]:
        recall_range = np.linspace(0.01, 1.0, 200)
        precision_at_f1 = (f1_val * recall_range) / (2 * recall_range - f1_val)
        mask = (precision_at_f1 > 0) & (precision_at_f1 <= 1)
        ax.plot(recall_range[mask], precision_at_f1[mask],
                "--", color="gray", alpha=0.3, linewidth=0.8)
        # Label
        idx = mask.nonzero()[0]
        if len(idx) > 0:
            mid = idx[len(idx) // 2]
            ax.annotate(f"F1={f1_val}", (recall_range[mid], precision_at_f1[mid]),
                        fontsize=6, color="gray", alpha=0.5)

    ax.set_xlabel("Recall (completeness)")
    ax.set_ylabel("Precision (correctness)")
    ax.set_title("Character-Level Precision vs Recall")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = FIGURES_DIR / "precision_recall_scatter.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_error_decomposition(metrics: list[dict]):
    """Stacked bar chart: S/D/I proportions per model.

    Shows whether a model's errors are primarily substitutions (confusion),
    deletions (omissions), or insertions (hallucinations).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    has_decomp = any("decomposition" in m for m in metrics)
    if not has_decomp:
        print("  Skipped: error_decomposition_stacked.png (no decomposition data)")
        return

    by_model = defaultdict(lambda: {"s": [], "d": [], "i": []})
    for m in metrics:
        if "decomposition" in m:
            by_model[m["model"]]["s"].append(m["decomposition"]["s_rate"])
            by_model[m["model"]]["d"].append(m["decomposition"]["d_rate"])
            by_model[m["model"]]["i"].append(m["decomposition"]["i_rate"])

    models = sorted(by_model.keys())
    if not models:
        return

    s_means = [sum(by_model[m]["s"]) / len(by_model[m]["s"]) for m in models]
    d_means = [sum(by_model[m]["d"]) / len(by_model[m]["d"]) for m in models]
    i_means = [sum(by_model[m]["i"]) / len(by_model[m]["i"]) for m in models]

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.2), 6))
    x = np.arange(len(models))

    ax.bar(x, s_means, label="Substitutions", color="#E74C3C", alpha=0.85)
    ax.bar(x, d_means, bottom=s_means, label="Deletions", color="#3498DB", alpha=0.85)
    bottoms = [s + d for s, d in zip(s_means, d_means)]
    ax.bar(x, i_means, bottom=bottoms, label="Insertions", color="#F39C12", alpha=0.85)

    ax.set_xlabel("Model")
    ax.set_ylabel("Error Rate (normalized by ref length)")
    ax.set_title("Error Decomposition: Substitutions / Deletions / Insertions")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.legend()
    plt.tight_layout()

    out_path = FIGURES_DIR / "error_decomposition_stacked.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_bootstrap_ci_forest(stat_files: list[Path]):
    """Forest plot: mean CER with 95% bootstrap CI whiskers per model.

    Shows which models are statistically distinguishable.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if not stat_files:
        print("  Skipped: bootstrap_ci_forest.png (no statistical files)")
        return

    # Combine CIs from all stat files
    all_cis = {}
    for path in stat_files:
        with open(path) as f:
            data = json.load(f)
        for model, ci in data.get("model_cis", {}).items():
            # If model appears in multiple files, keep the one with most pages
            if model not in all_cis:
                all_cis[model] = ci

    if not all_cis:
        print("  Skipped: bootstrap_ci_forest.png (no CI data)")
        return

    # Sort by point estimate
    models = sorted(all_cis.keys(), key=lambda m: all_cis[m]["point_estimate"])

    fig, ax = plt.subplots(figsize=(10, max(4, len(models) * 0.5)))
    y_pos = np.arange(len(models))

    points = [all_cis[m]["point_estimate"] for m in models]
    lowers = [all_cis[m]["point_estimate"] - all_cis[m]["ci_lower"] for m in models]
    uppers = [all_cis[m]["ci_upper"] - all_cis[m]["point_estimate"] for m in models]

    ax.errorbar(points, y_pos, xerr=[lowers, uppers], fmt="o",
                color="#2C3E50", capsize=4, capthick=1.5, markersize=6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=8)
    ax.set_xlabel("Mean CER (with 95% Bootstrap CI)")
    ax.set_title("Bootstrap Confidence Intervals for Mean CER")
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()

    out_path = FIGURES_DIR / "bootstrap_ci_forest.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_pairwise_heatmap(stat_files: list[Path]):
    """Heatmap of corrected p-values from pairwise Wilcoxon tests.

    Upper triangle shows p-values; green = statistically significant.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if not stat_files:
        print("  Skipped: pairwise_significance_heatmap.png (no statistical files)")
        return

    # Use the first stat file with pairwise data
    pairwise_data = None
    for path in stat_files:
        with open(path) as f:
            data = json.load(f)
        if data.get("pairwise"):
            pairwise_data = data
            break

    if not pairwise_data or not pairwise_data.get("pairwise"):
        print("  Skipped: pairwise_significance_heatmap.png (no pairwise data)")
        return

    # Build model list and p-value matrix
    models = set()
    for pair in pairwise_data["pairwise"]:
        models.add(pair["model_a"])
        models.add(pair["model_b"])
    models = sorted(models)
    n = len(models)
    model_idx = {m: i for i, m in enumerate(models)}

    matrix = np.full((n, n), np.nan)
    sig_matrix = np.zeros((n, n), dtype=bool)

    for pair in pairwise_data["pairwise"]:
        i = model_idx[pair["model_a"]]
        j = model_idx[pair["model_b"]]
        p_adj = pair["corrected"]["adjusted_p"]
        matrix[i, j] = p_adj
        matrix[j, i] = p_adj
        sig = pair["corrected"]["significant"]
        sig_matrix[i, j] = sig
        sig_matrix[j, i] = sig

    fig, ax = plt.subplots(figsize=(max(8, n * 0.8), max(6, n * 0.7)))

    # Custom colormap: green for significant, red for non-significant
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "sig", ["#27AE60", "#F1C40F", "#E74C3C"], N=256
    )
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=0.2, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(models, fontsize=7)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            if i != j and not np.isnan(matrix[i, j]):
                p_val = matrix[i, j]
                marker = "*" if sig_matrix[i, j] else ""
                color = "white" if p_val < 0.05 else "black"
                ax.text(j, i, f"{p_val:.3f}{marker}", ha="center", va="center",
                        color=color, fontsize=6)

    plt.colorbar(im, ax=ax, label="Adjusted p-value (Holm-Bonferroni)")
    ax.set_title("Pairwise Significance (Wilcoxon, * = p < 0.05)")
    plt.tight_layout()

    out_path = FIGURES_DIR / "pairwise_significance_heatmap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Table generators
# ---------------------------------------------------------------------------

def _load_summaries() -> dict:
    """Load all summary JSON files and merge."""
    merged = {}
    for path in sorted(REPORTS_DIR.glob("*_summary.json")):
        with open(path) as f:
            data = json.load(f)
        subset = data.get("subset", "")
        edition = data.get("edition", "")
        key = f"{subset}_{edition}"
        for model, stats in data.get("models", {}).items():
            if model not in merged:
                merged[model] = {}
            merged[model][key] = stats
    return merged


def generate_summary_table_md(metrics: list[dict]):
    """Generate Markdown summary table with all core metrics."""
    # Aggregate per model
    by_model = defaultdict(lambda: {
        "cer": [], "wer": [], "cer_n": [], "nls": [],
        "precision": [], "recall": [], "boc": [], "n": 0,
    })

    for m in metrics:
        model = m["model"]
        by_model[model]["cer"].append(m["cer"])
        by_model[model]["wer"].append(m["wer"])
        by_model[model]["n"] = len(by_model[model]["cer"])
        if "core" in m:
            by_model[model]["cer_n"].append(m["core"].get("cer_n", 0))
            by_model[model]["nls"].append(m["core"].get("nls", 0))
        if "decomposition" in m:
            by_model[model]["precision"].append(m["decomposition"].get("precision", 0))
            by_model[model]["recall"].append(m["decomposition"].get("recall", 0))
        if "order_independent" in m:
            by_model[model]["boc"].append(m["order_independent"].get("boc", 0))

    models = sorted(by_model.keys(),
                    key=lambda m: sum(by_model[m]["cer"]) / len(by_model[m]["cer"]))

    def _mean(vals):
        return sum(vals) / len(vals) if vals else 0

    lines = [
        "| Rank | Model | Pages | CER | CER_n | NLS | WER | Precision | Recall | BOC |",
        "|------|-------|-------|-----|-------|-----|-----|-----------|--------|-----|",
    ]
    for rank, model in enumerate(models, 1):
        d = by_model[model]
        lines.append(
            f"| {rank} | {model} | {d['n']} "
            f"| {_mean(d['cer']):.4f} "
            f"| {_mean(d['cer_n']):.4f} "
            f"| {_mean(d['nls']):.4f} "
            f"| {_mean(d['wer']):.4f} "
            f"| {_mean(d['precision']):.4f} "
            f"| {_mean(d['recall']):.4f} "
            f"| {_mean(d['boc']):.4f} |"
        )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLES_DIR / "summary_table.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Saved: {out_path.name}")


def generate_summary_table_tex(metrics: list[dict]):
    """Generate LaTeX summary table (paper-ready)."""
    by_model = defaultdict(lambda: {
        "cer": [], "cer_n": [], "nls": [], "wer": [],
        "precision": [], "recall": [], "n": 0,
    })

    for m in metrics:
        model = m["model"]
        by_model[model]["cer"].append(m["cer"])
        by_model[model]["wer"].append(m["wer"])
        by_model[model]["n"] = len(by_model[model]["cer"])
        if "core" in m:
            by_model[model]["cer_n"].append(m["core"].get("cer_n", 0))
            by_model[model]["nls"].append(m["core"].get("nls", 0))
        if "decomposition" in m:
            by_model[model]["precision"].append(m["decomposition"].get("precision", 0))
            by_model[model]["recall"].append(m["decomposition"].get("recall", 0))

    models = sorted(by_model.keys(),
                    key=lambda m: sum(by_model[m]["cer"]) / len(by_model[m]["cer"]))

    def _mean(vals):
        return sum(vals) / len(vals) if vals else 0

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{HTR Model Evaluation Summary}",
        r"\label{tab:htr-summary}",
        r"\begin{tabular}{clcccccc}",
        r"\toprule",
        r"Rank & Model & Pages & CER$\downarrow$ & CER$_n$$\downarrow$ & NLS$\uparrow$ & WER$\downarrow$ & P$\uparrow$ \\",
        r"\midrule",
    ]

    for rank, model in enumerate(models, 1):
        d = by_model[model]
        # Escape underscores in model names for LaTeX
        model_tex = model.replace("_", r"\_")
        lines.append(
            f"  {rank} & {model_tex} & {d['n']} "
            f"& {_mean(d['cer']):.4f} "
            f"& {_mean(d['cer_n']):.4f} "
            f"& {_mean(d['nls']):.4f} "
            f"& {_mean(d['wer']):.4f} "
            f"& {_mean(d['precision']):.4f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLES_DIR / "summary_table.tex"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate visualization reports from evaluation metrics",
    )
    parser.add_argument(
        "--metrics", nargs="+", type=Path, default=None,
        help="Metrics JSON files to load (default: auto-discover *_metrics.json)",
    )
    args = parser.parse_args()

    if args.metrics:
        metrics_files = args.metrics
    else:
        metrics_files = discover_metrics_files()

    if not metrics_files:
        print("No metrics files found in data/evaluation/reports/.")
        print("Run compute_metrics.py first.")
        sys.exit(1)

    print(f"Loading metrics from {len(metrics_files)} file(s):")
    for f in metrics_files:
        print(f"  {f.name}")

    metrics = load_metrics(metrics_files)
    if not metrics:
        print("No metric entries found.")
        sys.exit(1)

    print(f"Loaded {len(metrics)} metric entries")

    stat_files = discover_statistical_files()
    if stat_files:
        print(f"Found {len(stat_files)} statistical report(s)")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("\nGenerating existing figures...")
    plot_cer_boxplot(metrics)
    plot_cer_heatmap(metrics)
    plot_cer_by_century(metrics)

    print("\nGenerating new diagnostic figures...")
    plot_romein_waterfall(metrics)
    plot_precision_recall_scatter(metrics)
    plot_error_decomposition(metrics)
    plot_bootstrap_ci_forest(stat_files)
    plot_pairwise_heatmap(stat_files)

    print("\nGenerating tables...")
    generate_summary_table_md(metrics)
    generate_summary_table_tex(metrics)

    print(f"\nAll figures saved to {FIGURES_DIR.relative_to(PROJECT_ROOT)}/")
    print(f"All tables saved to {TABLES_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
