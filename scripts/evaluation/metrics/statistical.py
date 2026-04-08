"""Statistical testing: Bootstrap CI, Wilcoxon signed-rank, Holm-Bonferroni correction."""

import itertools

import numpy as np
from scipy import stats as scipy_stats


def bootstrap_ci(
    values: list[float],
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    statistic: str = "mean",
    rng_seed: int = 42,
) -> dict:
    """Compute bootstrap confidence interval for a metric.

    Args:
        values: Page-level metric values
        n_resamples: Number of bootstrap resamples
        confidence: Confidence level (default 0.95 for 95% CI)
        statistic: "mean" or "median"
        rng_seed: Random seed for reproducibility

    Returns:
        dict with point_estimate, ci_lower, ci_upper, ci_width
    """
    arr = np.array(values)
    rng = np.random.default_rng(rng_seed)

    stat_fn = np.mean if statistic == "mean" else np.median
    point = float(stat_fn(arr))

    # Bootstrap resampling
    boot_stats = np.empty(n_resamples)
    for idx in range(n_resamples):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_stats[idx] = stat_fn(sample)

    alpha = 1 - confidence
    ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    return {
        "point_estimate": round(point, 6),
        "ci_lower": round(ci_lower, 6),
        "ci_upper": round(ci_upper, 6),
        "ci_width": round(ci_upper - ci_lower, 6),
        "n_resamples": n_resamples,
        "confidence": confidence,
    }


def wilcoxon_test(
    values_a: list[float],
    values_b: list[float],
) -> dict:
    """Wilcoxon signed-rank test for paired page-level metrics.

    Non-parametric test appropriate for CER distributions (non-normal,
    right-skewed). Paired by page to respect the within-page correlation.

    Returns:
        dict with statistic, p_value, significant (at alpha=0.05)
    """
    a = np.array(values_a)
    b = np.array(values_b)
    diffs = a - b

    # If all differences are zero, no test needed
    if np.all(diffs == 0):
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
        }

    # Remove zero differences (Wilcoxon requires non-zero)
    nonzero = diffs != 0
    if nonzero.sum() < 2:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
        }

    result = scipy_stats.wilcoxon(a[nonzero], b[nonzero])
    return {
        "statistic": round(float(result.statistic), 6),
        "p_value": round(float(result.pvalue), 6),
        "significant": float(result.pvalue) < 0.05,
    }


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[dict]:
    """Apply Holm-Bonferroni correction for multiple comparisons.

    Args:
        p_values: Raw p-values from pairwise tests
        alpha: Family-wise error rate

    Returns:
        List of dicts with raw_p, adjusted_p, significant
    """
    n = len(p_values)
    if n == 0:
        return []

    # Sort indices by p-value
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    results = [None] * n
    max_adjusted = 0.0
    for rank, (orig_idx, raw_p) in enumerate(indexed):
        # Holm-Bonferroni: compare p_i with alpha / (n - rank)
        adjusted = raw_p * (n - rank)
        # Enforce monotonicity (adjusted p can't decrease)
        adjusted = max(adjusted, max_adjusted)
        adjusted = min(adjusted, 1.0)
        max_adjusted = adjusted

        results[orig_idx] = {
            "raw_p": round(raw_p, 6),
            "adjusted_p": round(adjusted, 6),
            "significant": adjusted < alpha,
        }

    return results


def pairwise_significance(
    model_page_metrics: dict[str, list[float]],
    n_resamples: int = 10_000,
) -> dict:
    """Run all pairwise Wilcoxon tests with Holm-Bonferroni correction.

    Args:
        model_page_metrics: {model_name: [page_cer_1, page_cer_2, ...]}
            All lists must have the same length (same pages in same order).
        n_resamples: Number of bootstrap resamples for CIs

    Returns:
        dict with:
          - model_cis: {model: bootstrap_ci_dict}
          - pairwise: [{model_a, model_b, wilcoxon, corrected}]
    """
    models = sorted(model_page_metrics.keys())

    # Bootstrap CIs per model
    model_cis = {}
    for model in models:
        model_cis[model] = bootstrap_ci(
            model_page_metrics[model],
            n_resamples=n_resamples,
        )

    # Pairwise Wilcoxon tests
    pairs = list(itertools.combinations(models, 2))
    raw_results = []
    raw_p_values = []

    for model_a, model_b in pairs:
        wtest = wilcoxon_test(
            model_page_metrics[model_a],
            model_page_metrics[model_b],
        )
        raw_results.append({
            "model_a": model_a,
            "model_b": model_b,
            "wilcoxon": wtest,
        })
        raw_p_values.append(wtest["p_value"])

    # Apply correction
    corrected = holm_bonferroni(raw_p_values)

    pairwise = []
    for result, correction in zip(raw_results, corrected):
        result["corrected"] = correction
        pairwise.append(result)

    return {
        "model_cis": model_cis,
        "pairwise": pairwise,
        "n_models": len(models),
        "n_comparisons": len(pairs),
    }
