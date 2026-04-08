"""Cross-check statistical tests against independent implementations.

Strategy:
  Holm-Bonferroni → our impl vs statsmodels.stats.multitest.multipletests()
  Wilcoxon        → our wrapper vs scipy.stats.wilcoxon directly + Darwin data
  Bootstrap CI    → our impl vs scipy.stats.bootstrap (approximate, different RNG)
"""

import numpy as np
import pytest
from scipy import stats as scipy_stats

from metrics.statistical import bootstrap_ci, holm_bonferroni, wilcoxon_test


# ═══════════════════════════════════════════════
# Holm-Bonferroni cross-check with statsmodels
# ═══════════════════════════════════════════════

class TestHolmBonferroniCrosscheck:
    """Cross-check against statsmodels multipletests(method='holm')."""

    @pytest.fixture(autouse=True)
    def _check_statsmodels(self):
        try:
            from statsmodels.stats.multitest import multipletests  # noqa: F401
        except ImportError:
            pytest.skip("statsmodels not available")

    @pytest.mark.crosscheck
    @pytest.mark.parametrize("p_values", [
        [0.01, 0.04, 0.03, 0.005],
        [0.001, 0.002, 0.003],
        [0.10, 0.20, 0.30],
        [0.50, 0.60],
        [0.001, 0.01, 0.05, 0.10, 0.50],
        [0.025, 0.025, 0.025, 0.025],
        [1e-10, 0.99, 0.5, 0.01, 0.001],
    ], ids=["standard", "all_sig", "none_sig", "capped",
            "spread", "tied", "extreme_range"])
    def test_vs_statsmodels(self, p_values):
        """Our adjusted p-values and significance must match statsmodels."""
        from statsmodels.stats.multitest import multipletests

        ours = holm_bonferroni(p_values)
        reject, adjusted, _, _ = multipletests(p_values, alpha=0.05, method='holm')

        for i in range(len(p_values)):
            assert abs(ours[i]["adjusted_p"] - adjusted[i]) < 1e-9, (
                f"p[{i}]: ours={ours[i]['adjusted_p']}, statsmodels={adjusted[i]}"
            )
            assert ours[i]["significant"] == reject[i], (
                f"p[{i}]: ours={ours[i]['significant']}, statsmodels={reject[i]}"
            )

    def test_empty(self):
        assert holm_bonferroni([]) == []

    def test_preserves_original_order(self):
        """Results must be in original order, not sorted order."""
        ours = holm_bonferroni([0.04, 0.01, 0.03])
        assert ours[0]["raw_p"] == 0.04
        assert ours[1]["raw_p"] == 0.01
        assert ours[2]["raw_p"] == 0.03

    def test_monotonicity_of_adjusted(self):
        """Adjusted p-values (sorted) must be non-decreasing."""
        ours = holm_bonferroni([0.01, 0.04, 0.03, 0.005])
        sorted_adj = sorted(r["adjusted_p"] for r in ours)
        for i in range(1, len(sorted_adj)):
            assert sorted_adj[i] >= sorted_adj[i - 1] - 1e-9

    def test_boundary_p_equals_alpha(self):
        """Edge case: p exactly equals alpha.

        Our implementation uses strict < (adjusted_p < alpha → not significant),
        while statsmodels uses ≤ (adjusted_p ≤ alpha → significant).
        This documents the known difference at the exact boundary.
        """
        ours = holm_bonferroni([0.05])
        assert ours[0]["adjusted_p"] == 0.05
        # Our convention: strict inequality → NOT significant at boundary
        assert ours[0]["significant"] is False

    def test_adjusted_capped_at_1(self):
        """Adjusted p-values must never exceed 1.0."""
        ours = holm_bonferroni([0.50, 0.60, 0.70, 0.80])
        for r in ours:
            assert r["adjusted_p"] <= 1.0


class TestHolmBonferroniKnownValues:
    """Known values from evaluation_verification.md."""

    def test_hb1(self):
        hb = holm_bonferroni([0.01, 0.04, 0.03, 0.005])
        assert hb[0] == {"raw_p": 0.01, "adjusted_p": 0.03, "significant": True}
        assert hb[1] == {"raw_p": 0.04, "adjusted_p": 0.06, "significant": False}
        assert hb[2] == {"raw_p": 0.03, "adjusted_p": 0.06, "significant": False}
        assert hb[3] == {"raw_p": 0.005, "adjusted_p": 0.02, "significant": True}

    def test_hb2_all_sig(self):
        hb = holm_bonferroni([0.001, 0.002, 0.003])
        assert all(r["significant"] for r in hb)

    def test_hb3_none_sig(self):
        hb = holm_bonferroni([0.10, 0.20, 0.30])
        assert not any(r["significant"] for r in hb)

    def test_hb4_capped(self):
        hb = holm_bonferroni([0.50, 0.60])
        assert hb[0]["adjusted_p"] == 1.0
        assert hb[1]["adjusted_p"] == 1.0


# ═══════════════════════════════════════════════
# Wilcoxon cross-check with scipy direct + Darwin data
# ═══════════════════════════════════════════════

class TestWilcoxonCrosscheck:
    """Cross-check our Wilcoxon wrapper against scipy.stats.wilcoxon directly."""

    # Darwin maize data (Fisher 1935): heights of cross- vs self-fertilized plants
    DARWIN_CROSS = [106, 108, 114, 116, 123, 124, 128, 129, 141, 52, 149, 156, 160, 33, 175]
    DARWIN_SELF = [100] * 15

    @pytest.mark.crosscheck
    def test_darwin_maize_statistic(self):
        """Darwin maize data: published statistic = 24.0."""
        result = wilcoxon_test(self.DARWIN_CROSS, self.DARWIN_SELF)
        assert abs(result["statistic"] - 24.0) < 1e-6, f"got {result['statistic']}"

    @pytest.mark.crosscheck
    def test_darwin_maize_pvalue(self):
        """Darwin maize data: published p ≈ 0.041."""
        result = wilcoxon_test(self.DARWIN_CROSS, self.DARWIN_SELF)
        assert abs(result["p_value"] - 0.041260) < 1e-3, f"got {result['p_value']}"
        assert result["significant"] is True

    @pytest.mark.crosscheck
    def test_vs_scipy_direct(self):
        """Our wrapper must produce same result as calling scipy directly."""
        a = np.array(self.DARWIN_CROSS, dtype=float)
        b = np.array(self.DARWIN_SELF, dtype=float)
        diffs = a - b
        nonzero = diffs != 0

        our_result = wilcoxon_test(self.DARWIN_CROSS, self.DARWIN_SELF)
        scipy_result = scipy_stats.wilcoxon(a[nonzero], b[nonzero])

        assert abs(our_result["statistic"] - float(scipy_result.statistic)) < 1e-6
        assert abs(our_result["p_value"] - float(scipy_result.pvalue)) < 1e-4

    @pytest.mark.crosscheck
    @pytest.mark.parametrize("a, b, desc", [
        ([0.1, 0.2, 0.3, 0.4, 0.5], [0.15, 0.25, 0.35, 0.45, 0.55], "shifted"),
        ([1, 2, 3, 4, 5, 6, 7, 8], [2, 3, 4, 5, 6, 7, 8, 9], "offset_by_1"),
        ([10, 20, 30, 40], [15, 18, 35, 38], "mixed"),
    ], ids=["shifted", "offset_by_1", "mixed"])
    def test_vs_scipy_various(self, a, b, desc):
        """Cross-check on various paired data."""
        our_result = wilcoxon_test(a, b)

        a_arr, b_arr = np.array(a, dtype=float), np.array(b, dtype=float)
        diffs = a_arr - b_arr
        nonzero = diffs != 0
        if nonzero.sum() < 2:
            assert our_result["p_value"] == 1.0
            return

        scipy_result = scipy_stats.wilcoxon(a_arr[nonzero], b_arr[nonzero])
        assert abs(our_result["statistic"] - float(scipy_result.statistic)) < 1e-6
        assert abs(our_result["p_value"] - float(scipy_result.pvalue)) < 1e-4

    def test_zero_differences(self):
        """All differences zero → p=1.0, not significant."""
        result = wilcoxon_test([0.5] * 3, [0.5] * 3)
        assert result["p_value"] == 1.0
        assert result["significant"] is False

    def test_single_nonzero(self):
        """Only one non-zero difference → insufficient for test."""
        result = wilcoxon_test([1.0, 0.5, 0.5], [0.5, 0.5, 0.5])
        assert result["p_value"] == 1.0


# ═══════════════════════════════════════════════
# Bootstrap CI cross-check with scipy.stats.bootstrap
# ═══════════════════════════════════════════════

class TestBootstrapCICrosscheck:
    """Cross-check bootstrap CI against scipy.stats.bootstrap (approximate)."""

    @pytest.mark.crosscheck
    def test_constant_data(self):
        """Constant data → zero-width CI."""
        ci = bootstrap_ci([0.5] * 5, n_resamples=1000)
        assert ci["point_estimate"] == 0.5
        assert ci["ci_width"] == 0.0

    @pytest.mark.crosscheck
    def test_point_estimate(self):
        """Point estimate must be the mean of the data."""
        data = [0.1, 0.2, 0.3, 0.4, 0.5]
        ci = bootstrap_ci(data, n_resamples=10000)
        assert ci["point_estimate"] == 0.3

    @pytest.mark.crosscheck
    def test_ci_contains_mean(self):
        """The CI must contain the point estimate."""
        data = [0.1, 0.2, 0.3, 0.4, 0.5]
        ci = bootstrap_ci(data, n_resamples=10000)
        assert ci["ci_lower"] <= ci["point_estimate"] <= ci["ci_upper"]

    @pytest.mark.crosscheck
    def test_determinism(self):
        """Fixed seed must give identical results."""
        data = [0.1, 0.2, 0.3, 0.4, 0.5]
        ci_a = bootstrap_ci(data, n_resamples=10000)
        ci_b = bootstrap_ci(data, n_resamples=10000)
        assert ci_a == ci_b

    @pytest.mark.crosscheck
    def test_ci_width_vs_scipy(self):
        """CI width should be roughly similar to scipy.stats.bootstrap.

        NOTE: Won't match exactly (different RNG paths), but widths should
        be in the same ballpark (within 0.10 absolute for small n).
        """
        data = [0.1, 0.2, 0.3, 0.4, 0.5]
        our_ci = bootstrap_ci(data, n_resamples=10000)

        scipy_ci = scipy_stats.bootstrap(
            (data,), np.mean, method="percentile",
            n_resamples=10000, random_state=42,
        )
        scipy_width = scipy_ci.confidence_interval.high - scipy_ci.confidence_interval.low

        assert abs(our_ci["ci_width"] - scipy_width) < 0.10, (
            f"CI width too different: ours={our_ci['ci_width']:.4f}, scipy={scipy_width:.4f}"
        )

    @pytest.mark.crosscheck
    def test_larger_sample_narrower_ci(self):
        """More data points → narrower CI."""
        small = bootstrap_ci([0.1, 0.5, 0.9], n_resamples=10000)
        large = bootstrap_ci([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], n_resamples=10000)
        assert large["ci_width"] < small["ci_width"], (
            f"Larger sample didn't narrow CI: small={small['ci_width']}, large={large['ci_width']}"
        )

    @pytest.mark.crosscheck
    def test_median_statistic(self):
        """Median bootstrap must use median, not mean."""
        data = [0.1, 0.2, 0.3, 0.4, 100.0]  # outlier
        ci_mean = bootstrap_ci(data, statistic="mean", n_resamples=10000)
        ci_median = bootstrap_ci(data, statistic="median", n_resamples=10000)
        # Median point estimate should be 0.3 (unaffected by outlier)
        assert ci_median["point_estimate"] == 0.3
        # Mean point estimate is pulled up by the outlier
        assert ci_mean["point_estimate"] > ci_median["point_estimate"]

    def test_ci_keys(self):
        """Bootstrap CI must return all expected keys."""
        ci = bootstrap_ci([1.0, 2.0, 3.0])
        expected_keys = {"point_estimate", "ci_lower", "ci_upper", "ci_width",
                         "n_resamples", "confidence"}
        assert set(ci.keys()) == expected_keys
