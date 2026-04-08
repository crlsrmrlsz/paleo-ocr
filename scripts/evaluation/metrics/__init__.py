"""HTR evaluation metrics package.

Re-exports all metric functions for convenient access:

    from metrics import compute_all_core, compute_romein_tiers, ...
"""

from .core import (
    compute_all_core,
    compute_cer,
    compute_cer_n,
    compute_nls,
    compute_ser,
    compute_wer,
)
from .decomposition import compute_decomposition, compute_opcodes
from .order_independent import compute_boc, compute_order_independent
from .romein import (
    TIER_NORMALIZERS,
    compute_romein_tiers,
    normalize_t1,
    normalize_t2,
    normalize_t3,
    normalize_t4,
)
from .statistical import (
    bootstrap_ci,
    holm_bonferroni,
    pairwise_significance,
    wilcoxon_test,
)

__all__ = [
    # Core
    "compute_cer",
    "compute_cer_n",
    "compute_nls",
    "compute_wer",
    "compute_ser",
    "compute_all_core",
    # Romein
    "compute_romein_tiers",
    "normalize_t1",
    "normalize_t2",
    "normalize_t3",
    "normalize_t4",
    "TIER_NORMALIZERS",
    # Decomposition
    "compute_opcodes",
    "compute_decomposition",
    # Order-independent
    "compute_boc",
    "compute_order_independent",
    # Statistical
    "bootstrap_ci",
    "wilcoxon_test",
    "holm_bonferroni",
    "pairwise_significance",
]
