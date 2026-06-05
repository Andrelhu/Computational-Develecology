"""
Loss components for calibration. All return a non-negative scalar (float).
See MODEL.md S15 for the composite formula.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def log_mae(observed: np.ndarray, simulated: np.ndarray, eps: float = 1.0) -> float:
    """
    Mean absolute error in log space. `eps` shifts to avoid log(0); set to
    median-scale of the series for stable behavior.
    """
    obs = np.asarray(observed, dtype=float)
    sim = np.asarray(simulated, dtype=float)
    if obs.shape != sim.shape:
        raise ValueError(f"Shape mismatch: {obs.shape} vs {sim.shape}")
    return float(np.mean(np.abs(np.log(obs + eps) - np.log(sim + eps))))


def mape(observed: np.ndarray, simulated: np.ndarray, eps: float = 1e-9) -> float:
    """Mean absolute percentage error (in %)."""
    obs = np.asarray(observed, dtype=float)
    sim = np.asarray(simulated, dtype=float)
    if obs.shape != sim.shape:
        raise ValueError(f"Shape mismatch: {obs.shape} vs {sim.shape}")
    return float(np.mean(np.abs(obs - sim) / (np.abs(obs) + eps)) * 100.0)


def wasserstein_1d_share(
    observed_share: np.ndarray,
    simulated_share: np.ndarray,
) -> float:
    """
    Sliced 1-D Wasserstein-1 distance between two share distributions over
    the same ordered support. Inputs are 1-D non-negative vectors summing to
    1 (treated as distributions over the index). Equivalent to the L1 norm
    of the difference of cumulative distributions.

    For pairs of distributions (multiple dates), call once per date and
    average; helper `wasserstein_share_panel` does this over time.
    """
    o = np.asarray(observed_share, dtype=float)
    s = np.asarray(simulated_share, dtype=float)
    if o.shape != s.shape:
        raise ValueError(f"Shape mismatch: {o.shape} vs {s.shape}")
    cdf_o = np.cumsum(o)
    cdf_s = np.cumsum(s)
    return float(np.sum(np.abs(cdf_o - cdf_s)))


def wasserstein_share_panel(
    observed_long: pd.DataFrame,
    simulated_long: pd.DataFrame,
    date_col: str = "date",
    dim_col: str = "dim",
    share_col: str = "share",
) -> float:
    """
    Mean per-date 1-D Wasserstein-1 between observed and simulated share
    distributions in long form.

    Both dataframes must have columns [date_col, dim_col, share_col] and
    identical sets of (date, dim) pairs.
    """
    obs = (
        observed_long.pivot(index=date_col, columns=dim_col, values=share_col)
        .fillna(0.0)
        .sort_index()
    )
    sim = (
        simulated_long.pivot(index=date_col, columns=dim_col, values=share_col)
        .fillna(0.0)
        .sort_index()
    )
    if not obs.index.equals(sim.index):
        raise ValueError("Date indexes differ between observed and simulated.")
    if not (obs.columns == sim.columns).all():
        # Align column order
        sim = sim.reindex(columns=obs.columns, fill_value=0.0)
    per_date = []
    for d in obs.index:
        per_date.append(wasserstein_1d_share(obs.loc[d].values, sim.loc[d].values))
    return float(np.mean(per_date)) if per_date else 0.0


def chi2_distribution(
    observed_counts: np.ndarray,
    expected_counts: np.ndarray,
    eps: float = 1e-9,
) -> float:
    """
    Pearson chi-square statistic between two count vectors. Inputs may be
    counts or proportions on the same scale; expected_counts must be
    positive after the eps shift.
    """
    o = np.asarray(observed_counts, dtype=float)
    e = np.asarray(expected_counts, dtype=float) + eps
    if o.shape != e.shape:
        raise ValueError(f"Shape mismatch: {o.shape} vs {e.shape}")
    return float(np.sum((o - e) ** 2 / e))


def composite_loss(components: Iterable[tuple[float, float]]) -> float:
    """
    Weighted sum of (weight, loss) pairs. No normalization is performed;
    callers control weighting per MODEL.md S15.
    """
    return float(sum(w * v for w, v in components))
