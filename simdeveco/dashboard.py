"""
Dashboard renderer for VectorDevecology run outputs.

Consumes the three dataframes produced by `VectorDevecology.get_dataframes`
(or `run_experiments`) and renders a multi-panel matplotlib figure
summarizing population, demographics, market, and collective dynamics.

Two modes, auto-detected from the presence of a `run` column with
multiple unique values:

- single-run:   plot the raw time series and final-state distributions
- multi-run:    plot per-step mean (solid) with 10-90% percentile band
                (shaded) across replicates, plus pooled distributions

CLI entrypoint exists in `simdeveco.main` (`--dashboard PATH`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

# Use non-interactive backend so the renderer works in headless / CI.
matplotlib.use("Agg", force=False)  # noqa: E402
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Curated panels: ordering controls dashboard layout.
TIME_SERIES_GROUPS: dict[str, list[str]] = {
    "Population": ["alive_count"],
    "Age structure (%)": [
        "pct_0_19", "pct_20_39", "pct_40_59", "pct_60_79", "pct_80_plus",
    ],
    "Age quantiles": ["age_q10", "age_q50", "age_q90"],
    "Role share": ["prop_children", "prop_adults"],
    "Taste similarity (within-pop)": ["sim_q10", "sim_q50", "sim_q90"],
    "Generational similarity": ["youth_mid", "mid_old", "youth_old"],
    "Products / month": ["products"],
    "Median org size": ["median_household_size", "median_community_size"],
}


def _is_multi_run(df: pd.DataFrame) -> bool:
    return "run" in df.columns and df["run"].nunique() > 1


def _bands(
    df: pd.DataFrame,
    series: str,
    group_col: str = "time",
    run_col: str = "run",
) -> pd.DataFrame:
    """Per-time mean and 10/90 percentile bands across runs."""
    g = df.groupby(group_col)[series]
    out = pd.DataFrame({
        "mean": g.mean(),
        "p10":  g.quantile(0.10),
        "p90":  g.quantile(0.90),
    })
    return out.reset_index()


def _plot_timeseries_panel(
    ax: plt.Axes,
    market_df: pd.DataFrame,
    series_list: list[str],
    title: str,
) -> None:
    multi = _is_multi_run(market_df)
    available = [s for s in series_list if s in market_df.columns]
    if not available:
        ax.set_title(f"{title} (no data)")
        ax.axis("off")
        return

    for s in available:
        if multi:
            band = _bands(market_df, s)
            ax.plot(band["time"], band["mean"], label=s, linewidth=1.2)
            ax.fill_between(band["time"], band["p10"], band["p90"], alpha=0.18)
        else:
            ax.plot(market_df["time"], market_df[s], label=s, linewidth=1.0)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("step (month)")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax.grid(True, alpha=0.25)


def _plot_age_pyramid_stacked(ax: plt.Axes, market_df: pd.DataFrame) -> None:
    """Stacked area of the 5 age buckets over time."""
    cols = ["pct_0_19", "pct_20_39", "pct_40_59", "pct_60_79", "pct_80_plus"]
    if any(c not in market_df.columns for c in cols):
        ax.set_title("Age pyramid (no data)")
        ax.axis("off")
        return
    multi = _is_multi_run(market_df)
    if multi:
        # Use cross-run mean for each bucket.
        agg = market_df.groupby("time")[cols].mean().reset_index()
        x = agg["time"].values
        stacks = [agg[c].values for c in cols]
    else:
        x = market_df["time"].values
        stacks = [market_df[c].values for c in cols]
    ax.stackplot(x, stacks, labels=cols, alpha=0.8)
    ax.set_title("Age pyramid (stacked)", fontsize=10)
    ax.set_xlabel("step (month)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="upper right", ncol=1, frameon=False)


def _plot_agent_age_hist(ax: plt.Axes, agent_df: pd.DataFrame) -> None:
    alive = agent_df[agent_df["alive"] == 1]
    if alive.empty:
        ax.set_title("Agent ages (no alive)")
        ax.axis("off")
        return
    ax.hist(alive["age"], bins=range(0, 101, 5), alpha=0.85, edgecolor="white")
    ax.set_title(f"Final agent ages (n={len(alive)} alive)", fontsize=10)
    ax.set_xlabel("age")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.25)


def _plot_consumed_hist(ax: plt.Axes, agent_df: pd.DataFrame) -> None:
    alive = agent_df[agent_df["alive"] == 1]
    if alive.empty or "consumed_count" not in alive.columns:
        ax.set_title("Consumption (no data)")
        ax.axis("off")
        return
    vals = alive["consumed_count"].values
    if vals.max() == vals.min():
        ax.set_title("Consumption (uniform)")
        ax.axis("off")
        return
    ax.hist(vals, bins=30, alpha=0.85, edgecolor="white")
    ax.set_title("Final consumed_count distribution", fontsize=10)
    ax.set_xlabel("products consumed (cum.)")
    ax.set_ylabel("agents")
    ax.grid(True, alpha=0.25)


def _plot_collective_size_hist(ax: plt.Axes, coll_df: pd.DataFrame) -> None:
    if coll_df.empty or "type" not in coll_df.columns:
        ax.set_title("Collective sizes (no data)")
        ax.axis("off")
        return
    types = sorted(coll_df["type"].unique())
    for t in types:
        sizes = coll_df.loc[coll_df["type"] == t, "size"].values
        if len(sizes) == 0:
            continue
        ax.hist(sizes, bins=20, alpha=0.55, label=t, edgecolor="white")
    ax.set_title("Final collective-size distributions", fontsize=10)
    ax.set_xlabel("size")
    ax.set_ylabel("count")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.25)


def _plot_role_pie(ax: plt.Axes, agent_df: pd.DataFrame) -> None:
    alive = agent_df[agent_df["alive"] == 1]
    if alive.empty:
        ax.set_title("Roles (no alive)")
        ax.axis("off")
        return
    n_child = int((alive["role"] == 0).sum())
    n_adult = int((alive["role"] == 1).sum())
    if n_child + n_adult == 0:
        ax.axis("off")
        return
    ax.pie(
        [n_child, n_adult],
        labels=[f"children ({n_child})", f"adults ({n_adult})"],
        autopct="%1.0f%%",
        startangle=90,
        colors=["#88c", "#c88"],
    )
    ax.set_title("Final role share", fontsize=10)


def build_dashboard(
    agent_df: pd.DataFrame,
    coll_df: pd.DataFrame,
    market_df: pd.DataFrame,
    out_path: Path | str | None = None,
    title: str | None = None,
) -> plt.Figure:
    """
    Build a single multi-panel summary figure.

    Parameters
    ----------
    agent_df, coll_df, market_df
        Outputs of `VectorDevecology.get_dataframes` or `run_experiments`.
    out_path
        If provided, the figure is saved to this path (PNG or PDF inferred
        from extension). Parent directory is created if needed.
    title
        Optional suptitle. Auto-derived if omitted.

    Returns
    -------
    matplotlib.figure.Figure
    """
    multi = _is_multi_run(market_df)
    n_runs = int(market_df["run"].nunique()) if "run" in market_df.columns else 1
    n_steps = int(market_df["time"].nunique())

    if title is None:
        if multi:
            title = f"Devecology run dashboard - {n_runs} replicates x {n_steps} steps"
        else:
            title = f"Devecology run dashboard - single run, {n_steps} steps"

    # 4 x 3 grid: 8 timeseries panels + 4 final-state panels
    fig, axes = plt.subplots(4, 3, figsize=(15, 14))
    flat = axes.ravel()

    ts_titles = list(TIME_SERIES_GROUPS.keys())
    # Slot 0: population
    _plot_timeseries_panel(flat[0], market_df, TIME_SERIES_GROUPS["Population"], "Population (alive)")
    # Slot 1: age pyramid stacked (special)
    _plot_age_pyramid_stacked(flat[1], market_df)
    # Slot 2: age quantiles
    _plot_timeseries_panel(flat[2], market_df, TIME_SERIES_GROUPS["Age quantiles"], "Age quantiles")
    # Slot 3: role share
    _plot_timeseries_panel(flat[3], market_df, TIME_SERIES_GROUPS["Role share"], "Role share")
    # Slot 4: taste similarity
    _plot_timeseries_panel(flat[4], market_df, TIME_SERIES_GROUPS["Taste similarity (within-pop)"], "Taste similarity (within-pop)")
    # Slot 5: generational similarity
    _plot_timeseries_panel(flat[5], market_df, TIME_SERIES_GROUPS["Generational similarity"], "Generational similarity (cohort cosine)")
    # Slot 6: products / month
    _plot_timeseries_panel(flat[6], market_df, TIME_SERIES_GROUPS["Products / month"], "Products per step")
    # Slot 7: median org size
    _plot_timeseries_panel(flat[7], market_df, TIME_SERIES_GROUPS["Median org size"], "Median organization sizes")

    # Final-state panels (use last-replicate agent/coll snapshot if multi)
    # Convention: agent_df / coll_df from run_experiments already concat all runs.
    if "run" in agent_df.columns and multi:
        last_run = agent_df["run"].max()
        agent_view = agent_df[agent_df["run"] == last_run]
        coll_view = coll_df[coll_df["run"] == last_run]
        snapshot_note = f" (run {last_run})"
    else:
        agent_view = agent_df
        coll_view = coll_df
        snapshot_note = ""

    _plot_agent_age_hist(flat[8], agent_view)
    _plot_consumed_hist(flat[9], agent_view)
    _plot_collective_size_hist(flat[10], coll_view)
    _plot_role_pie(flat[11], agent_view)

    if snapshot_note:
        # Annotate the snapshot-based panels so multi-run reader knows source
        for ax in flat[8:12]:
            ax.set_title(ax.get_title() + snapshot_note, fontsize=9)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=140, bbox_inches="tight")

    return fig


def build_replicates_overview(
    market_df: pd.DataFrame,
    out_path: Path | str | None = None,
    series: Iterable[str] = ("alive_count", "sim_q50", "products"),
) -> plt.Figure:
    """
    Multi-run-only spaghetti overview: one line per replicate per selected
    series. Useful for seeing variance and stochastic outliers across runs.
    """
    if not _is_multi_run(market_df):
        raise ValueError("build_replicates_overview requires a multi-run market_df.")

    series = [s for s in series if s in market_df.columns]
    n = len(series)
    if n == 0:
        raise ValueError("No requested series found in market_df.")

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for ax, s in zip(axes[0], series):
        for r, g in market_df.groupby("run"):
            ax.plot(g["time"], g[s], alpha=0.55, linewidth=0.9)
        # Mean overlay
        mean = market_df.groupby("time")[s].mean()
        ax.plot(mean.index, mean.values, color="black", linewidth=2.0, label="mean")
        ax.set_title(s, fontsize=10)
        ax.set_xlabel("step (month)")
        ax.legend(fontsize=8, frameon=False)
        ax.grid(True, alpha=0.25)

    fig.suptitle(f"Replicate overview - {market_df['run'].nunique()} runs", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=140, bbox_inches="tight")

    return fig
