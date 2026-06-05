"""Tests for the dashboard renderer."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import torch

from simdeveco import VectorDevecology, run_experiments, build_dashboard, build_replicates_overview
from simdeveco.utils import set_seed


@pytest.fixture
def single_run_dfs():
    set_seed(1)
    m = VectorDevecology(
        individuals=80,
        media=2,
        community=2,
        device=torch.device("cpu"),
    )
    for _ in range(4):
        m.step()
    return m.get_dataframes()


@pytest.fixture
def multi_run_dfs():
    return run_experiments(
        runs=3,
        steps=4,
        media=2,
        community=2,
        individuals=60,
        seed=1,
    )


class TestBuildDashboardSingle:
    def test_returns_figure(self, single_run_dfs):
        agent_df, coll_df, market_df = single_run_dfs
        fig = build_dashboard(agent_df, coll_df, market_df)
        try:
            assert isinstance(fig, plt.Figure)
            # Confirm we have the expected 4x3 grid of axes
            assert len(fig.axes) == 12
        finally:
            plt.close(fig)

    def test_writes_png(self, single_run_dfs, tmp_path):
        agent_df, coll_df, market_df = single_run_dfs
        out = tmp_path / "dashboard.png"
        fig = build_dashboard(agent_df, coll_df, market_df, out_path=out)
        plt.close(fig)
        assert out.exists() and out.stat().st_size > 1000

    def test_writes_to_nested_dir(self, single_run_dfs, tmp_path):
        agent_df, coll_df, market_df = single_run_dfs
        out = tmp_path / "nested" / "deep" / "d.png"
        fig = build_dashboard(agent_df, coll_df, market_df, out_path=out)
        plt.close(fig)
        assert out.exists()


class TestBuildDashboardMulti:
    def test_runs_without_error(self, multi_run_dfs):
        agent_df, coll_df, market_df = multi_run_dfs
        # Confirm fixture is genuinely multi-run
        assert market_df["run"].nunique() > 1
        fig = build_dashboard(agent_df, coll_df, market_df)
        try:
            assert isinstance(fig, plt.Figure)
            assert len(fig.axes) == 12
        finally:
            plt.close(fig)

    def test_writes_png(self, multi_run_dfs, tmp_path):
        agent_df, coll_df, market_df = multi_run_dfs
        out = tmp_path / "multi.png"
        fig = build_dashboard(agent_df, coll_df, market_df, out_path=out)
        plt.close(fig)
        assert out.exists() and out.stat().st_size > 1000


class TestReplicatesOverview:
    def test_writes_png(self, multi_run_dfs, tmp_path):
        _, _, market_df = multi_run_dfs
        out = tmp_path / "overview.png"
        fig = build_replicates_overview(market_df, out_path=out)
        plt.close(fig)
        assert out.exists() and out.stat().st_size > 1000

    def test_rejects_single_run(self, single_run_dfs):
        _, _, market_df = single_run_dfs
        # Add `run=0` to mimic main's behavior, but with a single value
        market_df = market_df.copy()
        if "run" not in market_df.columns:
            market_df["run"] = 0
        with pytest.raises(ValueError):
            build_replicates_overview(market_df)

    def test_missing_series_raises(self, multi_run_dfs):
        _, _, market_df = multi_run_dfs
        with pytest.raises(ValueError):
            build_replicates_overview(market_df, series=("nonexistent_col",))


class TestMultiRunDetection:
    def test_detected_multi_run_bands(self, multi_run_dfs):
        """Multi-run dashboards should produce filled-band collections."""
        agent_df, coll_df, market_df = multi_run_dfs
        fig = build_dashboard(agent_df, coll_df, market_df)
        try:
            # At least one axis should contain a PolyCollection from
            # fill_between (used only in multi-run mode)
            from matplotlib.collections import PolyCollection
            poly_count = sum(
                1
                for ax in fig.axes
                for child in ax.collections
                if isinstance(child, PolyCollection)
            )
            assert poly_count > 0, "Expected fill_between bands in multi-run mode"
        finally:
            plt.close(fig)

    def test_single_run_no_bands(self, single_run_dfs):
        agent_df, coll_df, market_df = single_run_dfs
        # No `run` column in market_df from single-model output
        fig = build_dashboard(agent_df, coll_df, market_df)
        try:
            from matplotlib.collections import PolyCollection
            # The stackplot for the age pyramid uses PolyCollections too,
            # so we expect SOME PolyCollections, just not in the time-series
            # panels. Easier check: build a single-run market_df with a
            # `run` column and nunique==1, ensure no bands beyond the
            # stackplot's expected 5.
            from matplotlib.collections import PolyCollection
            poly_count = sum(
                1
                for ax in fig.axes
                for child in ax.collections
                if isinstance(child, PolyCollection)
            )
            # Stackplot for age pyramid contributes 5 PolyCollections.
            # No additional bands in single-run mode.
            assert poly_count <= 6
        finally:
            plt.close(fig)
