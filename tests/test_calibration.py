"""Tests for simdeveco.calibration (losses, scenarios, targets, runner)."""
import numpy as np
import pandas as pd
import pytest

from simdeveco.calibration import (
    Target,
    TargetKind,
    build_targets_for_scenario,
    Scenario,
    SCENARIO_A,
    SCENARIO_B,
    scenarios_by_name,
    log_mae,
    wasserstein_1d_share,
    chi2_distribution,
    mape,
    composite_loss,
    CalibrationRunner,
)
from simdeveco.calibration.losses import wasserstein_share_panel
from simdeveco.data import load_monthly_panel, TASTE_DIMS


class TestLogMae:
    def test_zero_when_equal(self):
        x = np.array([10, 20, 30], float)
        assert log_mae(x, x) == 0.0

    def test_symmetric(self):
        a = np.array([10, 20, 30], float)
        b = np.array([15, 18, 35], float)
        assert log_mae(a, b) == pytest.approx(log_mae(b, a))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            log_mae(np.array([1, 2]), np.array([1, 2, 3]))


class TestMape:
    def test_zero_when_equal(self):
        x = np.array([10, 20, 30], float)
        assert mape(x, x) == pytest.approx(0.0, abs=1e-6)

    def test_known_value(self):
        # 10% error on each => 10% mean
        obs = np.array([100, 200], float)
        sim = np.array([110, 220], float)
        assert mape(obs, sim) == pytest.approx(10.0, abs=1e-6)


class TestWasserstein1D:
    def test_zero_when_equal(self):
        p = np.array([0.5, 0.3, 0.2])
        assert wasserstein_1d_share(p, p) == 0.0

    def test_nonnegative(self):
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.1, 0.6, 0.3])
        assert wasserstein_1d_share(p, q) >= 0

    def test_known_value(self):
        # Shift mass by 0.1 from bin 0 to bin 1, no further movement
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.4, 0.4, 0.2])
        # CDFs: p=[0.5,0.8,1.0], q=[0.4,0.8,1.0], abs diff = [0.1,0,0] -> 0.1
        assert wasserstein_1d_share(p, q) == pytest.approx(0.1)

    def test_panel_wrapper(self):
        dates = pd.to_datetime(["2000-01-31", "2000-02-29"])
        obs = pd.DataFrame(
            {"date": np.repeat(dates, 3),
             "dim": ["a", "b", "c"] * 2,
             "share": [0.5, 0.3, 0.2, 0.4, 0.4, 0.2]}
        )
        # Identical simulated -> mean Wasserstein 0
        assert wasserstein_share_panel(obs, obs) == 0.0


class TestChi2:
    def test_zero_when_equal(self):
        x = np.array([10, 20, 30], float)
        assert chi2_distribution(x, x) == pytest.approx(0.0, abs=1e-6)

    def test_nonnegative(self):
        a = np.array([10, 20, 30], float)
        b = np.array([15, 15, 30], float)
        assert chi2_distribution(a, b) >= 0


class TestComposite:
    def test_weighted_sum(self):
        assert composite_loss([(1.0, 0.1), (2.0, 0.2)]) == pytest.approx(0.5)

    def test_empty(self):
        assert composite_loss([]) == 0.0


class TestScenarios:
    def test_scenario_a_window(self):
        assert SCENARIO_A.start_year == 1940
        assert SCENARIO_A.end_year == 1970
        assert SCENARIO_A.n_years == 31
        assert SCENARIO_A.n_months == 31 * 12

    def test_scenario_b_window(self):
        assert SCENARIO_B.start_year == 1970
        assert SCENARIO_B.end_year == 2000
        assert SCENARIO_B.n_years == 31

    def test_lookup_by_name(self):
        assert scenarios_by_name("A_1940_1970") is SCENARIO_A
        assert scenarios_by_name("B_1970_2000") is SCENARIO_B

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError):
            scenarios_by_name("nope")

    def test_cohort_bins_nondecreasing(self):
        for sc in (SCENARIO_A, SCENARIO_B):
            starts = [b[0] for b in sc.cohort_bins]
            assert starts == sorted(starts)


class TestTargetBuilder:
    @pytest.fixture(scope="class")
    def panel(self):
        return load_monthly_panel()

    def test_scenario_a_target_set(self, panel):
        targets = build_targets_for_scenario(SCENARIO_A, panel)
        names = {t.name for t in targets}
        assert {
            "total_issues_monthly",
            "genre_share_monthly",
            "population_yearly",
            "age_pyramid_decadal",
        } <= names

    def test_total_issues_monthly_shape(self, panel):
        targets = build_targets_for_scenario(SCENARIO_A, panel)
        t = next(t for t in targets if t.name == "total_issues_monthly")
        assert t.kind == TargetKind.SCALAR_TIME_SERIES
        # 31 years * 12 months = 372
        assert len(t.observed) == SCENARIO_A.n_months
        assert {"date", "value"} <= set(t.observed.columns)

    def test_genre_share_monthly_long_form(self, panel):
        targets = build_targets_for_scenario(SCENARIO_A, panel)
        t = next(t for t in targets if t.name == "genre_share_monthly")
        assert t.kind == TargetKind.DISTRIBUTION_TIME_SERIES
        # n_months * n_dims
        assert len(t.observed) == SCENARIO_A.n_months * len(TASTE_DIMS)
        assert {"date", "dim", "share"} <= set(t.observed.columns)
        # Per-date shares either sum to 1 (months with data) or 0 (empty months)
        per_date = t.observed.groupby("date")["share"].sum().round(6)
        assert per_date.isin([0.0, 1.0]).all()

    def test_population_yearly_count(self, panel):
        targets = build_targets_for_scenario(SCENARIO_A, panel)
        t = next(t for t in targets if t.name == "population_yearly")
        assert t.kind == TargetKind.SCALAR_YEARLY
        assert len(t.observed) == SCENARIO_A.n_years


class TestCalibrationRunner:
    @pytest.fixture(scope="class")
    def runner(self):
        panel = load_monthly_panel()
        targets = build_targets_for_scenario(SCENARIO_A, panel)
        return CalibrationRunner(scenario=SCENARIO_A, targets=targets)

    def test_held_out_count(self, runner):
        held = runner.held_out_dates()
        expected = int(round(0.15 * SCENARIO_A.n_months))
        assert abs(len(held) - expected) <= 1

    def test_held_out_within_window(self, runner):
        held = runner.held_out_dates()
        years = {d.year for d in held}
        assert min(years) >= SCENARIO_A.start_year
        assert max(years) <= SCENARIO_A.end_year

    def test_held_out_reproducible(self, runner):
        a = runner.held_out_dates()
        b = runner.held_out_dates()
        assert a == b

    def test_evaluate_not_implemented(self, runner):
        with pytest.raises(NotImplementedError):
            runner.evaluate({})
