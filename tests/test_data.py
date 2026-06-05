"""Tests for simdeveco.data."""
import numpy as np
import pandas as pd
import pytest

from simdeveco.data import (
    load_monthly_panel,
    load_macro_enriched,
    project_panel_to_taste_dims,
    TASTE_DIMS,
    MONTHLY_GENRE_CROSSWALK,
    ANNUAL_GENRE_CROSSWALK,
)
from simdeveco.data import demographics, labor, migration


class TestMonthlyPanel:
    @pytest.fixture(scope="class")
    def panel(self):
        return load_monthly_panel()

    def test_loads_nonempty(self, panel):
        assert len(panel) > 1000

    def test_has_date_column(self, panel):
        assert "date" in panel.columns
        assert pd.api.types.is_datetime64_any_dtype(panel["date"])

    def test_has_total_issues_column(self, panel):
        assert "total_issues" in panel.columns
        assert (panel["total_issues"].fillna(0) >= 0).all()

    def test_covers_1940_2000(self, panel):
        years = panel["date"].dt.year
        assert years.min() <= 1940
        assert years.max() >= 2000

    def test_no_leading_trailing_whitespace_in_columns(self, panel):
        for c in panel.columns:
            assert c == c.strip()


class TestMacroEnriched:
    @pytest.fixture(scope="class")
    def macro(self):
        return load_macro_enriched()

    def test_loads_nonempty(self, macro):
        assert len(macro) > 50

    def test_has_year_column(self, macro):
        assert "Year" in macro.columns

    def test_year_range(self, macro):
        assert macro["Year"].min() <= 1940
        assert macro["Year"].max() >= 2000


class TestGenreCrosswalk:
    def test_taste_dims_unique(self):
        assert len(TASTE_DIMS) == len(set(TASTE_DIMS))

    def test_monthly_crosswalk_targets_valid(self):
        for src, target in MONTHLY_GENRE_CROSSWALK.items():
            assert target in TASTE_DIMS, (
                f"Monthly crosswalk maps {src!r} to {target!r} which is not a TASTE_DIM"
            )

    def test_annual_crosswalk_targets_valid(self):
        for src, target in ANNUAL_GENRE_CROSSWALK.items():
            assert target in TASTE_DIMS, (
                f"Annual crosswalk maps {src!r} to {target!r} which is not a TASTE_DIM"
            )

    def test_all_panel_genres_covered(self):
        """Every monthly-panel genre column should be in the crosswalk."""
        panel = load_monthly_panel()
        # Genre columns are panel columns 2..44 by position; equivalently,
        # everything between total_issues (excl) and genre_entropy (excl).
        cols = list(panel.columns)
        start = cols.index("total_issues") + 1
        end = cols.index("genre_entropy")
        genre_cols = cols[start:end]
        unmapped = [c for c in genre_cols if c not in MONTHLY_GENRE_CROSSWALK]
        assert not unmapped, f"Unmapped panel genre columns: {unmapped}"


class TestProjection:
    def test_output_columns_are_taste_dims(self):
        panel = load_monthly_panel()
        out = project_panel_to_taste_dims(panel.head(10))
        assert list(out.columns) == TASTE_DIMS

    def test_normalized_rows_sum_to_one_or_zero(self):
        panel = load_monthly_panel()
        out = project_panel_to_taste_dims(panel.head(20), normalize=True)
        sums = out.sum(axis=1).round(6)
        # Every row sums to 1 (has data) or 0 (entirely empty for that month)
        assert sums.isin([0.0, 1.0]).all()

    def test_unnormalized_preserves_totals(self):
        panel = load_monthly_panel()
        sample = panel.head(20)
        out = project_panel_to_taste_dims(sample, normalize=False)
        # Sum across all taste dims equals sum across all mapped source cols
        mapped_cols = [c for c in MONTHLY_GENRE_CROSSWALK if c in sample.columns]
        expected = sample[mapped_cols].fillna(0).sum(axis=1).values
        actual = out.sum(axis=1).values
        np.testing.assert_allclose(actual, expected, atol=1e-9)

    def test_known_genre_mapping(self):
        # Construct a synthetic frame where only one genre column has values
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2000-01-31"]),
                "superhero": [10],
                "romance": [0],
            }
        )
        out = project_panel_to_taste_dims(df, normalize=False)
        assert out.loc[0, "superhero"] == 10
        assert out.loc[0, "romance_drama"] == 0


class TestExternalDataLoaders:
    """The external loaders raise informative errors when files are absent."""

    def test_life_tables_missing_raises(self):
        with pytest.raises(FileNotFoundError) as e:
            demographics.load_life_tables(path="/nonexistent/life_tables.csv")
        assert "life_tables" in str(e.value).lower() or "missing" in str(e.value).lower()

    def test_lfp_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            labor.load_lfp_age_sex(path="/nonexistent/lfp.csv")

    def test_migration_totals_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            migration.load_migration_totals(path="/nonexistent/mig.csv")

    def test_list_expected_files_present_flag(self):
        info = demographics.list_expected_files()
        for fname, meta in info.items():
            assert "schema" in meta and "present" in meta
            # Since none of these files are in datasets/ yet, present is False.
            assert meta["present"] is False
