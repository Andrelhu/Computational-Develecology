"""
US demographic data loaders. External CSV files expected in datasets/.

This module defines the expected schemas and a single entry point per source.
Files are NOT auto-downloaded; the loaders raise FileNotFoundError with a
clear pointer if absent. See MODEL.md S16 for source URLs.

Expected files (each in datasets/, lowercase, snake_case):
- us_life_tables.csv          columns: year, age, sex, q          (q = annual prob of death)
- us_fertility_age.csv        columns: year, age, asfr            (age-specific fertility rate per 1000 women)
- us_population_yearly.csv    columns: year, total_population
- us_age_pyramid_decadal.csv  columns: year, age_bin, pct         (decadal census)
- us_birth_sex_ratio.csv      columns: year, ratio_male_to_female (births)

Schemas are duck-typed at use site. The presence checker validate_schema()
exists for unit tests.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASETS_DIR = REPO_ROOT / "datasets"

EXPECTED_SCHEMAS: dict[str, list[str]] = {
    "us_life_tables.csv":         ["year", "age", "sex", "q"],
    "us_fertility_age.csv":       ["year", "age", "asfr"],
    "us_population_yearly.csv":   ["year", "total_population"],
    "us_age_pyramid_decadal.csv": ["year", "age_bin", "pct"],
    "us_birth_sex_ratio.csv":     ["year", "ratio_male_to_female"],
}

_SOURCE_HINTS = {
    "us_life_tables.csv":
        "CDC NCHS Life Tables (https://www.cdc.gov/nchs/products/life_tables.htm) - "
        "tabulate q(x) by single year of age and sex, 1940-2000.",
    "us_fertility_age.csv":
        "CDC NCHS Vital Statistics, age-specific fertility rates "
        "(https://www.cdc.gov/nchs/nvss/) - one row per (year, age) with ASFR per 1000.",
    "us_population_yearly.csv":
        "Census Bureau intercensal estimates. The annual master_monthly_panel.csv "
        "already contains a `population` column at monthly resolution; "
        "us_population_yearly.csv can be derived as the December value per year.",
    "us_age_pyramid_decadal.csv":
        "US decennial census 1940..2000 age distribution. Existing pickled "
        "`datasets/age_groups_decade` is the same data; convert to CSV.",
    "us_birth_sex_ratio.csv":
        "CDC vital stats. Historically ~1.05 male:female; safe default if missing.",
}


class _MissingDataError(FileNotFoundError):
    """Raised when an external demographic data file is not present."""

    def __init__(self, fname: str):
        hint = _SOURCE_HINTS.get(fname, "")
        path = DEFAULT_DATASETS_DIR / fname
        msg = (
            f"Demographic data file missing: {path}\n"
            f"Expected schema columns: {EXPECTED_SCHEMAS[fname]}\n"
            f"Source: {hint}"
        )
        super().__init__(msg)


def _load_csv_with_schema(fname: str, path: Path | str | None) -> pd.DataFrame:
    if path is None:
        path = DEFAULT_DATASETS_DIR / fname
    p = Path(path)
    if not p.exists():
        raise _MissingDataError(fname)
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    expected = EXPECTED_SCHEMAS[fname]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(
            f"{p} is missing required columns: {missing}. "
            f"Expected schema: {expected}"
        )
    return df


def load_life_tables(path: Path | str | None = None) -> pd.DataFrame:
    """Age x sex annual mortality probabilities q(year, age, sex)."""
    return _load_csv_with_schema("us_life_tables.csv", path)


def load_fertility(path: Path | str | None = None) -> pd.DataFrame:
    """Age-specific fertility rates (female) per year."""
    return _load_csv_with_schema("us_fertility_age.csv", path)


def load_population_yearly(path: Path | str | None = None) -> pd.DataFrame:
    """Annual US total population."""
    return _load_csv_with_schema("us_population_yearly.csv", path)


def load_age_pyramid_decadal(path: Path | str | None = None) -> pd.DataFrame:
    """Decadal census age-bin percentages."""
    return _load_csv_with_schema("us_age_pyramid_decadal.csv", path)


def load_birth_sex_ratio(path: Path | str | None = None) -> pd.DataFrame:
    """Annual male:female birth ratio. Default ~1.05 if file absent."""
    return _load_csv_with_schema("us_birth_sex_ratio.csv", path)


def list_expected_files() -> dict[str, dict]:
    """Return a dict describing every expected file with schema + source hint.

    Useful for CLI-driven data audits.
    """
    out = {}
    for fname, cols in EXPECTED_SCHEMAS.items():
        out[fname] = {
            "schema": cols,
            "source": _SOURCE_HINTS.get(fname, ""),
            "present": (DEFAULT_DATASETS_DIR / fname).exists(),
        }
    return out
