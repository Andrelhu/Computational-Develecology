"""
US net migration by age x sex (INS/DHS yearbook + Census ACS). External data.
See MODEL.md S6 and S16.

Expected files:
- us_migration_yearly_totals.csv  columns: year, net_migration
  Positive = net inflow.
- us_migration_age_sex_dist.csv   columns: year, age, sex, share
  Share within (year) sums to 1 across (age, sex). If only an unconditional
  age-sex distribution is available, repeat it per year.

Out-migration is folded into the net inflow series in v0; explicit
out-migration is deferred.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASETS_DIR = REPO_ROOT / "datasets"

TOTALS_FILE = "us_migration_yearly_totals.csv"
DIST_FILE = "us_migration_age_sex_dist.csv"

EXPECTED_SCHEMAS = {
    TOTALS_FILE: ["year", "net_migration"],
    DIST_FILE:   ["year", "age", "sex", "share"],
}

_SOURCE_HINTS = {
    TOTALS_FILE:
        "DHS Office of Immigration Statistics Yearbook + Census Bureau "
        "intercensal net migration estimates.",
    DIST_FILE:
        "ACS migration flows by age and sex; if granular yearly data unavailable, "
        "use a single decadal distribution and repeat per year within the decade.",
}


def _load(fname: str, path: Path | str | None) -> pd.DataFrame:
    if path is None:
        path = DEFAULT_DATASETS_DIR / fname
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Migration data file missing: {p}\n"
            f"Expected schema: {EXPECTED_SCHEMAS[fname]}\n"
            f"Source: {_SOURCE_HINTS[fname]}"
        )
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in EXPECTED_SCHEMAS[fname] if c not in df.columns]
    if missing:
        raise ValueError(f"{p} missing columns: {missing}")
    return df


def load_migration_totals(path: Path | str | None = None) -> pd.DataFrame:
    return _load(TOTALS_FILE, path)


def load_migration_age_sex_dist(path: Path | str | None = None) -> pd.DataFrame:
    return _load(DIST_FILE, path)
