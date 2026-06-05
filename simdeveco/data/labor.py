"""
Labor-force participation by age x sex (BLS). External data, expected as CSV
in datasets/. See MODEL.md S11.

Expected file:
- us_lfp_age_sex.csv  columns: year, age, sex, lfp_rate
  lfp_rate in [0, 1].

The monthly master panel already contains aggregate `labor_force_part` and
`women_lfp` series; this loader is for the age-disaggregated rates needed to
gate firm activation per agent.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASETS_DIR = REPO_ROOT / "datasets"

EXPECTED_FILE = "us_lfp_age_sex.csv"
EXPECTED_COLUMNS = ["year", "age", "sex", "lfp_rate"]

_SOURCE_HINT = (
    "BLS Labor Force Statistics from the Current Population Survey "
    "(https://www.bls.gov/cps/) - tabulate LFP rate by single year of age "
    "(or 5-year bin) and sex, 1940-2000."
)


def load_lfp_age_sex(path: Path | str | None = None) -> pd.DataFrame:
    if path is None:
        path = DEFAULT_DATASETS_DIR / EXPECTED_FILE
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Labor data file missing: {p}\n"
            f"Expected schema columns: {EXPECTED_COLUMNS}\n"
            f"Source: {_SOURCE_HINT}"
        )
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{p} missing columns: {missing}")
    return df
