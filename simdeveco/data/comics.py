"""
Loaders for the in-repo comic-market data files.

Two files (see datasets/ and MODEL.md S16):
- master_monthly_panel.csv: 1900-01..2000-12, monthly, 43 genre columns +
  aligned monthly macro covariates + event flags + era classification.
- US_Comics_Macro_Enriched.csv: 1938..2000, annual, multi-media context
  (cinema, TV, radio, print) + social covariates + censorship flags.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASETS_DIR = REPO_ROOT / "datasets"

MONTHLY_PANEL_FILE = "master_monthly_panel.csv"
MACRO_ENRICHED_FILE = "US_Comics_Macro_Enriched.csv"


def load_monthly_panel(path: Path | str | None = None) -> pd.DataFrame:
    """
    Load the monthly comic-market panel.

    Returns a DataFrame indexed by month-end date (`date` column parsed
    as datetime). Columns are stripped of leading/trailing whitespace.
    """
    if path is None:
        path = DEFAULT_DATASETS_DIR / MONTHLY_PANEL_FILE
    df = pd.read_csv(path, parse_dates=["date"])
    df.columns = [c.strip() for c in df.columns]
    return df


def load_macro_enriched(path: Path | str | None = None) -> pd.DataFrame:
    """
    Load the annual macro-enriched panel.

    The source file uses inconsistent column naming (leading spaces,
    parenthetical units). Columns are normalized: stripped of whitespace.
    The `Year` column is left as int.
    """
    if path is None:
        path = DEFAULT_DATASETS_DIR / MACRO_ENRICHED_FILE
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df
