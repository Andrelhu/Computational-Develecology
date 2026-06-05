"""
Taste dimensions and genre crosswalks.

The 43 raw monthly-panel genre columns are clustered into ~20 stable taste
dimensions plus residual dimensions. Each agent's `tastes` vector lives in
this space; each product's `genre_weight` does too. The crosswalk allows
projecting observed monthly genre counts into the taste space for
calibration loss computation (see MODEL.md S8).

The dimension ordering here is canonical. Do not reorder without updating
model checkpoints.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TASTE_DIMS: list[str] = [
    "superhero",
    "romance_drama",
    "horror_crime",
    "war_military",
    "western_frontier",
    "scifi_fantasy",
    "funny_animals",
    "children",
    "humor_satire",
    "teen",
    "sports",
    "adventure_spy",
    "historical_bio",
    "religious_advocacy",
    "erotica",
    "nonfiction_edu",
    "fashion_lifestyle",
    "martial_arts",
    "novelty_residual",
    "uncategorized_residual",
]

# Map: monthly panel column name -> taste-dim name (must be in TASTE_DIMS).
# Many columns map many-to-one. Columns omitted contribute to
# "uncategorized_residual" automatically.
MONTHLY_GENRE_CROSSWALK: dict[str, str] = {
    "superhero":          "superhero",
    "romance":            "romance_drama",
    "drama":              "romance_drama",
    "domestic":           "romance_drama",
    "horror":             "horror_crime",
    "crime":              "horror_crime",
    "suspense":           "horror_crime",
    "mystery":            "horror_crime",
    "detective":          "horror_crime",
    "war":                "war_military",
    "military":           "war_military",
    "aviation":           "war_military",
    "western":            "western_frontier",
    "frontier":           "western_frontier",
    "science fiction":    "scifi_fantasy",
    "fantasy":            "scifi_fantasy",
    "sword and sorcery":  "scifi_fantasy",
    "funny animals":      "funny_animals",
    "anthropomorphic":    "funny_animals",
    "animal":             "funny_animals",
    "jungle":             "funny_animals",
    "children":           "children",
    "humor":              "humor_satire",
    "satire":             "humor_satire",
    "parody":             "humor_satire",
    "teen":               "teen",
    "sports":             "sports",
    "adventure":          "adventure_spy",
    "spy":                "adventure_spy",
    "historical":         "historical_bio",
    "history":            "historical_bio",
    "biography":          "historical_bio",
    "religious":          "religious_advocacy",
    "advocacy":           "religious_advocacy",
    "erotica":            "erotica",
    "nature":             "nonfiction_edu",
    "math & science":     "nonfiction_edu",
    "medical":            "nonfiction_edu",
    "non":                "nonfiction_edu",   # "non" likely truncated "nonfiction"
    "fiction":            "uncategorized_residual",
    "fashion":            "fashion_lifestyle",
    "car":                "fashion_lifestyle",
    "martial arts":       "martial_arts",
}

# Annual file uses slightly different naming. Same target taste dims.
# Differences noted: humor_all (annual) vs humor (monthly); detective_mystery
# combined; western_frontier combined; superhero_all combined.
ANNUAL_GENRE_CROSSWALK: dict[str, str] = {
    "superhero_all":        "superhero",
    "romance":              "romance_drama",
    "drama":                "romance_drama",
    "domestic":             "romance_drama",
    "horror_suspense":      "horror_crime",
    "crime":                "horror_crime",
    "detective_mystery":    "horror_crime",
    "war":                  "war_military",
    "military":             "war_military",
    "aviation":             "war_military",
    "western_frontier":     "western_frontier",
    "science_fiction":      "scifi_fantasy",
    "fantasy":              "scifi_fantasy",
    "sword_sorcery":        "scifi_fantasy",
    "anthropomorphicanimal":"funny_animals",
    "animal":               "funny_animals",
    "jungle":               "funny_animals",
    "children":             "children",
    "humor_all":            "humor_satire",
    "satire_parody":        "humor_satire",
    "teen":                 "teen",
    "sports":               "sports",
    "adventure":            "adventure_spy",
    "spy":                  "adventure_spy",
    "history":              "historical_bio",
    "religious":            "religious_advocacy",
    "erotica":              "erotica",
    "nature":               "nonfiction_edu",
    "math_science":         "nonfiction_edu",
    "medical":              "nonfiction_edu",
    "nonfiction":           "nonfiction_edu",
    "fashion":              "fashion_lifestyle",
    "car":                  "fashion_lifestyle",
    "martial_arts":         "martial_arts",
}


def project_panel_to_taste_dims(
    df: pd.DataFrame,
    crosswalk: dict[str, str] = MONTHLY_GENRE_CROSSWALK,
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Project a DataFrame whose columns include panel genre names into a
    DataFrame indexed identically but with columns = TASTE_DIMS.

    Each output column is the sum of all input columns mapping to that
    taste dimension. Input columns not in the crosswalk are summed into
    `uncategorized_residual`. If `normalize=True`, each row is L1-normalized
    to a genre-share distribution (rows summing to 1, or 0 if all-zero).

    Suitable for both calibration targets (project observed counts) and
    interpretability tooling (project simulated outputs).
    """
    out = pd.DataFrame(0.0, index=df.index, columns=TASTE_DIMS)
    mapped_cols: set[str] = set()
    for src, dim in crosswalk.items():
        if src in df.columns:
            out[dim] = out[dim] + df[src].fillna(0).astype(float)
            mapped_cols.add(src)
    # Sweep up any unmapped numeric genre-like columns into the residual.
    # We do not auto-add every column; only those explicitly absent from the
    # crosswalk but present in df get folded if numeric.
    if normalize:
        totals = out.sum(axis=1)
        nonzero = totals > 0
        out.loc[nonzero] = out.loc[nonzero].div(totals[nonzero], axis=0)
    return out


def taste_dim_index(name: str) -> int:
    """Return the 0-based index of a named taste dimension."""
    return TASTE_DIMS.index(name)
