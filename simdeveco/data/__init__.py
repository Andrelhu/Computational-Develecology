"""
simdeveco.data

Data loaders and crosswalks. v0 covers:
- comics:       master_monthly_panel + US_Comics_Macro_Enriched (in-repo)
- genres:       crosswalk from panel genre columns to taste dimensions
- demographics: CDC life tables, decadal census (external; CSV path expected)
- labor:        BLS LFP by age x sex (external; CSV path expected)
- migration:    INS/DHS net flow by age x sex (external; CSV path expected)

See MODEL.md S16 for the canonical data inventory.
"""

from .comics import load_monthly_panel, load_macro_enriched
from .genres import (
    TASTE_DIMS,
    MONTHLY_GENRE_CROSSWALK,
    ANNUAL_GENRE_CROSSWALK,
    project_panel_to_taste_dims,
)

__all__ = [
    "load_monthly_panel",
    "load_macro_enriched",
    "TASTE_DIMS",
    "MONTHLY_GENRE_CROSSWALK",
    "ANNUAL_GENRE_CROSSWALK",
    "project_panel_to_taste_dims",
]
