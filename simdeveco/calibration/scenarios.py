"""
Scenario definitions. See MODEL.md S3.

Two independent 30-year fits:
- A: 1940-1970 (Golden + early Silver age)
- B: 1970-2000 (Bronze + Modern + Speculator era)

Each scenario specifies:
- Date bounds (inclusive start, inclusive end)
- A reporting cohort partition (for plots; mechanism uses continuous kernel)
- A held-out validation strategy

Held-out validation in v0: random 15% of months drawn with a fixed seed,
stratified by year so each year contributes roughly the same number of
held-out months.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Scenario:
    name: str
    start_year: int            # inclusive
    end_year: int              # inclusive
    cohort_bins: tuple[tuple[int, int], ...]  # ((start, end), ...), inclusive
    validation_fraction: float = 0.15
    validation_seed: int = 17

    @property
    def n_years(self) -> int:
        return self.end_year - self.start_year + 1

    @property
    def n_months(self) -> int:
        return self.n_years * 12


SCENARIO_A = Scenario(
    name="A_1940_1970",
    start_year=1940,
    end_year=1970,
    cohort_bins=(
        (1880, 1924),  # pre-Boomer
        (1925, 1944),  # Greatest / Silent overlap
        (1945, 1970),  # early Boomer
    ),
)

SCENARIO_B = Scenario(
    name="B_1970_2000",
    start_year=1970,
    end_year=2000,
    cohort_bins=(
        (1945, 1964),  # Boomer
        (1965, 1980),  # Gen X
        (1981, 2000),  # early Millennial
    ),
)

_REGISTRY = {SCENARIO_A.name: SCENARIO_A, SCENARIO_B.name: SCENARIO_B}


def scenarios_by_name(name: str) -> Scenario:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown scenario {name!r}. Available: {list(_REGISTRY)}"
        )
    return _REGISTRY[name]
