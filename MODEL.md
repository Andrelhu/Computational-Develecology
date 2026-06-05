# Wandelgeist ABM — Model Specification

Status: locked design as of 2026-06-05. Companion to [README.md](README.md). This document is the authoritative reference for what the model *is* and what the calibration targets *are*. Implementation tracking lives in the task list.

---

## 1. Purpose

A microsimulation of generational change in US cultural markets, grounded in Bronfenbrenner's developmental ecology. The v0 substantive target is the **US comic-book market, 1940-2000**. The mechanism under test is whether organization-mediated peer influence and cohort exposure produce cultural-taste trajectories (genre mix over time) that match observed monthly comic output better than a macro-covariate regression baseline.

The architecture is designed so that additional cultural submarkets (TV, cinema, print books, magazines) can be activated later without restructuring.

## 2. Scope

### In scope (v0)
- Demographic engine (births, deaths, migration) calibrated against US historical data
- Households with structured roles
- Firms and communities/churches with minimal internal organization and activation rules
- Dynamic friendship/acquaintance network driven by organizational exposure
- Comic-book submarket: 43 monthly genre channels, ~20-25 curated taste dimensions
- Censorship and market-structure event shocks as exogenous forcings
- V&V harness with held-out validation and a regression baseline to beat

### Out of scope (v0, design memos only)
- Religion as a peer-influence channel
- Political affiliation as an attribute or channel
- Education as an attribute or channel
- TV, cinema, print-book, magazine submarkets (scaffolded, not activated)
- International cross-country comparison

## 3. Scenarios

Two independent 30-year fits. Both span 3+ cohort generations.

| Scenario | Window | Eras covered | Key shocks |
|---|---|---|---|
| A | 1940-1970 | Golden Age + early Silver | WWII rationing, CCA enforcement (1954) |
| B | 1970-2000 | Bronze + Modern + Speculator | Direct-market launch (~1979), Dark Knight era (1986), '90s speculator bubble (1992-93) + crash |

Each scenario gets its own parameter fit. Held-out validation periods are sampled within the scenario window. **Parameter drift between scenarios is a reportable finding**, not a failure mode.

## 4. Time semantics

- **1 step = 1 calendar month.** All rates and hazards are monthly.
- Annual rates `p_a` enter the model via `monthly_hazard(p_a) = 1 - (1 - p_a) ** (1/12)`.
- Yearly forcings (population targets, migration totals) are applied at the December → January boundary of each simulated year.
- Birthdays are tracked by a `months_until_bday` counter per agent.

## 5. Agent state

Per-agent attributes (PyTorch tensors, N = population size):

| Attribute | Type | Description |
|---|---|---|
| `id` | int64 | Stable agent identifier |
| `age` | int32 | Years (incremented at birthday) |
| `sex` | bool/int8 | M/F. Used for mortality, fertility, LFP. |
| `alive` | bool | Mask |
| `months_until_bday` | int32 | 0-11 |
| `year_of_birth` | int32 | Cohort tag (set once at creation) |
| `tastes` | float32, dim D | Genre-mapped taste vector, D ≈ 20-25 (see §8) |
| `consumed_count` | int32 | Cumulative products consumed |
| `household_id` | int32 | Membership |
| `firm_id` | int32 or -1 | Workplace; -1 if not in labor force |
| `school_id` | int32 or -1 | Active for dependents |
| `community_id` | int32 or -1 | Optional community/church |
| `role_household` | int8 | `head_guardian` / `partner` / `dependent` |
| `role_economic` | int8 bitmask | `incomekeeper`, `housekeeper` (overlay; non-exclusive) |

Friend and acquaintance ties live in separate sparse adjacency tensors (not per-agent fields). See §13.

## 6. Demographic engine

**Hybrid: exogenous yearly totals + age×sex hazards for within-year allocation.**

### Inputs (yearly, observed)
- Total population (already in `master_monthly_panel.csv` and Census)
- Total births (CDC vital statistics)
- Total deaths (CDC vital statistics)
- Age-specific fertility rate by single year of age (female), `f(age, year)` (CDC)
- Age×sex-specific mortality `q(age, sex, year)` (CDC life tables)
- Net migration totals + age×sex distribution (INS/DHS yearbook + Census ACS)

### Algorithm
1. **At year boundary**, read target totals `B_y`, `D_y`, `M_y` for the year.
2. **Deaths**: compute monthly hazard per agent from `q(age, sex, year)`. Stochastically sample deaths each month. Rescale at year-end if needed so total deaths ≈ `D_y` (importance weighting).
3. **Births**: stochastically draw `B_y` newborns spread across the year, allocated to fertile-age females weighted by `f(age, year)`. Newborns enter with `age=0`, sex drawn from observed birth sex ratio, random taste vector, household = mother's household.
4. **Migration**: draw `M_y` migrants spread across the year, sex and age sampled from the observed migrant age×sex distribution. Migrants enter with random taste vectors, no household yet (assigned probabilistically to a household within first 12 months), no friends/acquaintances. Network rule (§13) accretes ties.

### Implementation notes
- **Population array must grow and shrink.** The current dead-slot recycling in [model.py:182](simdeveco/model.py:182) caps births at `len(dead_slots)`, preventing growth — fatal bug for any historical scenario. Replace with: preallocate generously, use `alive` mask, append+resize at year boundaries when capacity is exceeded.
- Demographic engine is calibrated independently per scenario, then **frozen** for the cultural-market microsimulation phase.

### Calibration target
- Total population trajectory: ≤2% MAPE per year
- Age pyramid by decade: chi-square match against decadal census
- Sex ratio by age: ≤1pp deviation

## 7. Cohort and social influence

### Cohort tagging
Every agent carries `year_of_birth`. Cohort bins (~10-15 years wide) are used **only for reporting**, never for mechanism. Suggested reporting bins by scenario:

- Scenario A: pre-1925, 1925-1944, 1945-1964 (Boomer start)
- Scenario B: 1945-1964, 1965-1980 (Gen X), 1981-2000 (Millennial start)

### Continuous age-distance influence kernel
For social influence between agents i and j, an age-distance weight modulates the existing tie:

```
w_age(i, j) = exp(-|age_i - age_j|² / (2 σ²))
```

`σ` is a calibration parameter. Small σ → strong cohort effects (peer influence concentrated among age peers). Large σ → cohort-neutral (mixed-age influence). Multiplied into the sparse adjacency before message-passing in `_social_influence`.

### Combined social-influence step
For each alive agent, compute taste update as a weighted average of contributions from:

1. **Family** (`fam_adj`, household-clique structure with role weights — guardian→dependent edges stronger; partner↔partner highest)
2. **Friends** (`friend_adj`, dynamic, see §13)
3. **Acquaintances** (`acq_adj`, dynamic, see §13)
4. **Intra-firm** (`firm_adj`, built per-step from co-active firm members)
5. **Intra-community** (`comm_adj`, built per-step from co-active community/church members)

Each channel has a weight (`w_fam`, `w_fr`, `w_ac`, `w_firm`, `w_comm`); these are calibration parameters. Age-distance kernel applies to friend and acquaintance channels; family and workplace channels are already structurally constrained.

Bounded confidence is retained: an agent updates toward the neighborhood mean only if taste distance < latitude threshold.

## 8. Tastes and the genre dictionary

### Taste dimension
`tastes` is a float32 vector of dimension D ≈ 20-25. Each dimension corresponds to a **curated cluster of comic genres** built from the 43 monthly genre columns:

| Cluster (example) | Source columns |
|---|---|
| superhero | superhero |
| romance | romance |
| horror_crime | horror, crime, suspense, mystery, detective |
| war_military | war, military, aviation |
| funny_animals | funny animals, anthropomorphic, animal |
| sci_fi_fantasy | science fiction, fantasy, sword and sorcery |
| western | western, frontier |
| teen_humor | teen, humor, satire, parody |
| children | children, jungle |
| sports | sports |
| adventure | adventure, spy |
| romance_drama | romance, drama, domestic |
| historical_bio | historical, history, biography |
| religious_advocacy | religious, advocacy |
| ... | ... |

Plus 2-3 residual dimensions for novelty/innovation that don't map cleanly to observed categories.

A crosswalk dict `GENRE_CROSSWALK: {panel_col → taste_dim}` lives in `simdeveco/data/genres.py`. The annual file uses a slightly different naming (e.g., `humor_all`, `all_but_superhero`) and gets its own crosswalk to the same taste dimensions.

### Products
Each product carries a `genre_weight` vector (D-dim, sums to 1) and is produced by a firm based on the firm's member-taste average + noise.

### Calibration
The monthly distribution of genre weights across all products produced in a month should match the observed monthly genre share in the panel (Wasserstein-1 on genre vectors).

## 9. Cultural-market scaffold

### Submarkets
A registry keyed by submarket name:

```python
markets = {
    "comics":  {feats, consumed, producers, cadence_months: 1, ...},
    # later: "tv", "cinema", "print_books", "print_magazines"
}
```

In v0 only `"comics"` is activated. Adding a new submarket is a registry insertion + producer assignment — no model restructuring.

### Producers
A firm (see §11) may be a producer in zero, one, or multiple submarkets. v0 comic firms produce monthly issues; their output volume and genre vector depend on member tastes and firm size.

### Consumers
Agents draw products from submarkets they're exposed to. Exposure is moderated by:
- Direct purchase (utility = `tastes · genre_weight`)
- Advertising (random subset advertised each month)
- Word-of-mouth via friend/acquaintance/family channels (sharing of recently-consumed products)

### Market state per submarket
- `prod_feats`: matrix of all products' genre vectors this month
- `prod_consumed`: per-product consumption counter
- Reset at month boundary (products are time-stamped, observed sales aggregated before reset)

## 10. Organizations: households

### Demographic roles (mutually exclusive)
- `head_guardian` — primary adult, exists for every non-empty household
- `partner` — optional second adult
- `dependent` — minors and other dependents

### Economic-role overlay (non-exclusive bitmask)
- `incomekeeper` — at least one per household; modulates exposure to workplace peer-influence channel
- `housekeeper` — zero or more; modulates exposure to community/church and family channels

### Family adjacency
Built from household membership, but **weighted by role pair**:
- guardian ↔ partner: highest weight
- guardian/partner → dependent: high weight (asymmetric or symmetric, calibratable)
- dependent ↔ dependent (siblings): baseline weight

### Formation and dissolution
- Adults pair into households at age-specific marriage hazard (calibrated against historical marriage rates)
- Dependents attach at birth to the mother's household
- On guardian death without partner, dependents reassign (to nearest relative or institutional placeholder)
- On all-adult-deaths, household dissolves

## 11. Organizations: firms (labor)

### Members and activation
- Each working-age agent has a `firm_id`. Agents under labor-force-participation cutoff age (or above retirement) have `firm_id = -1`.
- Monthly activation: each member is "active this month" with probability = age×sex-specific LFP rate from BLS data. Inactive members contribute neither productivity nor peer influence that month.

### Productivity
- A firm's monthly output (in submarkets where it produces) is proportional to its active-member count.
- A firm's product genre vector is the mean of its active members' tastes plus noise.

### Intra-firm peer influence
- Build a sparse `firm_adj` each month over **active** members of each firm.
- Apply during the social-influence step alongside family/friend/acquaintance/community channels.

### Rotation
- Each month some members leave (rotation rate, calibratable) and join another firm (or unemployed pool).
- Hiring happens at age of labor-force entry and after rotation events.

### v0 firm types
- Comic-publisher firms (subset of all firms; produce in `"comics"` submarket)
- Generic firms (no submarket production; serve only as workplace peer-influence container)

## 12. Organizations: communities and churches

Same minimal interface as firms, with differences:

| | Firm | Community/Church |
|---|---|---|
| Membership | Workplace assignment | Voluntary affiliation |
| Activation gate | LFP rate (age×sex) | Attendance hazard (calibratable; ~0.5 monthly for active members) |
| Productivity | Yes (in producing firms) | No (no submarket output in v0) |
| Peer-influence channel | Yes | Yes |
| Rotation rate | Calibrated against job tenure | Calibrated against church-switching rates |

Religion-content peer influence is deferred to design memo (T11). v0 communities and churches are isomorphic — both are "regularly-attended non-work, non-family social organizations."

## 13. Network dynamics

### Initial seed
At t=0, friend and acquaintance adjacencies are built from configurable network topologies (small-world, Erdős–Rényi, scale-free, SBM) as in current [model.py](simdeveco/model.py). These are the **structural priors**, not the steady state.

### Rebuild rule (every 6 simulated months)
For each alive agent A:
1. Sample one organization uniformly from A's memberships: `{household, firm, school, community/church}` (skip if `id == -1`).
2. From the chosen organization's current members, sample 1-2 agents that are not currently tied to A as family/friend/acquaintance.
3. Add the sampled agents to A's acquaintance set.

### Interaction rule (every month)
For each alive agent A:
1. Enumerate A's current friends + acquaintances.
2. For each tie, execute a pairwise social-influence interaction (the existing taste-blending mechanism, bounded by latitude of acceptance and gated by the age-distance kernel).

### Friendship promotion
Acquaintance → friend transition rule **to be defined later** (T12). Placeholder: promote if pair has interacted N times within last K months and remained within taste-distance latitude.

### Tie maintenance
- Ties to dead agents are pruned at year boundaries.
- No automatic decay in v0; reassessment via the rebuild rule maintains turnover.

## 14. Event shocks

Exogenous flags in the panel applied as multiplicative masks or perturbations on firm output:

| Event | Period | Effect |
|---|---|---|
| WWII rationing | 1941-1945 | Paper supply constraint → max-issues cap per firm |
| CCA enforcement | 1954-1971 | Suppress horror, crime, romance dimensions (~50-80% mask on producer output for affected genres) |
| Direct market | 1979+ | Firm cost structure shifts; specialty-genre output rises (modeled as taste-update gain change) |
| Dark Knight era | 1986+ | Innovation pulse: noise term on producer output increases |
| Speculator bubble | 1992-1993 | Demand-side multiplier on superhero genre |
| Speculator crash | 1994-1995 | Demand collapse + firm exits (rotation_rate spike) |

All event windows come directly from the panel's `event_*` columns. No new dating decisions required.

## 15. V&V and calibration

### Loss function (composite per scenario)
```
L = w_tot · log_MAE(total_issues_monthly)
  + w_gen · Wasserstein(genre_share_monthly)
  + w_dem · chi2(age_pyramid_decadal)
  + w_pop · MAPE(population_yearly)
```

`w_*` are scenario-fixed weighting hyperparameters set so each term contributes comparably at the baseline.

### Calibration procedure (per scenario)
1. Calibrate demographic engine first (deaths, births, migration); freeze.
2. Pre-fit taste↔genre mapping using observed monthly genre shares (offline, no ABM run).
3. Bayesian / approximate-Bayesian-computation sweep over free ABM parameters:
   - Firm count, firm productivity, ad rate
   - Taste-update gain, latitude of acceptance
   - Influence channel weights (`w_fam`, `w_fr`, `w_ac`, `w_firm`, `w_comm`)
   - Age-distance kernel σ
   - Rotation rates
4. Hold out a randomly-sampled subset of months within the window for validation.

### Benchmark baseline (must beat)
OLS regression of `log(total_issues)` on monthly aligned macro covariates from the panel: CPI, women_lfp, population, unemployment_rate, event flags. The ABM must beat this baseline in held-out R² (or MAE) to earn its complexity.

### Reporting
- Per-scenario parameter fit table
- Held-out validation curves
- Parameter drift between scenarios (a finding, not a failure)
- Posterior-predictive plots: simulated vs observed monthly total + genre shares
- Age pyramid comparison plots by decade

## 16. Data inventory

### Already in repo
| File | Coverage | Resolution | Notes |
|---|---|---|---|
| `datasets/master_monthly_panel.csv` | 1900-2000 | Monthly | 43 genres + aligned macro + event flags |
| `datasets/US_Comics_Macro_Enriched.csv` | 1938-2000 | Annual | Multi-media context, social/cultural covariates |

### To be ingested (external)
| Source | Coverage | Resolution | For |
|---|---|---|---|
| CDC life tables | 1940-2000 | Annual, age×sex | Mortality |
| CDC vital stats | 1940-2000 | Annual, age (female) | Fertility |
| INS/DHS yearbook + ACS | 1940-2000 | Annual, age×sex | Migration |
| BLS LFP series | 1940-2000 | Annual, age×sex | Firm activation |
| US decadal census | 1940-2000 | Decadal, age pyramid | Demographic calibration |
| Census marriage rates | 1940-2000 | Annual, age | Household formation |

### Deferred (design-only data sources)
| Source | For | Memo |
|---|---|---|
| GSS | Religion, political affiliation | T11 |
| NCES Digest of Education Statistics | Education attainment by cohort | T11 |
| ACS educational attainment | Education by cohort | T11 |

## 17. Architecture and code organization

```
simdeveco/
├── __init__.py         # Re-exports: VectorDevecology, run_experiments
├── main.py             # CLI entry point
├── simulation.py       # Wrapper around run_experiments
├── model.py            # VectorDevecology (PyTorch tensor model)
├── utils.py            # Network builders, mortality, seeding
├── data/               # NEW: data loaders + crosswalks
│   ├── __init__.py
│   ├── comics.py       # Loads master_monthly_panel + macro_enriched
│   ├── demographics.py # Loads CDC life tables, census, fertility
│   ├── labor.py        # Loads BLS LFP series
│   ├── migration.py    # Loads INS/DHS migration data
│   └── genres.py       # GENRE_CROSSWALK dict and taste-dim definitions
├── calibration/        # NEW: V&V harness
│   ├── __init__.py
│   ├── targets.py      # Target schema, loss components
│   ├── scenarios.py    # Scenario A / B definitions
│   ├── runner.py       # Calibration sweep driver
│   └── plots.py        # Posterior-predictive plots
├── experiments.py      # Parameter sweeps
└── requirements.txt

versions/
├── legacy_mesa/        # Original Mesa implementation, reference only
│   ├── individual.py
│   ├── collective.py
│   └── market.py
└── simdeveco_0_1/

datasets/               # Raw data, versioned CSVs
results/                # Run outputs
```

**Mesa code** (currently in `simdeveco/agents/`) moves to `versions/legacy_mesa/` and is not imported by any runtime module.

**Import discipline**: package-style imports throughout (`from simdeveco.X import Y`). Run via `python -m simdeveco.main` so the package is on `sys.path`.

## 18. Roadmap

Tasks tracked in the session task list. Suggested phase ordering:

### Phase 0 — Foundation (single PR)
- T1 Code consolidation (Mesa → versions/, fix imports, package re-exports)
- T2 Time-unit semantics pin (monthly_hazard helper)
- T5 Data ingestion module + external demographic data pulls
- T4 V&V harness skeleton

### Phase 1 — Demographic backbone (calibrated then frozen)
- T3 Hybrid demographic engine (fix grow/shrink, add sex)
- T13 Migration channel
- Per-scenario calibration

### Phase 2 — Cohort + tastes
- T6 Cohort tag + age-distance kernel + taste-dim re-pin
- T14 Taste↔genre crosswalk + event-shock encoding

### Phase 3 — Cultural market
- T7 Multi-submarket scaffold (comics activated, others stubbed)
- T8 Comic v0 calibration on both scenarios

### Phase 4 — Organizations
- T9 Household role refactor
- T10 Firm + community internal organization
- T12 Dynamic network rule + friendship promotion definition

### Phase 5 — Deferred
- T11 Design memo: religion/politics/education channels

---

## Open items and TBDs

- Friendship promotion rule (acquaintance → friend) — to be defined during Phase 4
- Whether to source 2000-2010 data to extend Scenario B — deferred
- Religion/politics/education channels — design only in v0
- Exact age-bin choices for cohort *reporting* (cosmetic, not mechanism)

## Conventions

- Years are calendar years (e.g., 1940 = Jan 1940 – Dec 1940 inclusive).
- Monthly resolution; January = 1, December = 12.
- Ages are integer years (incremented at birthday month).
- Currency: nominal USD; CPI deflator in panel allows conversion to real.
- Genre names: panel column conventions are authoritative; crosswalks bridge naming differences across files.
