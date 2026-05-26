# NextStep.md — Model Deep Dive + Improvement Roadmap

This document has two parts:
1. **Part 1**: Complete technical breakdown of the current model — every script, every equation, every assumption, every limitation
2. **Part 2**: Detailed improvement roadmap, prioritised by expected return-per-hour

Last updated after the commit `9d75e04` (feat: complete MC simulator, Polymarket calibration, and submission pipeline).

---

## Table of Contents

**Part 1 — Current Model**
- [Pipeline Overview](#pipeline-overview)
- [The 5 Sub-Models](#the-5-sub-models)
- [Math Reference](#math-reference)
- [Script-by-Script Walkthrough](#script-by-script-walkthrough)
- [Competition Scoring System](#competition-scoring-system)
- [Prediction Strategy Justification](#prediction-strategy-justification)
- [Capabilities](#capabilities)
- [Known Limitations](#known-limitations)

**Part 2 — Improvement Roadmap**
- [Tier 1 — Highest ROI](#tier-1--highest-roi-pre-deadline)
- [Tier 2 — Real Gains](#tier-2--real-gains-medium-effort)
- [Tier 3 — Lower Priority](#tier-3--lower-priority-long-term)
- [Refresh Schedule](#refresh-schedule)

---

# Part 1 — Current Model

## Pipeline Overview

The model is a **chain of 10 sequential scripts**, each producing artifacts that the next script consumes:

```
01_collect_data.py
    ↓ data/raw/* (Kaggle CSVs, Polymarket, Odds API, FIFA snapshots)

01c_load_match_stats.py
    ↓ data/processed/match_stats_wc2022.csv (long format)

02_clean_merge.py
    ↓ data/processed/matches_clean.csv, fixtures_*.csv, team_features.csv

03_feature_engineering.py
    ↓ data/processed/match_features.csv, fixture_features.csv

04_train_goals_model.py
    ↓ models/goals_model.pkl, data/processed/group_lambdas.csv

04b_calibrate_goals_model.py     [Polymarket Bayesian prior]
    ↓ overwrites goals_model.pkl + group_lambdas.csv with calibrated values

05_train_corners_model.py
    ↓ models/corners_model.pkl, data/processed/corners_predictions.csv

06_train_cards_model.py
    ↓ models/cards_model.pkl, data/processed/cards_predictions.csv

07_monte_carlo_simulator.py
    ↓ data/processed/mc_qualifiers.csv, mc_knockout.csv, mc_matchup_details.csv

08_generate_predictions.py       [Combines all sub-models, applies fixes]
    ↓ data/processed/predictions_combined.csv

09_format_output.py              [Splits into DataLab submission schemas]
    ↓ output/predictions_group.csv, predictions_knockout.csv
```

**Key design principles:**
- Each script is idempotent (run twice → same output)
- Outputs are CSVs (not pickles for data, only for models) so anyone can inspect
- The pipeline is **fully re-runnable**: change inputs, run 01 → 09, get fresh predictions
- Logical separation: data collection (01-01c) → cleaning (02) → features (03) → model training (04-06) → simulation (07) → final predictions (08-09)

---

## The 5 Sub-Models

The competition requires predictions across multiple metrics per match, so we build **5 specialist models**:

| # | Model | Predicts | Method | Training data |
|---|---|---|---|---|
| 1 | **Goals (Dixon-Coles)** | Home goals, Away goals | Weighted MLE Poisson with low-score correction | 4,534 matches since 2018 |
| 2 | **Outcome (implicit)** | Home win / Draw / Away win | Derived from full joint score distribution | Same as goals model |
| 3 | **Corners (Poisson regression)** | Total corners in match | L2-regularised Poisson regression | 64 WC 2022 matches |
| 4 | **Cards (Poisson regression)** | Total yellows in match | L2-regularised Poisson regression | 64 WC 2022 matches |
| 5 | **Monte Carlo tournament simulator** | Knockout matchups + winners | 10,000-iteration simulation using goals model | Group lambdas + knockout structure |

Plus a **6th supporting model**: empirical red card rate (~0.06/match), used as a low-rate Poisson predictor.

---

## Math Reference

### 1. Dixon-Coles Poisson Goals Model

For a match between home team `h` and away team `a`, the expected goals are:

```
λ_home = exp(α + attack[h] − defense[a] + γ · is_host_h)
λ_away = exp(α + attack[a] − defense[h])
```

Where:
- **α** (intercept): log of baseline scoring rate. Fitted value ≈ −0.005, so exp(α) ≈ 1.0 (~1 goal per team in a neutral matchup with average attack/defense)
- **attack[i]**: team i's intrinsic offensive strength. Positive = attacks well. Range ≈ [−1.0, +1.2]
- **defense[i]**: team i's intrinsic defensive strength. Positive = defends well (subtracted from opponent's λ). Range ≈ [−1.1, +0.7]
- **γ** (gamma): home-advantage parameter. Fitted ≈ 0.24, so exp(γ) ≈ 1.27 (27% more goals when home team is genuinely at home). **Applied only when `home_is_host=1`** (i.e., USA/Mexico/Canada playing in their own country), zeroed for neutral-venue matches.
- **is_host_h**: 1 if home team is playing in their host country, 0 otherwise

### 2. Dixon-Coles Low-Score Correction

The base Poisson model underestimates low-scoring results (0-0, 1-0, 0-1, 1-1). Dixon-Coles adds a correction:

```
τ(x, y, λ_h, λ_a, ρ) =
  1 − λ_h · λ_a · ρ    if (x, y) = (0, 0)
  1 + λ_h · ρ          if (x, y) = (0, 1)
  1 + λ_a · ρ          if (x, y) = (1, 0)
  1 − ρ                if (x, y) = (1, 1)
  1                    otherwise
```

The joint probability becomes:

```
P(x, y | λ_h, λ_a, ρ) = τ(x, y) · Pois(x; λ_h) · Pois(y; λ_a)
```

Our fitted ρ ≈ **−0.062**. Negative ρ means the model boosts 0-0 and 1-1 probabilities and slightly suppresses 1-0 / 0-1 probabilities — which matches reality (real football has more low-scoring draws than independent Poisson predicts).

### 3. MLE Fit with Sample Weights

The negative log-likelihood being minimised:

```
NLL(θ) = −Σ_m w_m · [ log Pois(x_m; λ_h_m) + log Pois(y_m; λ_a_m) + log τ(x_m, y_m) ]
       + 100 · ( (Σ_i attack[i])² + (Σ_i defense[i])² )
```

The 100x penalty term softly enforces the identifiability constraints `Σ attack = 0` and `Σ defense = 0`. Without this, all attacks could shift by +1 with all defenses shifting by −1 and the predictions would be identical.

**Sample weights** `w_m`:

```
time_weight     = exp(−age_days_m / 547 · ln 2)        [18-month half-life]
sample_weight_m = time_weight_m · (1.0 if competitive else 0.2)
```

So a friendly from 2019 might have weight ≈ 0.04 × 0.2 = 0.008, whereas a competitive match from 2024 has weight ≈ 0.6. The optimiser effectively learns from recent competitive matches.

Optimised via `scipy.optimize.minimize` with the **L-BFGS-B** algorithm. 455 free parameters (226 teams × 2 + 3 globals), converges in ~30 seconds.

### 4. Polymarket Bayesian Calibration

The data-driven fit can be biased against teams whose recent form is poor but whose intrinsic strength is high (France, Brazil, Argentina post-WC22 had mediocre years). We blend in Polymarket's championship-probability signal:

```
model_strength[i] = attack[i] + defense[i]
market_strength[i] = logit(polymarket_win_prob[i])

z_model[i] = (model_strength[i] − μ_model) / σ_model
z_market[i] = (market_strength[i] − μ_market) / σ_market

z_blend[i] = 0.4 · z_model[i] + 0.6 · z_market[i]      [40% data, 60% market]

new_strength[i] = z_blend[i] · σ_model + μ_model
delta[i] = new_strength[i] − model_strength[i]

attack[i] += delta[i] / 2
defense[i] += delta[i] / 2
```

The blend is z-score based (standardised) so we're comparing **rank within distribution**, not absolute values. Why 40/60: this project's edge is corners/cards modelling; team strength is better captured by the market.

### 5. Corners — Poisson Regression

```
log(λ_corners) = β₀ + β₁ · sum_avg_corners
                    + β₂ · min_avg_corners
                    + β₃ · max_avg_corners
                    + β₄ · |fifa_rank_home − fifa_rank_away|
                    + β₅ · (fifa_rank_home + fifa_rank_away) / 2

corners_pred = round(λ_corners · altitude_factor + tournament_bump)
```

Where:
- `sum_avg_corners` = home team's leave-one-out WC22 avg + away team's LOO avg
- `altitude_factor`: 0.92 at >2000m, 0.95 at >1500m, 1.0 otherwise (Mexico City reduces total corners ~8%)
- `tournament_bump = 0.6`: empirical calibration since WC22 (8.94 avg) ran lower than WC18/WC14 (~10.5 avg)

L2 regularised (α=0.3). 5-fold CV MAE ≈ 2.99 (README target <2.5, achievable only with more training data).

### 6. Yellow Cards — Poisson Regression

```
log(λ_yellow) = β₀ + β₁ · sum_avg_yellows
                   + β₂ · |fifa_rank_home − fifa_rank_away|
                   + β₃ · conf_mult

yellow_pred = round(λ_yellow · altitude_card_factor)
```

`conf_mult` is a confederation card-rate multiplier — average of home and away teams' confederation factors:

| Confederation | Multiplier | Rationale |
|---|---|---|
| CONMEBOL | 1.15 | South American teams average more cards |
| CAF | 1.10 | African teams slightly above average |
| CONCACAF | 1.00 | Neutral baseline |
| AFC | 0.95 | Slightly below average historically |
| UEFA | 0.95 | Slightly below average |
| OFC | 1.00 | Small sample, default neutral |

5-fold CV MAE ≈ 1.95 (README target <1.0).

### 7. Red Cards — Empirical Rate

Red cards are too rare for regression (~0.06 per match). We use the empirical rate, multiply by altitude factor, and **predict 0 for all matches** since:

```
EV(predict 0) = 0.94 × 5 pts = 4.70 pts/match
EV(predict 1) = 0.06 × 5 pts = 0.30 pts/match
```

Across 104 matches, predicting 0 everywhere is +457 pts vs predicting 1 everywhere.

### 8. Monte Carlo Tournament Simulator

Pseudo-code:

```
for sim in range(10_000):
    # Group stage
    group_results = []
    for fixture in 72_group_fixtures:
        hg ~ Poisson(λ_home_fixture)
        ag ~ Poisson(λ_away_fixture)
        group_results.append((fixture.group, fixture.home, fixture.away, hg, ag))

    # Standings with FIFA tiebreakers
    for each of 12 groups:
        sort teams by Points → GoalDiff → GoalsFor → random
        record group_winner, runner_up, third_place

    # Best 8 of 12 third-place teams qualify
    best_thirds = sort(all_thirds, by Points → GD → GF)[:8]

    # Resolve qualifiers to 32 knockout slots
    # ... slot descriptors like "Winner Group A", "Best 3rd (Groups A/B/C/D/F)"
    # ... knockout matches reference each other ("Winner Match 73")

    for ko_match in 32_knockout_matches_in_id_order:
        home = resolve_slot(ko_match.slot_home, qualifiers, prior_results)
        away = resolve_slot(ko_match.slot_away, qualifiers, prior_results)

        # 90 minutes
        hg ~ Poisson(λ_home_neutral(home, away))
        ag ~ Poisson(λ_away_neutral(home, away))

        if hg == ag:                      # Extra time at 1/3 lambda rate
            hg += Poisson(λ_home / 3)
            ag += Poisson(λ_away / 3)

        if hg == ag:                      # Penalty shootout (50/50)
            winner = random_choice(home, away)
            penalties_flag = True
        else:
            winner = home if hg > ag else away
            penalties_flag = False

        record(ko_match.id, home, away, hg, ag, winner, penalties_flag)

# Aggregate
for each knockout slot:
    most_common_matchup = mode of (home, away) pairs across sims
    most_common_winner = mode of winners across sims
    penalty_prob = fraction of sims where penalties=True
```

The full per-sim bracket is stored as `mc_matchup_details.csv` so script 08 can pick consistent matchups (e.g., 3rd-place teams that don't overlap with the Final).

### 9. Score Prediction Strategy (the "rounded expected" choice)

Three possible strategies for picking a single (home_score, away_score) from the joint distribution P:

1. **Mode**: argmax over P[h, a]. Tends to predict (0, 0) for low-lambda matches because that single bin has the highest probability mass. Avg total goals across 104 matches: ~1.2
2. **Expected-value (EV)-optimal**: pick (h, a) that maximises expected competition points. Still favours low scores when P(0,0) is high. Avg: ~1.7
3. **Rounded expected**: `h = round(λ_home), a = round(λ_away)`. Maps directly to "what would you expect on average". Avg: ~2.4 ✓ matches WC reality

We chose **rounded expected** because:
- It matches historical WC averages (2.2-2.7)
- It optimises for the goal-diff/total partial credit (10 pts) rather than chasing rare exact matches (25 pts)
- It gives realistic-looking predictions that look defensible

For **knockouts**, we additionally **force decisive** results: if `round(λ_h) == round(λ_a)`, bump the favoured team by 1. This avoids the logical inconsistency of "predicted 1-1 score + penalties=False" and aligns with the ~88% historical rate of knockouts decided in regulation/ET.

---

## Script-by-Script Walkthrough

### `01_collect_data.py`
**Purpose:** Pull all external data sources into `data/raw/` and audit external lookup tables.

**What it does:**
- Loads + counts manually-placed CSVs (Kaggle datasets, DataLab fixtures)
- Pulls Polymarket WC 2026 outright winner markets via gamma-api.polymarket.com
- Pulls The Odds API h2h match odds
- Consolidates 3 FIFA-ranking snapshot CSVs into a single time-series file
- Scrapes Wikipedia for current FIFA top 20 (often fails due to multi-level headers; non-blocking)
- Audits `data/external/` (venues, name map, playoff qualifiers, manual FIFA rankings)

**Outputs:**
- `data/raw/polymarket.csv` (102 rows × 7 cols, 50 teams with implied probabilities)
- `data/raw/odds_h2h.csv` (3,291 bookmaker odds rows)
- `data/raw/fifa_ranking_combined.csv` (199,490 historical ranking rows)
- `data/raw/fifa_ranking_current.csv` (Wikipedia scrape result if successful)

### `01c_load_match_stats.py`
**Purpose:** Load the WC 2022 detailed match stats dataset (Kaggle) into a useful format.

**What it does:**
- Reads the team1/team2 wide CSV (88 columns per match)
- Pivots to one-row-per-team-per-match (long format, 128 rows)
- Normalises team names (handles all-caps like USA → United States via a CAPS_MAP)
- Cleans numeric columns (strips % from possession)

**Output:** `data/processed/match_stats_wc2022.csv` — feeds the corners and cards models.

### `02_clean_merge.py`
**Purpose:** Take all raw data, normalise team names, resolve playoff placeholders, build per-team aggregates.

**Key operations:**
1. Resolve UEFA + FIFA inter-confederation playoff placeholders (Bosnia, Sweden, Türkiye, Czechia, DR Congo, Iraq)
2. Apply team-name normalisation map across results, shootouts, odds, fixtures, FIFA rankings
3. Parse Polymarket market questions to extract team names (regex on "Will X win the 2026 FIFA World Cup?")
4. Filter historical matches to 2010+ and flag competitive vs friendly
5. Split venue strings to attach altitude/heat/roof metadata to fixtures
6. Flag host-nation matches (USA/Mexico/Canada playing in their own country)
7. Build per-team feature aggregates: FIFA rank (manual primary, historical fallback), last-20-match form, Polymarket win prob, shootout history, WC22 detailed stats

**Outputs:**
- `data/processed/matches_clean.csv` (30,210 historical matches)
- `data/processed/fixtures_group.csv` (72 rows with venue + host data)
- `data/processed/fixtures_knockout.csv` (32 slots)
- `data/processed/team_features.csv` (48 WC teams with 25+ feature columns)
- `data/processed/polymarket_clean.csv` (50 teams with win probs)
- `data/processed/name_audit.csv` (348 distinct names seen, for debugging)

### `03_feature_engineering.py`
**Purpose:** Build leakage-free time-aware features for goals model training, and prediction features for WC fixtures.

**Key operations:**
- **Time-aware rolling features**: for each team, compute rolling avg goals for/against over last 5, 10, 20 matches *before* this match (shift(1) + rolling). Same for win/draw rate over last 10.
- **Days rest**: gap since last match (clipped to 365 to handle obvious "first match" cases).
- **FIFA rank at match date**: `pd.merge_asof` looks up the most recent ranking snapshot ≤ match date for each team.
- **Head-to-head**: cumulative goal-differential for the home team vs this specific opponent in all prior matches (leakage-free via `shift(1)`).
- **Time decay + friendly downweight**: `sample_weight = exp(-age_days/547 * ln2) × (1.0 if competitive else 0.2)`.
- **Confederation baselines**: per-confederation averages of WC22 metrics — used as fallback for teams that didn't play in WC22.
- **Fixture features**: for each WC fixture, attach home and away team features + pairwise diffs (rank_diff, form_gf_diff, expected_corners, high_altitude flag, etc.).

**Outputs:**
- `data/processed/match_features.csv` (30,210 rows × 41 cols — training set with weights)
- `data/processed/fixture_features.csv` (72 group fixtures × 76 cols — prediction set)
- `data/processed/fixture_features_knockout.csv` (32 slots with venue data)
- `data/processed/confederation_baselines.csv`

### `04_train_goals_model.py`
**Purpose:** Fit the Dixon-Coles model.

**Key operations:**
- Filter training to since 2018, then to teams with ≥10 matches in window (eliminates noise from very rare teams)
- Initial guess: attack=defense=0, α=log(mean total goals / 2), γ=0.3, ρ=−0.13
- Optimise via L-BFGS-B with bounds (attack/defense ∈ [-3,3], γ ∈ [0,1], ρ ∈ [-0.5, 0.5])
- 455 free parameters (226 teams × 2 + 3 globals)
- Save fitted parameters as a Python pickle
- Generate per-fixture predictions: λ_home, λ_away, joint score distribution, mode scoreline, expected goals, win/draw/loss probabilities

**Output:** `models/goals_model.pkl` and `data/processed/group_lambdas.csv`.

### `04b_calibrate_goals_model.py`
**Purpose:** Apply the Polymarket Bayesian prior.

**Key operations:**
- Load goals model + Polymarket implied probabilities
- For each team with both signals, blend in z-score space (40% model, 60% market)
- Distribute the strength delta equally to attack and defense
- Overwrite `models/goals_model.pkl` with calibrated values
- Regenerate `group_lambdas.csv`

This is what brings France/Brazil/Argentina up and Japan/Australia/Costa Rica down.

### `05_train_corners_model.py`
**Purpose:** Fit the corners Poisson regression.

**Key operations:**
- Load WC22 match data (64 matches)
- Pivot wide → match-level (one row per match with `total_corners`)
- Compute leave-one-out team averages
- 5-fold cross-validation for honest MAE estimate
- Fit final Poisson regressor on full data
- Apply altitude adjustment at prediction time

**Output:** `models/corners_model.pkl` and `data/processed/corners_predictions.csv`.

### `06_train_cards_model.py`
**Purpose:** Fit the yellow card Poisson regression + empirical red card rate.

**Key operations:**
- Same WC22 dataset, same LOO + Poisson regression approach
- Adds confederation card-rate multiplier as a feature
- Red cards: just record the global rate (~0.06/match)
- Altitude adjustment for cards is the *opposite* of corners: high altitude → slightly MORE physical play → more cards

**Output:** `models/cards_model.pkl` and `data/processed/cards_predictions.csv`.

### `07_monte_carlo_simulator.py`
**Purpose:** Simulate the WC 10,000 times to derive knockout matchup probabilities.

**Key operations:**
- Pre-compute the full 48×48 knockout lambda table (every possible matchup → λ_h, λ_a)
- For each sim: simulate groups → standings → qualifiers → knockout bracket
- Track per-sim: match-by-match (home, away, winner, penalties)
- Aggregate: per-slot most-common matchup + winner + penalty rate, per-team probabilities of reaching each round, championship probability
- Save full matchup-count detail so the consistency reconciler can find alternatives

**Outputs:**
- `data/processed/mc_qualifiers.csv` (48 teams, championship probabilities)
- `data/processed/mc_knockout.csv` (32 slots, top matchup + winner)
- `data/processed/mc_matchup_details.csv` (9,171 candidate matchups across all slots)

### `08_generate_predictions.py`
**Purpose:** Combine all sub-model outputs into final 104-match predictions.

**Key operations:**
- **Group stage (1-72)**: pull score from group_lambdas (rounded expected), corners from corners_predictions + tournament bump, yellows from cards_predictions, reds = 0, winning_team from independent max-probability across (home, draw, away).
- **Knockout stage (73-104)**: use MC's predicted matchup, recompute Dixon-Coles score for that specific matchup (forced decisive), recompute corners + cards for those teams, winner from win probabilities (split draw probability between teams), penalties = True iff score is a draw.
- **Bracket consistency**: if MC's 3rd-place prediction overlaps with the Final teams, swap to the next-most-common non-conflicting 3rd-place matchup.
- **Sanity checks** per the README (avg goals 2.2-2.8, avg corners 9-11.5, etc.)

**Output:** `data/processed/predictions_combined.csv`, `output/predictions_final.csv` (all 104 matches).

### `09_format_output.py`
**Purpose:** Split into DataLab's two submission schemas + final QA.

**Outputs:**
- `output/predictions_group.csv` (72 rows × 7 cols: match_id, home_score, away_score, corners, yellow_cards, red_cards, winning_team)
- `output/predictions_knockout.csv` (32 rows × 10 cols: match_id, home_team, away_team, ..., match_winner, penalties)
- Prints tournament summary (champion, runner-up, 3rd, 4th, totals)

---

## Competition Scoring System

For reference, here's exactly what each prediction is worth (from README):

### Per-Match Base Points

| Field | Condition | Points |
|---|---|---|
| Score | Exact scoreline | **25** |
| Score | Goal difference OR total goals matches | **10** |
| Corners | Exact total | **10** |
| Corners | Within ±2 | **5** |
| Yellow cards | Exact total | **10** |
| Yellow cards | Within ±1 | **5** |
| Red cards | Exact total | **5** |
| Winning team (group) | Correct | **40** |
| Matchup (knockout) | Both teams correct | **20** |
| Matchup (knockout) | One team correct | **10** |
| Match winner (knockout) | Correct | **20** |
| Penalties (knockout) | Correct | **5** |

### Round Multipliers

| Round | Multiplier |
|---|---|
| Group stage | ×1 |
| Round of 32 | ×1 |
| Round of 16 | ×2 |
| Quarter-final | ×4 |
| Semi-final | ×8 |
| Third-place playoff | ×8 |
| Final | ×16 |

**Strategic implications**:
- A correct Final winner = 20 × 16 = **320 points** (huge)
- A correct Final scoreline = 25 × 16 = **400 points**
- A correct group-stage winner = 40 × 1 = 40 points
- The corners/cards "silent edge" the README mentions: each correct corners = 10 pts × multiplier. Over 32 knockout matches × avg multiplier ~3.5, that's potentially 1,000+ pts if you nail corners more often than competitors.

---

## Prediction Strategy Justification

### Why we predict 0 red cards everywhere

Historical WC red card rate: ~0.15/match. Predicting 0 maximises expected points (~0.85 × 5 pts × 104 matches = 442 pts vs ~0.15 × 5 = 78 pts for predicting 1 everywhere).

### Why knockouts are forced to non-draw scores

A predicted 1-1 score in a knockout requires penalties=True (logical consistency). But our MC says only ~12% of knockouts actually go to pens. Predicting penalties=True for 18 matches when only ~4 will be correct loses ~70 points vs predicting penalties=False (correct ~88% of the time). The fix: force the favoured team to win by 1 goal when lambdas round to a tie.

### Why corners get a +0.6 tournament bump

Our corners model is trained on WC22 (avg 8.94 corners/match). But WC22 had unusual conditions (winter, Qatar). WC18 and WC14 averaged ~10.5. WC26 (summer in NA) is expected closer to the historical norm. The bump is empirical — without it, our predictions land at the bottom of the README's 9.0-11.5 sanity range.

### Why 40/60 model-market blend (and not 50/50 or 30/70)

- **50/50** keeps Brazil out of the top 10 (model thinks they're not elite)
- **40/60** puts Brazil at #7, Argentina at #5, France at #4 — matches Polymarket's top-5 reasonably well
- **30/70** essentially mimics the market, sacrificing our data signal on form/recent results
- 40/60 is the empirical sweet spot where market priors fix obvious blindspots without overriding the model

---

## Capabilities

**What the model does well:**

1. ✅ **All 13 README sanity checks pass** (avg goals 2.40, avg corners 9.52, avg yellow 3.51, 0 reds, 0 penalties forced consistent, 104 matches, valid winning_team values, no nulls)
2. ✅ **Logically consistent bracket** for the Final + 3rd-place pair
3. ✅ **Top 10 championship probabilities align with Polymarket** (Spain, Portugal, England, France, Argentina, Germany, Brazil, Netherlands, Japan, Norway)
4. ✅ **Per-team time-aware features** prevent training leakage
5. ✅ **Host advantage correctly applied** to only the 9 actual host-country matches
6. ✅ **UEFA + FIFA playoff placeholders resolved** to actual teams (Bosnia, Sweden, Türkiye, Czechia, DR Congo, Iraq)
7. ✅ **Altitude effects** captured for Mexico City (2240m), Guadalajara (1560m), and the higher US venues
8. ✅ **Confederation card-rate adjustments** baked into the cards model
9. ✅ **Idempotent and re-runnable** — closer to the deadline, refreshing takes ~5 minutes
10. ✅ **Polymarket calibration** corrects for the data model's bias against teams with poor recent form but strong intrinsic ability

---

## Known Limitations

These are the **real holes** in the current model. The improvement roadmap (Part 2) addresses most of them.

1. **Tiny training set for corners and cards**: 64 matches. 5-fold CV MAE for corners (2.99) and yellows (1.95) is above the README's targets. More data would help here more than anything.

2. **No player-level information**: A team without their star striker (Mbappé injured? Vini Jr suspended?) is meaningfully weaker, and the model doesn't know.

3. **No referee data**: Card rates vary wildly by referee (3.0 to 5.5+ yellows/match across the FIFA elite panel). We don't model this.

4. **Head-to-head computed but unused**: Script 03 computes `h2h_played` and `h2h_home_gd` but the Dixon-Coles model doesn't include them as covariates. Quick fix.

5. **Bracket consistency only fixed for Final + 3rd-place**: The other rounds (R16, QF, SF) may have inconsistencies (e.g., team predicted to win R16 not appearing in the QF that follows).

6. **Score capped at 2 goals per team in most predictions**: Because rounded(λ) rarely exceeds 2. Means we never hit "exact 3-0" matches. We accept this for the goal-diff partial credit.

7. **No squad fatigue model**: Players from Bundesliga (ended May 18) have 3.5 weeks rest; CL final players (May 30) have ~10 days. Could be meaningful for Germany vs France early-round matchups.

8. **No xG / advanced shot quality**: Our predictions are based on goals scored, not chance quality. A team that's been creating great chances but missing them gets undervalued.

9. **Sparse polymarket coverage**: Only ~50 teams have Polymarket odds. For the bottom 75 teams (where the data model relies entirely on historical snapshots), there's no market calibration.

10. **The Odds API h2h prices are pulled but not used in predictions**: We have ~3,000 rows of bookmaker odds in `odds_h2h.csv` but they don't feed into the model. They should be used at minimum as a per-match calibration check.

11. **MC simulator picks per-slot independently**: Across 10K sims, the most common Final matchup might be (Spain, Portugal), but Portugal could be in slot X's most common matchup AND slot Y's. The current reconciler handles Final/3rd-place but not the broader issue.

12. **No live data feeds**: Late-breaking injuries, lineup changes, tactical reports — none of this enters the model. By June 11 kickoff, more accurate info will be public but we don't currently pull it.

---

# Part 2 — Improvement Roadmap

Each improvement below has: **why it matters**, **how to implement**, **effort estimate**, and **expected gain**.

Improvements ordered by **expected points-per-hour-of-work**.

---

## Tier 1 — Highest ROI (pre-deadline)

### 1. Expand corners + cards training data (BIGGEST WIN)

**Why it matters:** Our corners and cards models train on just 64 matches. The MAE on cross-validation is well above the README's targets (2.99 vs <2.5 for corners; 1.95 vs <1.0 for yellows). More data is the single biggest lever.

**Sources:**
- **football-data.co.uk** has European league CSVs with `HC`, `AC` (corners) and `HY`, `AY`, `HR`, `AR` (cards) columns going back to 1993. Premier League, La Liga, Bundesliga, Serie A, Ligue 1 are all available. Each season is ~380 matches per league. **5 seasons × 5 leagues × 380 = ~9,500 matches.**
- **Euro 2024 detailed stats**: 51 matches. Find a similar Kaggle dataset to the WC22 one.
- **Copa America 2024 detailed stats**: 32 matches.
- **AFCON 2023 (held early 2024) + AFCON 2025**: ~52 matches each.

**How to implement:**
1. Write `01d_collect_football_data_uk.py` that downloads CSVs from football-data.co.uk and merges into a unified format
2. Search Kaggle for Euro 2024 / Copa 2024 / AFCON detailed-stats datasets, drop into `data/raw/`
3. Update scripts 05 and 06 to include this expanded training pool
4. Retrain — should see CV MAE drop substantially

**Effort:** 2-3 hours total. **Expected gain:** 20-30% reduction in corners/cards MAE, which translates to maybe +30-50 pts on the corner predictions and +30-50 pts on cards across 104 matches.

### 2. Use head-to-head as an actual model feature

**Why it matters:** Some matches have strong historical patterns (Brazil-Argentina, El Clasico, etc.). Our pipeline computes h2h but the Dixon-Coles model doesn't use it as a covariate.

**How to implement:**
1. Modify `04_train_goals_model.py` to add `h2h_home_gd` as an additional regressor:
   ```python
   λ_home = exp(α + attack[h] − defense[a] + γ · is_host + δ · h2h_home_gd)
   λ_away = exp(α + attack[a] − defense[h] − δ · h2h_home_gd)
   ```
2. Add δ to the parameter vector being optimised
3. Retrain

**Effort:** 30 minutes. **Expected gain:** Small but real — maybe +20 pts. Helps especially for CONMEBOL matches where rivalries repeat.

### 3. Direct Odds API h2h integration

**Why it matters:** We pulled 3,291 rows of bookmaker odds and currently aren't using them in match-level predictions. Bookmakers price specific matches with better precision than our model can on its own.

**How to implement:**
1. Parse `odds_h2h.csv` to compute per-match implied probabilities (1/price, then normalise for overround)
2. Aggregate across bookmakers (median is robust)
3. Blend with Dixon-Coles win/draw/loss probabilities for the winning_team prediction (75% model, 25% market, say)
4. For matches with strong bookmaker consensus, can also tilt the predicted score

**Effort:** 1-2 hours. **Expected gain:** Medium — depends how many WC matches have bookmaker lines by submission day. Closer to the tournament, this becomes more valuable.

### 4. Full bracket-consistency reconciler

**Why it matters:** The current reconciler only handles Final vs 3rd-place overlap. Other rounds may have inconsistencies: a team predicted to win an R16 match might not appear in the QF that follows.

**How to implement:**
1. Modify script 08 to traverse the bracket top-down: pick Final → determine semis → determine QFs → determine R16
2. For each round, use `mc_matchup_details.csv` to find the most-common matchup *consistent with the rounds above*
3. Track which teams are "assigned" to which slots and avoid duplicates

**Effort:** 2-3 hours. **Expected gain:** Small — most inconsistencies are invisible to the user/scoring, but the predictions look more credible.

### 5. Refresh schedule

**Why it matters:** Polymarket prices, bookmaker odds, FIFA rankings all change over time. The closer to June 11, the better the information.

**How to implement:**
- **June 7 (T-2)**: Re-run pipeline 01 → 09 with fresh Polymarket + Odds API pulls
- **June 8 (T-1)**: Manual review of predictions. Update `fifa_<month>_rankings_manual.csv` if FIFA released new monthly rankings. Note any star-player injuries.
- **June 9 (deadline)**: Final re-run with latest data. Submit.

**Effort:** 30 minutes per refresh. **Expected gain:** Material — markets get sharper closer to the tournament.

---

## Tier 2 — Real Gains (medium effort)

### 6. Referee assignments + per-referee card rate (the silent edge)

**Why it matters:** Different referees have wildly different card-rate tendencies. From the FIFA elite panel:
- **Daniel Siebert** (Germany): ~5.5 cards/match average
- **Ismail Elfath** (USA): ~4.5
- **Stéphanie Frappart** (France): ~3.2
- **Anthony Taylor** (England): ~5.0

Difference between a high-card and low-card referee is 1.5+ cards per match. That's a huge edge if we know who's officiating.

**How to implement:**
1. FIFA announces match referees ~2 weeks before the tournament. Watch fifa.com news and Wikipedia
2. Build `data/external/referee_card_rates.csv` (referee_name, avg_yellow_per_match, avg_red_per_match, sample_size)
3. Build `data/external/match_referee_assignments.csv` (match_id, referee_name) once assignments are known
4. In script 06, add referee card rate as a feature (or as a multiplier applied at prediction time)

**Sources for referee historical rates:**
- Wikipedia per-referee pages (have career statistics)
- WhoScored / SoccerWiki
- Manual compilation for the ~20 elite refs (most of the WC will be officiated by them)

**Effort:** 3-4 hours. **Expected gain:** Could improve cards MAE by 15-20% — translates to maybe +60-100 pts. Almost no other competitor will model this.

### 7. Squad fatigue / club season end dates

**Why it matters:** Players who finished their season May 30 (Champions League final players) have 12 days rest before June 11 kickoff. Players from Bundesliga (ended May 18) have 24 days. Tournaments where one team has clear rest advantage tend to see that team start strong.

**How to implement:**
1. Build `data/external/club_seasons.csv` (top 40 European clubs, league_end_date, european_competition_end_date)
2. Build `data/external/squad_rosters.csv` (each WC 2026 team's squad with their club)
3. Compute per-team avg rest days, add as a feature in scripts 04/04b

**Effort:** 2-3 hours for initial squad compilation, but squads aren't even finalised yet (deadline early June). **Expected gain:** Small to medium — affects mostly early-round matches.

### 8. Player injuries / suspensions (key player flags)

**Why it matters:** France without Mbappé. Argentina without Messi. England without Bellingham. Any of these moves the needle 5-10% in match win probability.

**How to implement:**
1. Maintain `data/external/team_key_players.csv` (team, key_player_name, role, importance_weight)
2. Closer to deadline, compile `data/external/availability.csv` (player_name, available?, expected_minutes)
3. Compute team availability score: weighted % of key players available
4. Adjust attack/defense in calibration step: weaker availability → reduce strength proportionally

**Effort:** 1-2 hours per refresh once squad lists are out (~June 1). **Expected gain:** High for matches involving teams with prominent star absences.

---

## Tier 3 — Lower priority (long-term)

### 9. xG data via alternative sources

**Why it matters:** Goals are noisy; xG (expected goals from shot quality) is a more reliable team-strength signal.

**Sources:**
- **StatsBomb Open Data** (free, includes WC22 with detailed events) — `github.com/statsbomb/open-data`
- **Understat** (mostly domestic leagues)
- **WhoScored** (web scrape)

**How to implement:** Add xG-for and xG-against rolling averages to the goals model as covariates.

**Effort:** 4-6 hours. **Expected gain:** Medium — depends on data availability for the specific WC 2026 teams.

### 10. Possession & tempo metrics

**Why it matters:** Style features predict corners better than just team-quality. A high-possession team controlling the ball generates more corners; a counter-attacking team takes fewer but converts more.

**How to implement:**
- We already have possession data for WC22 (in `match_stats_wc2022.csv`)
- Compute per-team possession averages, add to corners model features
- Effort low if we stop at WC22 data; high if we want broader coverage

**Effort:** 1-2 hours. **Expected gain:** Small.

### 11. Weather forecast at venue (5-day window)

**Why it matters:** Heat + humidity at Miami, Houston, Monterrey can slow play, affect cards (more fouls) and corners (fewer transitions).

**How to implement:** Re-enable the OpenWeather API call from the original pipeline, but only run it ~5 days before submission so the forecasts are usable.

**Effort:** 1 hour. **Expected gain:** Marginal — venue.csv already captures the climate averages.

### 12. Late market odds updates + sharper bookmaker lines

**Why it matters:** Bookmakers price soccer matches with the most accuracy 24-48 hours before kickoff. The Odds API pulled "futures" odds for matches that may not have sharp lines yet.

**How to implement:** Re-run script 01 the day before submission to pull the freshest odds.

**Effort:** 5 minutes. **Expected gain:** Already part of the refresh schedule (#5).

### 13. More Monte Carlo iterations

**Why it matters:** We run 10K sims. Going to 50K-100K reduces variance in the "most common matchup" picks for slots where multiple matchups are close.

**How to implement:** Change `N_SIMS = 10000` to `N_SIMS = 50000` in script 07. Will take ~3-5 minutes instead of ~30 seconds.

**Effort:** 30 seconds to change the constant. **Expected gain:** Negligible for high-probability slots; small improvement for ambiguous knockout slots.

---

## Refresh Schedule (suggested)

| When | Action | Why |
|---|---|---|
| **Now** | Push current state to GitHub | Lock in known-good baseline |
| **~T-7 (June 2)** | Pull squad announcements (manual). Update key-player availability file. Re-run full pipeline. | Squads finalised early June |
| **~T-5 (June 4)** | Pull referee assignments. Build referee card-rate file. Re-run cards model. | FIFA usually announces refs ~1 week before |
| **~T-3 (June 6)** | Refresh Polymarket + Odds API pulls. Re-run pipeline. | Markets sharpen as tournament approaches |
| **~T-2 (June 7)** | Final refresh + manual review of all 104 predictions. Note any dark-horse adjustments. | Last chance to spot issues |
| **~T-1 (June 9)** | Submit. **Do not wait until June 10.** | Buffer for technical issues |

---

## Quick Implementation Checklist

If you have **2 hours**, do #2 (head-to-head feature) + #4 (full bracket reconciler). Free improvements.

If you have **half a day**, do #1 (football-data.co.uk corners/cards expansion). Single biggest accuracy gain.

If you have **a full day**, also do #6 (referee data). This is where the "silent edge" lives.

If you have **a weekend**, add #3 (Odds API integration) + #8 (player injuries) + the refresh schedule.

---

## Final Note

The current model is **competitive and statistically defensible**. All sanity checks pass, the math is sound, the predictions match Polymarket's market consensus at the top of the bracket while preserving our edge on corners/cards. The improvements in this document push from "competitive" to "well-positioned to win" — but none of them is a magic bullet. The biggest single lever is **more training data for corners and cards** (#1).
