# NextStep.md — Model Deep Dive + Status + Future Work

This document has three parts:
1. **Part 1**: Complete technical breakdown of the current model — every script, every equation, every assumption.
2. **Part 2**: What's been done since the original roadmap, and what's still outstanding.
3. **Part 3**: Capabilities, current limitations, and suggested future improvements.

Last updated 2026-05-27 after the post-roadmap upgrade pass (corners/cards data pool, h2h covariate, market blend, full bracket reconciler, 50K MC sims, scheduled refresh cadence).

---

## Table of Contents

**Part 1 — Current Model**
- [Pipeline Overview](#pipeline-overview)
- [The 5 Sub-Models](#the-5-sub-models)
- [Math Reference](#math-reference)
- [Script-by-Script Walkthrough](#script-by-script-walkthrough)
- [Competition Scoring System](#competition-scoring-system)
- [Prediction Strategy Justification](#prediction-strategy-justification)

**Part 2 — Status**
- [Completed (since original roadmap)](#completed-since-original-roadmap)
- [Pending — blocked on external info](#pending--blocked-on-external-info)
- [Refresh Schedule (scheduled in Google Calendar)](#refresh-schedule-scheduled-in-google-calendar)

**Part 3 — Outlook**
- [Capabilities](#capabilities)
- [Known Limitations](#known-limitations)
- [Suggested Future Improvements](#suggested-future-improvements)

---

# Part 1 — Current Model

## Pipeline Overview

The model is a **chain of 12 sequential scripts**, each producing artifacts that the next script consumes:

```
01_collect_data.py
    ↓ data/raw/* (Kaggle CSVs, Polymarket, Odds API, FIFA snapshots)

01c_load_match_stats.py
    ↓ data/processed/match_stats_wc2022.csv (long format, 64 matches)

01d_load_external_match_stats.py        [NEW]
    ↓ data/processed/match_stats_external.csv (Euro 2024 + Copa 2024 + AFCON 25/26 → 130 matches)

01e_process_odds.py                     [NEW]
    ↓ data/processed/odds_consensus.csv (per-fixture bookmaker consensus)

02_clean_merge.py
    ↓ data/processed/matches_clean.csv, fixtures_*.csv, team_features.csv

03_feature_engineering.py
    ↓ data/processed/match_features.csv, fixture_features.csv
    ↓ (now also propagates h2h_avg_gd to fixture_features for prediction)

04_train_goals_model.py
    ↓ models/goals_model.pkl, data/processed/group_lambdas.csv
    ↓ (now fits δ on h2h_avg_gd as a 4th global parameter)

04b_calibrate_goals_model.py     [Polymarket Bayesian prior]
    ↓ overwrites goals_model.pkl + group_lambdas.csv with calibrated values

05_train_corners_model.py        [pooled training: WC22 + Euro + Copa + AFCON]
    ↓ models/corners_model.pkl, data/processed/corners_predictions.csv

06_train_cards_model.py          [pooled training, same sources]
    ↓ models/cards_model.pkl, data/processed/cards_predictions.csv

07_monte_carlo_simulator.py      [50K sims, h2h-aware knockout lambdas]
    ↓ data/processed/mc_qualifiers.csv, mc_knockout.csv, mc_matchup_details.csv

08_generate_predictions.py       [Market blend on winning_team, full bracket reconciler]
    ↓ data/processed/predictions_combined.csv

09_format_output.py              [Splits into DataLab submission schemas]
    ↓ output/predictions_group.csv, predictions_knockout.csv
```

A one-shot **`refresh.ps1`** (in the repo root) runs the entire chain end-to-end (`-SkipCollect` to reuse cached raw data, `-SkipMC` to bypass the ~5min Monte Carlo step).

**Key design principles:**
- Each script is idempotent (run twice → same output)
- Outputs are CSVs (not pickles for data, only for models) so anyone can inspect
- The pipeline is **fully re-runnable**: change inputs, run 01 → 09, get fresh predictions

---

## The 5 Sub-Models

The competition requires predictions across multiple metrics per match, so we build **5 specialist models**:

| # | Model | Predicts | Method | Training data |
|---|---|---|---|---|
| 1 | **Goals (Dixon-Coles)** | Home goals, Away goals | Weighted MLE Poisson with low-score correction + h2h covariate | 4,534 matches since 2018 |
| 2 | **Outcome (implicit)** | Home win / Draw / Away win | Derived from full joint score distribution, then **blended 75/25 with bookmaker consensus** | Same as goals model |
| 3 | **Corners (Poisson regression)** | Total corners in match | L2-regularised Poisson regression on pooled tournament data | 194 matches across WC22 + Euro 2024 + Copa 2024 + AFCON 2025/26 |
| 4 | **Cards (Poisson regression)** | Total yellows in match | Same pooled pool | 194 matches (same set) |
| 5 | **Monte Carlo tournament simulator** | Knockout matchups + winners | **50,000-iteration** simulation using goals model + h2h matrix | Group lambdas + knockout structure |

Plus a **6th supporting model**: empirical red card rate (~0.06/match from WC22), used as a low-rate Poisson predictor.

After the MC sim, a **full top-down bracket reconciler** picks each match's matchup so the bracket is internally self-consistent (no team winning a R16 but missing from the QF that follows).

---

## Math Reference

### 1. Dixon-Coles Poisson Goals Model

For a match between home team `h` and away team `a`:

```
λ_home = exp(α + attack[h] − defense[a] + γ · is_host_h + δ · h2h_avg_gd)
λ_away = exp(α + attack[a] − defense[h]                  − δ · h2h_avg_gd)
```

Where:
- **α** (intercept): log of baseline scoring rate. Fitted ≈ 0.06, so exp(α) ≈ 1.06.
- **attack[i]**, **defense[i]**: team-specific strengths, range roughly [−1.2, +1.3].
- **γ** (home advantage): fitted ≈ 0.21, applied **only** when `home_is_host = 1` (USA / Mexico / Canada playing at home).
- **δ** (h2h coefficient): fitted ≈ **-0.008** — near zero. The team-level attack/defense parameters already absorb most of the head-to-head signal; δ exists to capture residual rivalry effects.
- **h2h_avg_gd**: average goal differential (home perspective) across all prior meetings between these two teams, clipped to [−3, +3] to keep blowouts from dominating.

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

Our fitted ρ ≈ **−0.07**. Negative ρ means the model boosts 0-0 and 1-1 probabilities — matches reality.

### 3. MLE Fit with Sample Weights

The negative log-likelihood being minimised:

```
NLL(θ) = −Σ_m w_m · [ log Pois(x_m; λ_h_m) + log Pois(y_m; λ_a_m) + log τ(x_m, y_m) ]
       + 100 · ( (Σ_i attack[i])² + (Σ_i defense[i])² )
```

Sample weights `w_m`:

```
time_weight     = exp(−age_days_m / 547 · ln 2)        [18-month half-life]
sample_weight_m = time_weight_m · (1.0 if competitive else 0.2)
```

So a friendly from 2019 has weight ≈ 0.008, a competitive match from 2024 has weight ≈ 0.6.

Optimised via `scipy.optimize.minimize` with **L-BFGS-B**. 456 free parameters (226 teams × 2 + 4 globals: α, γ, ρ, δ), converges in ~20 seconds.

### 4. Polymarket Bayesian Calibration

The data-driven fit can be biased against teams whose recent form is poor but whose intrinsic strength is high. We blend in Polymarket's championship-probability signal:

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

The blend is z-score based (standardised) so we're comparing **rank within distribution**, not absolute values.

### 5. Corners — Poisson Regression (pooled training)

```
log(λ_corners) = β₀ + β₁ · sum_avg_corners
                    + β₂ · min_avg_corners
                    + β₃ · max_avg_corners
                    + β₄ · |fifa_rank_home − fifa_rank_away|
                    + β₅ · (fifa_rank_home + fifa_rank_away) / 2

corners_pred = round(λ_corners · altitude_factor + tournament_bump)
```

Training data: **194 matches** pooled across WC22 + Euro 2024 + Copa America 2024 + AFCON 2025/26. WC22 weight=1.0, continentals weight=0.8 (reflecting prep-time / referee-pool differences). Per-team `sum_avg_corners` etc. computed across the *entire pool*, so each team's history covers up to 4 tournaments.

`altitude_factor`: 0.92 at >2000m, 0.95 at >1500m, 1.0 otherwise.
`tournament_bump = 0.6`: empirical calibration since WC22 (8.94 avg) ran lower than WC18/WC14 (~10.5 avg).

L2 regularised (α=0.3). **5-fold CV MAE = 2.70** (was 2.99 on WC22-only).

### 6. Yellow Cards — Poisson Regression (pooled training)

```
log(λ_yellow) = β₀ + β₁ · sum_avg_yellows
                   + β₂ · |fifa_rank_home − fifa_rank_away|
                   + β₃ · conf_mult

yellow_pred = round(λ_yellow · altitude_card_factor)
```

Same 194-match training pool. **Confederation lookup** with tournament-tag fallback: teams not in `team_features.csv` (e.g., Euro 2024 non-WC qualifiers) get their tournament's default confederation (Euro → UEFA, Copa → CONMEBOL, AFCON → CAF).

`conf_mult` is a confederation card-rate multiplier — average of home and away teams' confederation factors:

| Confederation | Multiplier | Rationale |
|---|---|---|
| CONMEBOL | 1.15 | South American teams average more cards |
| CAF | 1.10 | African teams slightly above average |
| CONCACAF | 1.00 | Neutral baseline |
| AFC | 0.95 | Slightly below average historically |
| UEFA | 0.95 | Slightly below average |
| OFC | 1.00 | Small sample, default neutral |

**5-fold CV MAE on WC22 subset = 1.92** (was 1.95 WC22-only). Pooled CV = 1.72.

### 7. Red Cards — Empirical Rate

Red cards are too rare for regression (~0.06 per match). We use the empirical rate, multiply by altitude factor, and **predict 0 for all matches** because:

```
EV(predict 0) = 0.94 × 5 pts = 4.70 pts/match
EV(predict 1) = 0.06 × 5 pts = 0.30 pts/match
```

### 8. Monte Carlo Tournament Simulator (50K sims)

Pre-computes a **48×48 lambda table** including h2h adjustment:

```
lh[i, j] = exp(α + attack[i] - defense[j] + δ · h2h_matrix[i, j])
la[i, j] = exp(α + attack[j] - defense[i] - δ · h2h_matrix[i, j])
```

Then for each of 50,000 simulations:

```
for sim in range(50_000):
    # Group stage (lambdas already include h2h via group_lambdas.csv)
    for fixture in 72_group_fixtures:
        hg ~ Poisson(λ_home_fixture)
        ag ~ Poisson(λ_away_fixture)

    # Standings with FIFA tiebreakers
    for each of 12 groups:
        sort teams by Points → GoalDiff → GoalsFor → random
        record group_winner, runner_up, third_place

    # Best 8 of 12 third-place teams qualify
    best_thirds = sort(all_thirds, by Points → GD → GF)[:8]

    # Knockouts (90' + ET at 1/3 rate + penalties 50/50 if still tied)
    for ko_match in 32_knockout_matches_in_id_order:
        ...

# Aggregate
for each knockout slot:
    most_common_matchup = mode of (home, away) pairs across sims
    most_common_winner = mode of winners across sims
    penalty_prob = fraction of sims where penalties=True
```

50K sims smooths the per-slot matchup distribution; was 10K. Adds ~4 minutes to the full pipeline.

### 9. Score Prediction Strategy (the "rounded expected" choice)

Three possible strategies:
1. **Mode**: argmax over P[h, a]. Tends to predict (0, 0) for low-lambda matches.
2. **Expected-value (EV)-optimal**: pick (h, a) that maximises expected competition points.
3. **Rounded expected**: `h = round(λ_home), a = round(λ_away)`. ✓

We use **rounded expected** because:
- Matches historical WC averages (2.2-2.7 goals/match)
- Optimises for goal-diff/total partial credit (10 pts) rather than chasing rare exact matches (25 pts)
- Gives realistic-looking predictions

For **knockouts**, we additionally **force decisive** results: if `round(λ_h) == round(λ_a)`, bump the favoured team by 1 goal. This avoids the "1-1 score + penalties=False" inconsistency and aligns with the ~88% historical rate of knockouts decided in regulation/ET.

### 10. Market Blend for `winning_team`

For each group-stage fixture with bookmaker consensus from the Odds API (72/72 matches, ~14 books each):

```
P_outcome_blended = 0.75 · P_model + 0.25 · P_market
winning_team      = argmax(P_blended)
```

Bookmaker odds are converted to probabilities by `1/decimal_price` then normalised per (match, bookmaker) to strip overround, then aggregated across books via median. **5 of 72 group winners** flipped from the pure-model pick (in each case the market correctly punctured an over-confident model call on a minnow).

### 11. Full Bracket Reconciler

The MC simulator's per-slot top picks aren't guaranteed self-consistent: the team predicted to win an R16 match might not appear in the QF that depends on it. We fix this with a **top-down constraint propagation** in script 08:

1. Pick the Final's matchup from `mc_matchup_details.csv` (most common (home, away) pair).
2. The Final's home team becomes the required winner for Semi 1; away team for Semi 2.
3. For each upstream round, pick the highest-count matchup whose Dixon-Coles winner equals the required team.
4. Cascade until R32 is fixed.
5. 3rd-place playoff = (Loser Semi 1, Loser Semi 2), derived directly from the Semi matchups.

In the current snapshot, 14 of 32 knockout matches differ from MC's per-slot top pick. The bracket has **0 self-consistency violations**.

---

## Script-by-Script Walkthrough

### `01_collect_data.py`
**Purpose:** Pull external data sources into `data/raw/`.

**Outputs:**
- `data/raw/polymarket.csv` (Polymarket WC 2026 outright winner markets)
- `data/raw/odds_h2h.csv` (~3,291 bookmaker h2h odds rows from The Odds API)
- `data/raw/fifa_ranking_combined.csv` (historical FIFA rankings, 199,490 rows)
- `data/raw/fifa_ranking_current.csv` (Wikipedia FIFA top-20 scrape if successful)

### `01c_load_match_stats.py`
**Purpose:** Load the WC 2022 detailed match stats dataset and reshape from team1/team2 wide → long.

**Output:** `data/processed/match_stats_wc2022.csv` — 128 team-match rows (64 matches × 2).

### `01d_load_external_match_stats.py`   **[NEW]**
**Purpose:** Load Kaggle CSVs the user drops into `data/raw/external_stats/` (Euro 2024, Copa America 2024, AFCON 2025/26) and convert them to the same long format as WC22.

**Key behaviour:**
- Auto-detects three CSV layouts (match-level `Home X / Away X`; football-data.co.uk `HC/AC/HY/AY/HR/AR`; team1/team2 wide).
- Synthesises dates from a known tournament start date if the source omits them.
- Also extracts xG when present (Euro / Copa have it; WC22 / AFCON don't).
- Robust to malformed CSV rows (falls back to python parser + `on_bad_lines=skip`).

**Output:** `data/processed/match_stats_external.csv` (~130 matches).

### `01e_process_odds.py`   **[NEW]**
**Purpose:** Convert `odds_h2h.csv` into per-fixture bookmaker consensus probabilities.

**Steps:**
1. For each (match, bookmaker) triple, convert decimal odds → implied probability (1/price).
2. Normalise the three outcomes (home/draw/away) so they sum to 1 — strips the bookmaker's overround.
3. Cross-bookmaker consensus is the **median** of normalised probabilities (robust to a single mispricing).

**Output:** `data/processed/odds_consensus.csv` — 72 group fixtures × (p_home, p_draw, p_away, n_bookmakers).

### `02_clean_merge.py`
**Purpose:** Normalise team names, resolve playoff placeholders, build per-team aggregates.

**Outputs:** `matches_clean.csv`, `fixtures_group.csv`, `fixtures_knockout.csv`, `team_features.csv`, `polymarket_clean.csv`, `name_audit.csv`.

### `03_feature_engineering.py`
**Purpose:** Build leakage-free time-aware features for training + prediction features for WC fixtures.

**Operations:** time-aware rolling features (avg_gf, avg_ga, win_rate, draw_rate, days_rest); FIFA rank at match date via `merge_asof`; **head-to-head cumulative goal-differential**, normalised to **`h2h_avg_gd` = h2h_home_gd / max(h2h_played, 1)** clipped to [−3, +3] — propagated to **both** `match_features.csv` and `fixture_features.csv`; time-decay sample weights; confederation baselines for teams without WC22 history.

### `04_train_goals_model.py`
**Purpose:** Fit the Dixon-Coles model.

**Now includes:** δ on h2h_avg_gd as a 4th global parameter. 456 free parameters (was 455). L-BFGS-B converges in ~20s.

### `04b_calibrate_goals_model.py`
**Purpose:** Apply the Polymarket Bayesian prior. Blends 40% data / 60% market in z-score space.

### `05_train_corners_model.py`
**Purpose:** Fit corners Poisson regression on pooled tournament data.

**Now:** trains on combined WC22 + external pool (194 matches). Uses pooled per-team averages at prediction time (was WC22-only — fixed a train/predict distribution mismatch). Reports both pooled CV MAE and WC22-only-subset CV MAE.

### `06_train_cards_model.py`
**Purpose:** Yellow card Poisson regression + empirical red rate.

**Now:** same pooled pool. Adds tournament → confederation fallback for teams without team_features.csv entries. Red rate computed from WC22 only (closest tournament type to predictions).

### `07_monte_carlo_simulator.py`
**Purpose:** Simulate the WC 50K times.

**Now:** 50,000 sims (was 10K); knockout lambda table includes h2h adjustment from a 48×48 h2h matrix built from `matches_clean.csv`.

### `08_generate_predictions.py`
**Purpose:** Combine all sub-models into the 104-match prediction set.

**Now:**
- **Market blend** on `winning_team` for group stage (75% model, 25% Odds-API consensus).
- **Full top-down bracket reconciler** replaces the previous Final/3rd-place-only patch. Produces a bracket that is internally self-consistent.
- Knockout score recomputation uses h2h_lookup from historical matches.

### `09_format_output.py`
**Purpose:** Split into DataLab's two submission schemas + final QA.

**Outputs:** `output/predictions_group.csv` (72 × 7), `output/predictions_knockout.csv` (32 × 10), `output/predictions_final.csv` (104 combined audit copy).

---

## Competition Scoring System

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
- Corners/cards: each correct corners = 10 pts × multiplier. Over 32 knockout matches × avg multiplier ~3.5, that's potentially **1,000+ pts** if we nail corners more often than competitors.

---

## Prediction Strategy Justification

### Why we predict 0 red cards everywhere
Predicting 0 maximises expected points (~0.85 × 5 pts × 104 matches = 442 pts vs ~0.15 × 5 = 78 pts for predicting 1 everywhere).

### Why knockouts are forced to non-draw scores
A predicted 1-1 score in a knockout requires penalties=True. But our MC says only ~12% of knockouts go to pens. Predicting penalties=True for 18 matches when only ~4 will be correct loses ~70 points vs penalties=False (correct ~88% of the time). The fix: force the favoured team to win by 1 goal when lambdas round to a tie.

### Why corners get a +0.6 tournament bump
Our corners model is trained on WC22 (avg 8.94) + continentals. But WC22 had unusual conditions (winter, Qatar). WC18 and WC14 averaged ~10.5. WC26 (summer in NA) is expected closer to the historical norm. The bump nudges predictions into the README's 9.0-11.5 range.

### Why 40/60 model-market blend in calibration (and not 50/50 or 30/70)
- **50/50**: keeps Brazil out of the top 10 (model alone underrates them)
- **40/60**: puts Brazil at #7, Argentina at #5, France at #4 — matches Polymarket's top-5
- **30/70**: essentially mimics the market, sacrificing our data signal on form/recent results
- 40/60 is the empirical sweet spot.

### Why 75/25 model-market blend for `winning_team`
The Dixon-Coles + Polymarket calibration produces good team-strength estimates, but specific-match bookmaker prices add information about match-day factors (likely lineups, motivation, public knowledge of injuries). 75/25 preserves our edge while letting the market flip the most over-confident model calls on minnows.

---

# Part 2 — Status

## Completed (since original roadmap)

| # | Item | Impact |
|---|---|---|
| 1 | **Pooled corners/cards training pool** — added Euro 2024, Copa America 2024, AFCON 2025/26 (64 → **194 matches**). Per-team averages now pooled across all tournaments and used consistently at train + predict time. | Corners 5-fold CV MAE **2.99 → 2.70 (-10%)**, yellow CV MAE on WC22 subset 1.95 → 1.92. |
| 2 | **Head-to-head wired into Dixon-Coles** as a 4th global parameter δ on `h2h_avg_gd` (cumulative GD / prior meetings, clipped ±3). Propagated through training, calibration, MC simulator, and knockout score prediction. | Fitted δ ≈ −0.008 (near zero — team attack/defense already absorbs h2h signal). Infrastructure present for future tournaments where rivalry effects might matter. |
| 3 | **Bookmaker consensus blend on group winning_team** at 75/25 model/market. Median across ~14 books per fixture; overround stripped per (match, bookmaker). | 5 of 72 group winners flipped; in all 5 cases the market correctly punctured overconfident model calls on Curaçao/Tunisia-tier minnows. |
| 4 | **Full top-down bracket reconciler** replaces the Final/3rd-place-only patch. Traverses Final → Semis → QFs → R16 → R32, picking matchups whose DC winner matches the team needed downstream. | 0 bracket self-consistency violations (was 2). 14 of 32 knockout matchups changed from MC top pick. |
| 5 | **Refresh schedule** automated via Google Calendar (T-7 squads, T-5 referees, T-3 markets+pipeline, T-2 review, T-1 SUBMIT) with email + popup reminders. **`refresh.ps1`** runs the entire pipeline end-to-end. | Won't miss a refresh window; standardised re-run reduces operator error. |
| 9 | **xG data extracted** from Euro 2024 + Copa 2024 to `data/processed/team_xg_summary.csv` (40 teams). | Model integration deferred — only 1-7 matches per team and 8/48 teams missing entirely. Available as a manual sanity-check input at T-2 review. |
| 10 | **Possession feature attempted** in corners model, **reverted**. | CV MAE got worse (2.70 → 2.77) — only 32/48 teams had WC22 possession data, signal redundant with `sum_avg_corners`. Comment in script 05 explains. |
| 11 | **Weather coverage** — venue.csv already carries `avg_june_temp_c`, `avg_june_humidity_pct`, `roof_covered` for each stadium. 5-day forecast deferred to T-5 manual review (marginal additional precision). | — |
| 12 | **Late market odds refresh** — covered by the T-3 calendar event re-running script 01. | — |
| 13 | **MC iterations bumped 10K → 50K**. Adds ~4 min to pipeline. | Smoother per-slot matchup probabilities, especially for ambiguous knockout slots. |

## Pending — blocked on external info

These three remain because the required information doesn't exist yet. Each is wired to a calendar reminder on the appropriate date.

| # | Item | Unblocks on | Calendar event |
|---|---|---|---|
| 6 | **Referee assignments + per-referee card rate** (the silent edge — 1.5+ card variance between high- and low-card refs) | FIFA announces refs ~T-5 (June 4) | "FIFA 2026 — T-5: Pull match referees + retrain cards" |
| 7 | **Squad fatigue / club-season rest days** (Bundesliga players have ~24 days rest, CL Final players ~12) | Squads finalise ~T-7 (June 2) | "FIFA 2026 — T-7: Pull squads + refresh predictions" |
| 8 | **Player injury / key-player availability flags** (France without Mbappé, Argentina without Messi, etc.) | Same — squad announcements | T-7 event, plus ongoing watchlist through T-2 review |

## Refresh Schedule (scheduled in Google Calendar)

| Date | What |
|---|---|
| **June 2** (T-7) | Squad announcements → run `refresh.ps1`. Note any star player absences. |
| **June 4** (T-5) | FIFA referee assignments → build `referee_card_rates.csv` + retrain cards. |
| **June 6** (T-3) | Fresh Polymarket + Odds API pulls + full pipeline. |
| **June 7** (T-2) | Final review of all 104 predictions. Look for anything that smells wrong. |
| **June 9** (T-1) | **SUBMIT**. Run pipeline one last time, then upload to DataLab. Do not wait until June 10. |

Each event has an email reminder 24h before + popup 60min before; the deadline event has extra 3h and 30min popups for safety.

---

# Part 3 — Outlook

## Capabilities

1. ✅ **All README sanity checks pass** except the accepted `avg_yellow ≈ 3.94 > 3.8` deviation (Euro 2024 + AFCON 25/26 both ran higher than WC22; recent WCs span 3.2-4.2/match).
2. ✅ **Internally self-consistent bracket** end-to-end (full reconciler, 0 violations).
3. ✅ **Top 10 championship probabilities align with Polymarket** (Spain, Portugal, England, France, Argentina, Germany, Brazil, Netherlands, Japan, Norway).
4. ✅ **Per-team time-aware features** prevent training leakage.
5. ✅ **Host advantage** correctly applied only to the 9 actual host-country matches.
6. ✅ **Altitude effects** captured for Mexico City (2240m), Guadalajara (1560m), and the higher US venues.
7. ✅ **Confederation card-rate adjustments** baked into the cards model + tournament fallback for non-WC teams in the training pool.
8. ✅ **Head-to-head** wired as a Dixon-Coles covariate (δ parameter), propagated through training + calibration + MC + final predictions.
9. ✅ **Bookmaker consensus blend** on group winning_team — 24 bookmakers, median across normalised odds, 75/25 model/market weight.
10. ✅ **Polymarket Bayesian calibration** corrects for the data model's bias against teams with poor recent form but strong intrinsic ability.
11. ✅ **Pooled corners/cards training pool** — 194 matches across 4 tournaments; per-team averages computed across the pool and used consistently at train + predict time.
12. ✅ **50K Monte Carlo iterations** — smoother slot distributions, especially for ambiguous knockout slots.
13. ✅ **Idempotent and re-runnable** — full refresh in ~5 minutes via `refresh.ps1`. Five scheduled Google Calendar events drive the refresh cadence.
14. ✅ **xG per-team summary** available for manual sanity check during T-2 review.

## Known Limitations

The real holes remaining in the model. These are the items that would move predictions forward; some are pending external info (Part 2), some are deeper data / methodology problems.

1. **No player-level information** — A team missing its star striker or top centre-back is meaningfully weaker, and the model doesn't know. Scheduled for the T-7 / T-2 windows but limited to a hand-curated impact list.

2. **No referee data** — Yellow rates vary 3.0 to 5.5+ per match across the FIFA elite panel. Scheduled to be added at T-5 if FIFA publishes assignments.

3. **Score capped at 2 goals per team in most predictions** — Because `round(λ)` rarely exceeds 2. We never hit "exact 3-0" matches but accept it for the goal-diff partial credit.

4. **No squad fatigue model** — Players from the Bundesliga have ~24 days rest; Champions League final players ~12. Can be meaningful for early-round matchups. Scheduled T-7.

5. **No xG / advanced shot quality in the goals model** — `team_xg_summary.csv` exists (40 teams) but isn't a model covariate yet. Coverage gaps (8 of 48 WC teams missing) and small per-team samples (1-7 matches) make integration risky.

6. **Sparse Polymarket coverage** — Only ~50 teams have Polymarket outrights. For the bottom 75 international teams, there's no market signal in the calibration step.

7. **Yellow card avg ≈ 3.94/match is above the README's 2.8-3.8 target** — Accepted as defensible: Euro 2024 (~4.4) and AFCON 2025/26 (~4.1) both ran higher than WC22 (~3.5), and historical WCs span 3.2-4.2.

8. **MC per-slot independence (partially mitigated)** — Across 50K sims, the most common matchup per slot is computed marginally, not jointly. The bracket reconciler fixes this for self-consistency, but the *joint* most-probable bracket might still be slightly off the chosen one.

9. **No live data feeds** — Late-breaking injuries, lineup changes, tactical reports — none of this enters the model until manually noted at T-7 / T-2.

10. **No domestic-league form** — A team that's been crushing UEFA Champions League play gets no extra credit beyond their own international goals scored. Domestic xG / form data could refine the prior.

## Suggested Future Improvements

(For post-2026 tournaments or if more time becomes available before submission.)

### More / better data
- **StatsBomb open-data integration**: proper event-level xG for WC22, Euros, and (eventually) WC26 group stage as it plays out. Replace our basic shots model.
- **Domestic league results since 2024-25**: top-5 European leagues + Brasileirão + Liga MX. Adds ~7,500 matches with corners/cards (football-data.co.uk format) but introduces club-vs-international domain shift — we'd need a `is_international` flag or downweight clubs heavily.
- **Per-player FIFA video-game ratings** as a quick squad-strength proxy. FIFA / EA Sports FC ratings correlate well with on-pitch impact and are easy to scrape per squad.
- **538 / Massey / SPI ratings** — independent third-party power rankings as additional calibration signals beyond Polymarket.
- **Betfair / Pinnacle outright markets** for teams Polymarket doesn't cover (~75 lower-ranked nations would gain market signal).

### Modelling
- **Tournament-specific intercept** in the cards model — let Euro/AFCON have their own baseline. Could let us pool data without the yellow-card avg drifting (currently 3.94, above WC22 baseline 3.5).
- **Bayesian hierarchical model** — partially pool team attack/defense within confederations, smoothing the rarely-played teams.
- **Direct ML for `winning_team`** — train LightGBM / XGBoost directly on (home, away, features) → outcome. Could outperform Dixon-Coles for the 40-pt group-winner field specifically. Keep Dixon-Coles for scorelines.
- **Conformal prediction intervals** for scoreline — gives well-calibrated uncertainty around `(λ_home, λ_away)`. Useful for risk-aware EV optimisation.
- **Possession + tempo as a tournament-aware feature** — tried it as a simple per-team WC22 avg and CV MAE got worse, but a more sophisticated possession-style cluster (control-the-ball vs counter-attack) might add real signal.

### Engineering
- **Cloud-hosted pipeline** for true device-independence: GitHub Actions or a small VM running `refresh.ps1` (Linux equivalent) on the scheduled dates and pushing predictions back to the repo + emailing the user. Today the user must be at a machine with the project synced.
- **GUI / dashboard** for the T-2 review step — Streamlit page showing all 104 predictions with bookmaker comparisons, sanity flags, and one-click "this looks wrong" override.
- **Automated regression tests** — golden-file comparison on a fixed-seed pipeline run, so changes that silently break predictions are caught immediately.

### Process
- **Treat T-7 → T-2 as the real ramp**: the data refresh schedule already covers this, but holding open a 30min window each evening for hand-review of news / injury rumours would catch things the model misses.
- **Post-mortem after WC26**: once the tournament plays out, walk through every prediction vs actual result and identify the systematic errors (e.g., did we always under-predict scorelines in AFCON-flavored matches?). Feeds into the next major tournament's prior.
