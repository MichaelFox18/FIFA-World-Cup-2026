# FIFA World Cup 2026 Prediction Model
## Claude Code Project Guide

> This README is the single source of truth for building this project step by step in VS Code using Claude Code.
> Read this fully before writing any code. Every phase, file, and decision is documented here.

---

## Competition Overview

- **What:** Predict scores, corners, yellow cards, and red cards for all 104 FIFA World Cup 2026 matches
- **Deadline:** June 10, 2026 at 09:00 UTC — submit predictions to DataLab notebook
- **Format:** 48 teams, 12 groups of 4, then knockout rounds (Round of 32 through Final)
- **Key rule:** Predictions must be submitted before a single match is played

### What to Predict Per Match

| Prediction | Group Stage | Knockout |
|---|---|---|
| Score (home-away) | ✅ | ✅ |
| Corners | ✅ | ✅ |
| Yellow cards | ✅ | ✅ |
| Red cards | ✅ | ✅ |
| Winning team (`home`/`away`/`draw`) | ✅ | ❌ |
| Matchup (which two teams are playing) | ❌ | ✅ |
| Match winner (`home`/`away`) | ❌ | ✅ |
| Penalties (`True`/`False`) | ❌ | ✅ |

### Scoring System

| Prediction | Condition | Points |
|---|---|---|
| Score | Exact scoreline | 25 |
| Score | Correct goal difference OR correct total goals | 10 |
| Corners | Exact | 10 |
| Corners | Off by ±2 | 5 |
| Yellow cards | Exact | 10 |
| Yellow cards | Off by ±1 | 5 |
| Red cards | Exact | 5 |
| Winning team (group) | Correct | 40 |
| Matchup (knockout) | Both teams correct | 20 |
| Matchup (knockout) | One team correct | 10 |
| Match winner (knockout) | Correct | 20 |
| Penalties (knockout) | Correct | 5 |

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

---

## Competitive Strategy

The following principles should guide every modelling decision:

1. **Outcome > exact score.** A correct group stage winner (40 pts) beats a perfect score prediction (25 pts). Prioritise getting the right winner.

2. **Corners and cards are the silent edge.** Most competitors will guess these. A dedicated model for corners and cards can earn hundreds of quiet points across 72 group stage matches that others leave on the table.

3. **Use betting markets as a Bayesian prior.** Prediction markets aggregate millions of informed opinions. Trust them as a baseline and only diverge when you have a specific structural reason (fatigue, travel, referee profile, altitude).

4. Always predict whatever maximises your expected points, regardless of what anyone else does.

5. **The UEFA Playoff teams are an edge.** Matches 2, 3, 5, and 12 involve unknown UEFA Playoff qualifiers. Most models will assign them generic UEFA-average strength. Research the actual playoff paths and assign real probability-weighted team strength estimates.

6. **Validate against the right tournaments.** Use 2022 World Cup, Euro 2024, and Copa América 2024 as validation sets — not random historical internationals. Tournament football has different dynamics.

---

## Project Structure

Build the project in this exact folder structure:

```
worldcup2026/
├── data/
│   ├── raw/                    # Downloaded files, never modified
│   ├── processed/              # Cleaned, merged DataFrames
│   └── external/               # Manually compiled tables (venues, name maps)
├── models/
│   ├── goals_model.pkl
│   ├── outcome_model.pkl
│   ├── corners_model.pkl
│   └── cards_model.pkl
├── scripts/
│   ├── 01_collect_data.py
│   ├── 02_clean_merge.py
│   ├── 03_feature_engineering.py
│   ├── 04_train_goals_model.py
│   ├── 05_train_corners_model.py
│   ├── 06_train_cards_model.py
│   ├── 07_monte_carlo_simulator.py
│   ├── 08_generate_predictions.py
│   └── 09_format_output.py
├── notebooks/
│   └── eda.ipynb               # Exploratory analysis only
├── output/
│   └── predictions_final.csv   # Final formatted predictions for DataLab
├── requirements.txt
└── README.md                   # This file
```

---

## Tech Stack

```
Python 3.11+

Core:
- pandas
- numpy
- scipy

Modelling:
- scikit-learn
- xgboost
- lightgbm
- statsmodels          # for Poisson/Dixon-Coles

Scraping:
- requests
- beautifulsoup4
- soccerdata           # wraps FBRef and other sources

Utilities:
- tqdm                 # progress bars
- python-dotenv        # API keys
- joblib               # model serialisation
```

Create `requirements.txt` with all of the above. Install with `pip install -r requirements.txt`.

---

## Data Sources

### Priority 1 — Download Immediately (Free Static Files)

| Dataset | URL | Method |
|---|---|---|
| International match results 1872–2024 | https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017 | Kaggle download |
| FIFA World Rankings historical | https://www.kaggle.com/datasets/cashncarry/fifaworldranking | Kaggle download |
| football-data.co.uk internationals | https://www.football-data.co.uk/international.php | Direct CSV |

### Priority 2 — Scrape (Automated)

| Dataset | Source | Script |
|---|---|---|
| Club Elo ratings | http://clubelo.com/API | `01_collect_data.py` — clean REST API, returns CSV |
| World Football Elo ratings | https://www.eloratings.net | `01_collect_data.py` — scrape table |
| Team xG, xGA, possession, corners | https://fbref.com/en/comps/international | `01_collect_data.py` — use `soccerdata` package |
| Squad market values | https://www.transfermarkt.com/weltmeisterschaft-2026/teilnehmer/pokalwettbewerb/WM26 | `01_collect_data.py` — scrape |
| Disciplinary records (yellows/reds) | https://fbref.com | `01_collect_data.py` |

### Priority 3 — API

| Dataset | Source | Notes |
|---|---|---|
| Betting odds (match outcomes) | https://the-odds-api.com | Free tier: 500 requests/month. Sign up for API key. Store in `.env`. |
| Weather by venue (June forecast) | https://openweathermap.org/api | Free tier. Query each of the 16 venue cities. |

### Priority 4 — Manual Compilation

Build these as hardcoded Python dictionaries or small CSV files in `data/external/`:

**Venue lookup table** — 16 venues with:
- City
- Altitude (metres above sea level)
- Average temperature in June (°C)
- Surface type (grass/hybrid)

**Team name master map** — every team's name variant across all datasets normalised to one canonical name. Critical — mismatched names will silently break all joins. Example:
```python
NAME_MAP = {
    "Ivory Coast": "Côte d'Ivoire",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "USA": "United States",
    # etc.
}
```

**UEFA Playoff probability table** — for each of the 4 playoff slots (A, B, C, D), list the possible teams and their estimated probability of qualifying. Research the actual playoff bracket and assign weights. Store as a dict of dicts.

**Club season end dates** — for major clubs with World Cup players, note when their domestic/European season ended. Players with <3 weeks rest before first match get a fatigue flag.

---

## Model Architecture

Build five separate specialist models. Do not combine them into one.

### Model 1: Goals / Score Model (Dixon-Coles Poisson)

**Purpose:** Predict home goals and away goals independently for every match.

**Method:**
- Fit attack and defense strength parameters per team using Maximum Likelihood Estimation on historical data
- `lambda_home = exp(intercept + attack_home + defense_away + home_advantage_flag)`
- `lambda_away = exp(intercept + attack_away + defense_home)`
- Apply Dixon-Coles correction for low-scoring results (0-0, 1-0, 0-1, 1-1 are slightly underestimated by base Poisson)
- Simulate each match 10,000 times to get full score probability distribution
- Pick prediction strategy: use the mode of the distribution for "safe" predictions, or use expected value — test both in validation

**Key features:**
- Elo rating differential
- FIFA ranking differential
- Recent form (last 10 matches): goals scored, goals conceded
- xG and xGA rolling averages (last 15 matches)
- Head-to-head record
- Tournament adjustment (teams play more cautiously in World Cups than friendlies)
- Home advantage proxy (venue proximity to team's home country)
- Altitude of venue
- Days rest since last match

**Validation target:** Mean Absolute Error on goals per team per match < 0.8

### Model 2: Match Outcome Model

**Purpose:** Predict win/draw/loss independently of the score model. Use as a cross-check and for group stage winner predictions.

**Method:** XGBoost multiclass classifier (3 classes: home win, draw, away win)

**Key features:** Elo diff, FIFA rank diff, recent form, head-to-head, betting market implied probabilities

**Validation target:** Accuracy > 55% on 2022 World Cup matches

### Model 3: Corners Model

**Purpose:** Predict total corners in a match. This is the primary competitive differentiator.

**Method:** XGBoost regressor

**Key features:**
- Both teams' average corners for and against (last 15 matches)
- Possession stats (high possession teams win more corners)
- Tempo/press intensity (high press = more turnovers = fewer corners)
- Venue (some grounds have tighter angles, affecting corner rates)
- Match importance flag (dead rubbers in group stage have more open play)
- Attacking vs defensive team matchup (0-0 slogfests = few corners; open games = many)

**Validation target:** Mean Absolute Error < 2.5 corners per match on 2022 World Cup

### Model 4: Yellow Cards Model

**Purpose:** Predict yellow cards per match.

**Method:** Poisson regression (yellow cards are count data)

**Key features:**
- Both teams' average yellows per match (last 20 matches)
- Confederation (South American and African teams average higher card rates)
- Referee assignment (when available from FIFA — add this late in the process)
- Match stakes (knockout matches and elimination group matches have more cards)
- Altitude (physical, aerial play increases at altitude)
- Style matchup (counter-attacking vs possession creates more fouls)

**Note on red cards:** Red cards are rare (~0.15 per match). Model separately with a very low-lambda Poisson. Default prediction for most matches should be 0.

**Validation target:** MAE < 1.0 yellow cards per match on 2022 World Cup

### Model 5: Monte Carlo Tournament Simulator

**Purpose:** Simulate the full tournament 50,000+ times to generate probability distributions for every possible knockout matchup. This feeds all knockout stage predictions.

**Method:**
1. For each simulation, use the goals model to simulate all 72 group stage matches
2. Calculate group standings using correct FIFA tiebreaker rules (points → GD → GF → H2H → drawing of lots)
3. Determine which 32 teams qualify (group winners, runners-up, and 8 best third-placed teams)
4. Simulate knockout bracket through to the final
5. Record results: who plays who in every round slot, who wins, does it go to penalties

**Output:** For each of the 32 knockout slots, a probability distribution over all possible matchup combinations. Use the highest-probability matchup as your prediction.

**Penalties model:** Historically ~25-30% of knockout matches that reach 90 minutes level go to extra time, and roughly half of those go to penalties. Use this base rate adjusted by team styles (defensive teams more likely to go to pens).

---

## Build Timeline

### Phase 1: Data Collection (May 22–25)

- **Day 1 (May 22):** Set up project structure, install dependencies, download all static Kaggle datasets, save competition CSVs from DataLab
- **Day 2 (May 23):** Write and run scrapers for Club Elo, World Football Elo, current FIFA rankings. Build team name master map.
- **Day 3 (May 24):** Scrape FBRef for xG, corners, disciplinary data. Scrape Transfermarkt for squad values.
- **Day 4 (May 25):** Set up The Odds API, pull current WC odds. Manually compile venue lookup table and UEFA Playoff probability table. Note club season end dates for key squads.

### Phase 2: Feature Engineering (May 26–28)

- **Day 5 (May 26):** Clean all raw data, standardise team names using master map, merge all sources into one team features DataFrame. Build historical match-level dataset with features for both teams at time of each match.
- **Day 6 (May 27):** EDA — plot goal/corner/card distributions, validate Elo predictive power, check for data quality issues, identify which features correlate with outcomes.
- **Day 7 (May 28):** Build rolling average features (last 5, 10, 20 matches), tournament-specific adjustment factors, match importance flags, head-to-head records.

### Phase 3: Model Building (May 29 – June 2)

- **Day 8 (May 29):** Build and fit Dixon-Coles Poisson goals model. Implement 10,000-simulation match scorer.
- **Day 9 (May 30):** Build XGBoost outcome classifier. Cross-validate against goals model.
- **Day 10 (May 31):** Build corners regression model. Tune on Euro 2024 and Copa América 2024.
- **Day 11 (June 1):** Build yellow cards model. Build red cards low-lambda Poisson. Validate both on 2022 WC.
- **Day 12 (June 2):** Build Monte Carlo tournament simulator. Run 50,000 iterations. Inspect bracket probability outputs.

### Phase 4: Validation & Calibration (June 3–5)

- **Day 13 (June 3):** Full backtest — run entire pipeline on 2022 World Cup as if predicting blind. Score using competition's exact scoring system. Identify weakest sub-models.
- **Day 14 (June 4):** Calibrate against betting odds. Where your outcome probabilities diverge significantly from market odds, investigate why. Tune corners/cards models on Euro 2024.
- **Day 15 (June 5):** Sensitivity analysis on bracket predictions. Decide on contrarian dark horse pick. Lock model parameters — no more changes after today.

### Phase 5: Predictions & Submission (June 6–9)

- **Day 16 (June 6):** Generate all 72 group stage predictions. Sanity check averages.
- **Day 17 (June 7):** Generate all 32 knockout predictions using Monte Carlo output. Apply score model to most probable matchups.
- **Day 18 (June 8):** Format all predictions into DataLab notebook structure. Final null check. Final sanity checks (see below).
- **Day 19 (June 9):** Submit. Do not wait until June 10.

---

## Sanity Checks Before Submission

Run these checks on your final predictions before submitting:

```python
# Goals
assert 2.2 <= predictions['total_goals'].mean() <= 2.8, "Average goals out of range"

# Corners
assert 9.0 <= predictions['corners'].mean() <= 11.5, "Average corners out of range"

# Yellow cards
assert 2.8 <= predictions['yellow_cards'].mean() <= 3.8, "Average yellows out of range"

# Red cards
assert predictions['red_cards'].mean() < 0.25, "Too many red cards predicted"
assert predictions['red_cards'].sum() <= 20, "Red card total too high for 104 matches"

# No nulls
assert predictions.isnull().sum().sum() == 0, "Null values found — will score 0"

# All 104 matches present
assert len(predictions) == 104, "Wrong number of matches"

# Penalties only in knockout matches
knockout = predictions[predictions['match_id'] >= 73]
assert knockout['penalties'].isin([True, False]).all(), "Invalid penalties values"
```

---

## Output Format

The final `predictions_final.csv` must have these columns:

**Group stage (match_id 1–72):**
```
match_id, home_score, away_score, corners, yellow_cards, red_cards, winning_team
```

**Knockout stage (match_id 73–104):**
```
match_id, home_team, away_team, home_score, away_score, corners, yellow_cards, red_cards, match_winner, penalties
```

Where:
- `winning_team` is `"home"`, `"away"`, or `"draw"`
- `match_winner` is `"home"` or `"away"` (no draw in knockout)
- `penalties` is `True` or `False`
- Scores are after 90 minutes + extra time only (penalties not counted in score)

---

## Key Implementation Notes

- Always filter historical data to **competitive matches only** (remove friendlies or weight them at 0.2×) when training outcome models
- Apply **time decay** to historical results — a match from 2019 is less relevant than a match from 2024. Use exponential decay with half-life of ~18 months.
- The **Dixon-Coles correction** parameter (rho) typically fits around -0.13. This adjusts the joint probability of 0-0, 1-0, 0-1, and 1-1 results.
- When predicting knockout matches, the score model should use the **team strength estimates at the time of the tournament**, not historical averages. The Monte Carlo simulation will handle team routing correctly.
- For **UEFA Playoff teams**: treat them as a probability-weighted average of their possible qualifying teams when generating group stage predictions for those specific matches. This propagates uncertainty correctly rather than just picking one team.
- Store all intermediate DataFrames as CSV/parquet in `data/processed/` so you can resume without re-scraping.
- Use `joblib.dump()` to serialise trained models to `models/` directory.

---

## Instructions for Claude Code

When building this project step by step:

1. **Always work in the `worldcup2026/` directory**
2. **Follow the script numbering** — each script is a self-contained step. Run them in order.
3. **Never modify files in `data/raw/`** — always read from raw, write to processed
4. **After each script completes**, print a brief summary of what was produced (row counts, column names, any warnings)
5. **If a data source is unavailable**, use the fallback: FIFA rankings as the sole strength metric and historical goal averages from the Kaggle dataset
6. **When in doubt about a modelling decision**, default to the simpler model first — a well-tuned Poisson regression will outperform a poorly-tuned neural network
7. **The corners model is the highest-priority differentiator** — give it the most feature engineering effort
8. **Test every script** before moving to the next phase
