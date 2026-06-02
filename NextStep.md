# NextStep.md — Status + Remaining Work to Submission

**Last updated**: 2026-06-02 (T-7)
**Submission deadline**: 2026-06-09 (T-1)
**Tournament kickoff**: 2026-06-11

This doc is intentionally focused on **what still needs to happen** between now and submission. The technical model breakdown is in Part 4 at the bottom for reference, but the action items are at the top.

---

## 1. Where we stand right now

### Current predictions (committed `c30f8cc`)

| | |
|---|---|
| Champion | **Spain** |
| Runner-up | **Portugal** |
| 3rd place | **Japan** |
| 4th place | **Germany** |
| avg_goals | 2.70 ✓ (target 2.2-2.8) |
| avg_corners | 9.29 ✓ (target 9.0-11.5) |
| avg_yellow | 3.96 ⚠ above 3.8 target (accepted — recent continentals run hotter than WC22) |
| avg_red | 0.00 ✓ |
| match_count | 104 ✓ |

### What got built since 2026-05-27 (T-13)

- Pooled corners/cards training data (WC22 + Euro 2024 + Copa America 2024 + AFCON 2025/26 = 194 matches)
- Head-to-head wired into Dixon-Coles goals model as a `delta` parameter
- Bookmaker h2h consensus blended 75/25 with model for `winning_team` predictions
- Full top-down bracket reconciler (matchups now consistent with downstream winners)
- 50K Monte Carlo iterations (was 10K) for smoother slot distributions
- 5 Google Calendar reminders for the refresh cadence
- Polymarket scraper fixed (markets moved from `/events` to `/markets` endpoint)
- Date-parsing bug fixed in `02_clean_merge.py` (was silently letting in wrong-decade matches)
- 31 hand-curated pre-WC friendlies folded in (May 26 – June 1)
- `goalscorers.csv` integrated → `team_key_players_enriched.csv` shows recent-form goals per star
- **Odds API now pulls totals + spreads** (was h2h only) → market lambdas blended 50/50 with model for group-stage scoreline prediction. Germany vs Curaçao went 2-1 → 3-1, Brazil vs Haiti 2-1 → 3-0, etc.

### Odds API budget

- **Used: 5 of 500 credits this month** (as of 2026-06-02)
- Each refresh costs 3 credits (h2h + spreads + totals)
- Remaining scheduled refreshes (T-5/T-3/T-2/T-1) = 12 credits max
- Check anytime: `py scripts/check_odds_api_budget.py`

---

## 2. What still needs to happen (in priority order)

### Tier A — Definitely do

#### A1. Keep adding friendlies as they're played (ongoing through T-1)

Source CSV: `data/raw/recent_results_supplemental.csv`. Already has 31 matches from May 26 – June 1. Append new rows as more friendlies happen (June 2-9).

Schema reminder:
```
date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
2026-06-04,France,Italy,2,1,Friendly,Paris,France,FALSE
2026-06-07,Brazil,Croatia,3,0,Friendly,Rio de Janeiro,Brazil,FALSE
```

Rules:
- Tournament = `Friendly` (script 03 weights at 0.2× × time decay)
- `neutral = TRUE` if venue isn't home country of either team
- Use canonical team names (`United States` not `USA`, `Türkiye` not `Turkey`, etc.)
- Skip U-21 / women's / club friendlies — senior men's internationals only

How to use after appending:
```powershell
.\refresh.ps1                    # full refresh including MC sim (~5min)
.\refresh.ps1 -SkipMC            # quick refresh if MC didn't change much (~45sec)
```

#### A2. Manual cards / corners odds collection (replaces the planned referee model)

**Why this matters**: The Odds API doesn't carry corner or yellow card markets for soccer. We were planning to compensate by hand-building a per-referee card-rate table at T-5 (item #6 in the old roadmap). But if you can find bookmaker odds on **total cards** and **total corners** per match, those are sharper than any referee-based model would be — and we use the same blending mechanic that just worked so well for goals.

**Where to find them**: DraftKings, FanDuel, Bovada, Stake, Pinnacle all offer:
- "Match Total Corners" — over/under X.5 (typically 8.5, 9.5, 10.5)
- "Match Total Yellow Cards" — over/under X.5 (typically 3.5, 4.5)

These are anti-bot protected so can't be scraped without effort. You'd manually collect from the bookmaker UI.

**Format I need from you** (drop in `data/raw/manual_cards_corners_odds.csv`):
```
date,home_team,away_team,market,line,over_price,under_price,bookmaker
2026-06-11,Mexico,South Africa,corners,9.5,1.90,1.95,draftkings
2026-06-11,Mexico,South Africa,yellow_cards,4.5,2.10,1.75,draftkings
2026-06-13,Germany,Curaçao,corners,11.5,1.85,1.95,draftkings
2026-06-13,Germany,Curaçao,yellow_cards,3.5,2.05,1.80,draftkings
```

I'll write the loader + blend logic the same day you give me data. **Priority order**: the high-multiplier matches (Round of 16 onwards) matter most per match, but those won't be priced until ~2 weeks after group stage ends — for the submission, focus on **the 8-10 highest-multiplier knockout slots** where you can find lines.

For group-stage matches, only worth collecting for the ones with extreme expectations (Germany-Curaçao corners might be way above 11.5, etc.).

**If you can't find bookmaker lines for cards/corners**: fall back to the referee plan (T-5 calendar event has the details).

#### A3. T-2 final review (June 7)

The June 7 calendar event has the detailed checklist. Walk through all 104 predictions in `output/predictions_final.csv` looking for:
- Implausible scorelines (3+ goal margins between balanced teams)
- `winning_team` that contradicts heavy bookmaker favourites
- Knockout matchups whose teams don't make sense (reconciler edge cases — see limitations below)
- Cross-reference key player absences against `team_key_players_enriched.csv` (the `goals_12mo` column shows who's actually in form)

#### A4. Submit (June 9)

Run `.\refresh.ps1` one last time. Upload `output/predictions_group.csv` + `output/predictions_knockout.csv` to DataLab. **Do not wait until June 10**.

### Tier B — Probably do (lower ROI)

#### B1. Knockout-stage odds when bookmakers price them

Bookmakers typically don't price knockout matches until group stage ends. WC group stage ends ~June 26. That's well after our June 9 submission, so this is **only relevant if we can re-submit** (the rules typically don't allow this).

For our purposes: knockout predictions stay pure-model (Dixon-Coles + h2h covariate + bracket reconciler).

### Tier C — Skip unless I'm wrong about Tier A2

#### C1. Per-referee card rate model

This was originally Tier 1 item #6 in the old roadmap. Plan was to build `data/external/referee_card_rates.csv` from per-referee historical averages (Daniel Siebert ~5.5 cards/match, Stéphanie Frappart ~3.2, etc.) and apply as a multiplier.

**Skip if A2 succeeds**: bookmaker card markets already incorporate the referee's effect (bookies see the same FIFA announcement we would). Manual line collection gives us the same info with less work.

**Do if A2 fails**: 3-4 hour build. Sources for ref stats: Wikipedia per-referee pages, WhoScored. FIFA announces match referees ~T-5 (June 4). The June 4 calendar event has the plan.

---

## 3. The remaining 7 days at a glance

| Date | T-minus | What's scheduled | Manual data to bring |
|---|---|---|---|
| Jun 2 | T-7 (today) | — | ✓ Done: 31 friendlies added |
| Jun 3 | T-6 | — | Optional: any June 2-3 friendlies |
| **Jun 4** | **T-5** | Calendar event fires | Referee assignments (if going Tier C) **OR** start collecting cards/corners odds (Tier A2) |
| Jun 5 | T-4 | — | More friendlies if played |
| **Jun 6** | **T-3** | Calendar event: full refresh + market re-pull | June 4-5 friendlies appended to supplemental CSV |
| **Jun 7** | **T-2** | Calendar event: final manual review | Any last-minute matches |
| Jun 8 | T-1.5 | — | (Buffer day) |
| **Jun 9** | **T-1** | Calendar event: **SUBMIT** | None — final pipeline run only |

---

## 4. Known limitations (still unresolved on 2026-06-02)

These are real but accepted for submission. Documented so we know what NOT to spend more time on:

1. **Same-team-in-two-slots in lower bracket rounds** — When MC's marginal matchup distribution is sparse for a required winner, the reconciler accepts a duplicate. Currently ~3-5 entries affected in R32/R16/QF. Impact bounded by low multipliers (R32 = ×1, R16 = ×2). A fix would require beam-search over MC's joint distribution (2-3hr rewrite).

2. **Yellow card avg ≈ 3.96 above README target 3.8** — Defensible: Euro 2024 and AFCON 25/26 both ran higher than WC22. Historical WCs span 3.2-4.2 per match. Accepted on 2026-05-27.

3. **Knockout matches have no bookmaker odds yet** — Not bookable until group stage plays out. Knockouts use pure model + bracket reconciler. Group stage uses the model+market blend.

4. **1 of 72 group fixtures (US vs Türkiye) missing market lambdas** — Odds API hasn't priced it. Falls back to pure-model lambdas for that one match.

5. **40-team xG coverage but no model integration** — Data extracted in `team_xg_summary.csv` but only 1-7 matches per team and 8/48 teams missing. Used only as a T-2 sanity check, not a model feature.

6. **Goalscorers watchlist 65% match rate** — 62 of 176 watchlist entries don't appear in `goalscorers.csv` (mostly GKs and defenders who legitimately don't score). The 114 that do match (forwards/mids) have real `goals_12mo` data.

7. **Polymarket coverage is 48 of ~200 international teams** — Only the contenders have championship markets. For the 75+ bottom-tier teams, the goals model has no market signal in calibration.

---

## 5. Suggested future improvements (post-WC, for next major tournament)

- **Beam search over MC joint distribution** — fixes the bracket-reconciler duplicate-team issue properly (2-3 hr).
- **Tournament-specific intercept in the cards model** — let Euro/AFCON have their own baseline so pooling doesn't push avg_yellow up.
- **Direct ML for winning_team** (LightGBM / XGBoost) trained directly on (home, away, features) → outcome. Could outperform Dixon-Coles for the 40-pt group-winner field.
- **Cloud-hosted pipeline** for true device-independence (GitHub Actions runs refresh.ps1 on the scheduled dates and pushes back; user just pulls).
- **Streamlit dashboard** for the T-2 review step with one-click override capability.
- **Post-WC error analysis** — walk through every prediction vs actual result, identify the systematic errors, fold lessons into the next major tournament's prior.

---

## 6. Pipeline reference (for context)

### 12 sequential scripts

```
01_collect_data.py                   Polymarket + Odds API (h2h/spreads/totals) + FIFA rankings + Kaggle CSVs
01c_load_match_stats.py              WC 2022 detailed stats -> long format
01d_load_external_match_stats.py     Euro 2024 / Copa 2024 / AFCON 2025/26 -> long format
01e_process_odds.py                  Bookmaker h2h consensus + spreads/totals -> market lambdas
01f_compute_player_form.py           goalscorers.csv aggregation -> watchlist enrichment
02_clean_merge.py                    Normalise team names, resolve playoff placeholders,
                                     load recent_results_supplemental.csv if present
03_feature_engineering.py            Time-decay weights, rolling team form, h2h covariate
04_train_goals_model.py              Dixon-Coles MLE (4 globals + 226 team attack + 226 team defense)
04b_calibrate_goals_model.py         Polymarket Bayesian prior (40/60 model/market in z-space)
05_train_corners_model.py            Poisson regression on 194-match pool
06_train_cards_model.py              Poisson regression on 194-match pool + WC22 red-rate
07_monte_carlo_simulator.py          50,000 tournament simulations
08_generate_predictions.py           Market-blend lambdas + winning_team blend + full bracket reconciler
09_format_output.py                  Split into DataLab submission schemas
```

One-shot runner: `.\refresh.ps1` (full) or `.\refresh.ps1 -SkipMC` (skip the 5-min MC sim) or `.\refresh.ps1 -SkipCollect -SkipMC` (just retrain from current data).

### Key data files

| Path | What | User-managed? |
|---|---|---|
| `data/raw/results.csv` | Kaggle international results 1872-2026 | No (auto) |
| `data/raw/recent_results_supplemental.csv` | Hand-curated recent friendlies | **YES — append** |
| `data/raw/manual_cards_corners_odds.csv` | Bookmaker odds for cards + corners totals | **YES — populate if going Tier A2** |
| `data/raw/external_stats/*.csv` | Euro 2024, Copa 2024, AFCON 2025/26 tournament details | No (one-time) |
| `data/external/team_key_players_enriched.csv` | Watchlist + recent goals per star | Read at T-2 review |
| `data/external/fifa_april_rankings_manual.csv` | April 2026 FIFA rankings (current) | No |
| `output/predictions_group.csv` | DataLab submission (72 group matches) | **The deliverable** |
| `output/predictions_knockout.csv` | DataLab submission (32 knockout matches) | **The deliverable** |

### Model snapshot at 2026-06-02

```
Goals (Dixon-Coles): 226 teams × 2 (attack/defense) + 4 globals (α, γ, ρ, δ)
                     trained on 7,952 post-2018 matches
                     in-sample MAE: home 1.04, away 0.85
                     calibrated against 48-team Polymarket signal at 40/60 model/market

Corners: Poisson regression, 5 features, 194-match training pool
         5-fold CV MAE: 2.70 (WC22-subset) — README target <2.5

Yellow cards: Poisson regression, 3 features, same 194-match pool
              5-fold CV MAE: 1.92 (WC22-subset) — README target <1.0

Red cards: empirical rate (~0.06/match) from WC22, all predictions = 0

Monte Carlo: 50,000 iterations
             Top 10 championship probs (post-calibration):
               Spain, Portugal, Japan, Germany, England, France,
               Argentina, Brazil, Netherlands, Norway
```
