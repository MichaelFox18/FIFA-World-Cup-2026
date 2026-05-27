"""
04b_calibrate_goals_model.py
Calibrates the trained Dixon-Coles model using Polymarket championship
probabilities as a Bayesian prior. Per the README:
  "Use betting markets as a Bayesian prior. Trust them as a baseline..."

For every team that has BOTH a model parameter and a Polymarket win
probability, we blend their data-driven strength with their market-implied
strength via a weighted average on z-scores, then push the adjustment back
into attack/defense (split equally).

The blend weight (BLEND_W_MODEL) controls how much the data-driven model
dominates. 0.5 = equal blend; lower trusts the market more.

Outputs:
  models/goals_model.pkl              -- overwritten with calibrated values
  data/processed/group_lambdas.csv    -- regenerated with calibrated lambdas

Run AFTER scripts 02/03/04 have been run:
  py scripts/04b_calibrate_goals_model.py
"""

import math
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import poisson

PROCESSED = Path("data/processed")
MODELS = Path("models")

# 0.5 = balanced; lower trusts the market more.
# 0.4 = market-leaning: we keep the data edge on corners/cards but defer to
# Polymarket for team strength (which aggregates a lot of soft info our
# results-based fit misses, e.g. tournament psychology, player quality).
BLEND_W_MODEL = 0.4


def section(t): print(f"\n-- {t} " + "-" * (60 - len(t)))
def ok(name, detail=""): print(f"  [OK]   {name:<26} {detail}")
def warn(name, detail=""): print(f"  [WARN] {name:<26} {detail}")


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def score_distribution(lh, la, rho, max_goals=8):
    grid = np.arange(max_goals + 1)
    P = np.outer(poisson.pmf(grid, lh), poisson.pmf(grid, la))
    P[0, 0] *= max(1 - lh * la * rho, 0)
    P[0, 1] *= max(1 + lh * rho, 0)
    P[1, 0] *= max(1 + la * rho, 0)
    P[1, 1] *= max(1 - rho, 0)
    s = P.sum()
    if s > 0:
        P /= s
    return P


def predict_fixture(fx, gm):
    """Return (lambda_home, lambda_away, mode, expected, win/draw/lose) for one fixture."""
    team_to_idx = gm["team_to_idx"]
    h, a = fx["home_team"], fx["away_team"]
    if h not in team_to_idx or a not in team_to_idx:
        return None
    hi, ai = team_to_idx[h], team_to_idx[a]
    alpha, attack, defense, gamma, rho = (gm["alpha"], gm["attack"], gm["defense"],
                                          gm["gamma"], gm["rho"])
    delta_h2h = float(gm.get("delta", 0.0))
    home_is_host = int(fx.get("home_is_host", 0) or 0)
    away_is_host = int(fx.get("away_is_host", 0) or 0)
    g_h = gamma if home_is_host else 0.0
    g_a = gamma if away_is_host else 0.0
    h2h_fx = float(fx.get("h2h_avg_gd", 0.0) or 0.0)
    lh = math.exp(alpha + attack[hi] - defense[ai] + g_h + delta_h2h * h2h_fx)
    la = math.exp(alpha + attack[ai] - defense[hi] + g_a - delta_h2h * h2h_fx)
    P = score_distribution(lh, la, rho)
    mh, ma = np.unravel_index(np.argmax(P), P.shape)
    grid = np.arange(P.shape[0])
    eh = float((P.sum(axis=1) * grid).sum())
    ea = float((P.sum(axis=0) * grid).sum())
    p_home = float(np.tril(P, k=-1).sum())
    p_draw = float(np.diag(P).sum())
    p_away = float(np.triu(P, k=1).sum())
    return lh, la, int(mh), int(ma), eh, ea, p_home, p_draw, p_away


def main():
    print("=" * 60)
    print("04b_calibrate_goals_model.py  |  Polymarket Bayesian prior")
    print("=" * 60)

    section("Loading data")
    with open(MODELS / "goals_model.pkl", "rb") as f:
        gm = pickle.load(f)
    team_features = pd.read_csv(PROCESSED / "team_features.csv")
    fixtures = pd.read_csv(PROCESSED / "fixture_features.csv")
    ok("goals_model", f"{len(gm['teams'])} teams")
    ok("team_features", f"{len(team_features)} WC teams")
    ok("fixtures", f"{len(fixtures)} group matches")

    # Step 1: per-team model strength (attack + defense)
    teams_in_gm = gm["teams"]
    attack = np.asarray(gm["attack"]).astype(float).copy()
    defense = np.asarray(gm["defense"]).astype(float).copy()
    model_strength = attack + defense

    # Step 2: per-team polymarket probability
    market_lookup = dict(zip(team_features["team"], team_features["polymarket_win_prob"]))
    market_p = np.array([market_lookup.get(t, np.nan) for t in teams_in_gm], dtype=float)
    has_market = ~np.isnan(market_p)
    ok("teams_with_polymarket", f"{int(has_market.sum())}/{len(teams_in_gm)}")

    # Step 3: standardize both signals over the subset that has both
    valid_model = model_strength[has_market]
    valid_market_logit = logit(market_p[has_market])

    z_model = (valid_model - valid_model.mean()) / valid_model.std()
    z_market = (valid_market_logit - valid_market_logit.mean()) / valid_market_logit.std()

    # Step 4: blend in z-space, then convert back to model strength scale
    z_blended = BLEND_W_MODEL * z_model + (1 - BLEND_W_MODEL) * z_market
    blended_strength = z_blended * valid_model.std() + valid_model.mean()
    delta = blended_strength - valid_model

    # Step 5: distribute the delta equally to attack and defense
    attack[has_market] = attack[has_market] + delta / 2
    defense[has_market] = defense[has_market] + delta / 2

    section("Calibration deltas (largest absolute changes)")
    df_delta = pd.DataFrame({
        "team":          teams_in_gm,
        "polymarket_p":  market_p,
        "model_str":     model_strength,
        "new_str":       attack + defense,
        "delta":         (attack + defense) - model_strength,
    })
    df_delta_market = df_delta[has_market].assign(
        abs_delta=lambda d: d["delta"].abs()
    ).sort_values("abs_delta", ascending=False).drop(columns="abs_delta")
    print(df_delta_market.head(15).round(3).to_string(index=False))

    section("Sanity: teams whose strength was boosted most")
    boosted = df_delta_market.sort_values("delta", ascending=False).head(8)
    print(boosted.round(3).to_string(index=False))

    section("Sanity: teams whose strength was reduced most")
    reduced = df_delta_market.sort_values("delta", ascending=True).head(8)
    print(reduced.round(3).to_string(index=False))

    # Step 6: save calibrated model (overwrite original; original is regeneratable from 04)
    section("Saving calibrated model")
    gm_cal = dict(gm)
    gm_cal["attack"] = attack
    gm_cal["defense"] = defense
    gm_cal["calibration"] = {
        "blend_weight_model": BLEND_W_MODEL,
        "n_teams_calibrated": int(has_market.sum()),
        "source": "polymarket championship probabilities",
    }
    with open(MODELS / "goals_model.pkl", "wb") as f:
        pickle.dump(gm_cal, f)
    ok("goals_model.pkl", "overwritten with calibrated values")

    # Step 7: re-predict group lambdas with calibrated model
    section("Recomputing group_lambdas.csv with calibrated model")
    rows = []
    missing = []
    for _, fx in fixtures.iterrows():
        result = predict_fixture(fx, gm_cal)
        if result is None:
            missing.append(fx["home_team"] if fx["home_team"] not in gm_cal["team_to_idx"]
                           else fx["away_team"])
            rows.append({
                "match_id": fx["match_id"],
                "home_team": fx["home_team"], "away_team": fx["away_team"],
                "lambda_home": np.nan, "lambda_away": np.nan,
                "mode_home_score": np.nan, "mode_away_score": np.nan,
                "exp_home_score": np.nan, "exp_away_score": np.nan,
                "p_home_win": np.nan, "p_draw": np.nan, "p_away_win": np.nan,
            })
            continue
        lh, la, mh, ma, eh, ea, ph, pd_, pa = result
        rows.append({
            "match_id": fx["match_id"],
            "home_team": fx["home_team"], "away_team": fx["away_team"],
            "lambda_home": lh, "lambda_away": la,
            "mode_home_score": mh, "mode_away_score": ma,
            "exp_home_score": eh, "exp_away_score": ea,
            "p_home_win": ph, "p_draw": pd_, "p_away_win": pa,
        })
    if missing:
        warn("missing_in_model", str(sorted(set(missing))))

    preds = pd.DataFrame(rows)
    preds.to_csv(PROCESSED / "group_lambdas.csv", index=False)
    ok("group_lambdas.csv", f"{len(preds)} fixtures regenerated")

    section("Sample of recalibrated predictions (first 5)")
    print(preds.head(5)[[
        "match_id", "home_team", "away_team",
        "lambda_home", "lambda_away",
        "mode_home_score", "mode_away_score",
        "p_home_win", "p_draw", "p_away_win"
    ]].round(3).to_string(index=False))

    print("\n" + "=" * 60)
    print("Calibration complete.  Now re-run script 07 to refresh the MC sim.")
    print("=" * 60)


if __name__ == "__main__":
    main()
