"""
06_train_cards_model.py
Trains TWO models from WC 2022 card data (64 matches):
  - Yellow cards: Poisson regression on total yellows per match
  - Red cards: low-rate empirical Poisson (default 0 per match)

Run from worldcup2026/:  py scripts/06_train_cards_model.py

Outputs:
  models/cards_model.pkl
  data/processed/cards_predictions.csv
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

PROCESSED = Path("data/processed")
MODELS = Path("models")
MODELS.mkdir(exist_ok=True)


def section(t): print(f"\n-- {t} " + "-" * (60 - len(t)))
def ok(name, detail=""): print(f"  [OK]   {name:<26} {detail}")
def warn(name, detail=""): print(f"  [WARN] {name:<26} {detail}")


# Confederation card-rate multipliers (per README: SA & African teams higher)
CONF_CARD_MULT = {
    "CONMEBOL": 1.15,
    "CAF":      1.10,
    "CONCACAF": 1.00,
    "AFC":      0.95,
    "UEFA":     0.95,
    "OFC":      1.00,
}


def altitude_card_adj(altitude_m) -> float:
    """More physical/aerial play at altitude -> slightly more cards."""
    if pd.isna(altitude_m):
        return 1.0
    if altitude_m > 2000:
        return 1.05
    if altitude_m > 1500:
        return 1.03
    return 1.0


def main():
    print("=" * 60)
    print("06_train_cards_model.py  |  Yellow + Red")
    print("=" * 60)

    section("Loading training data")
    long = pd.read_csv(PROCESSED / "match_stats_wc2022.csv", parse_dates=["date"])
    ok("wc22_long", f"{len(long)} team-match rows")

    long["pair_key"] = long.apply(
        lambda r: tuple(sorted([str(r["team"]), str(r["opponent"])])), axis=1
    )
    long["match_id"] = long.groupby(["pair_key", "date"]).ngroup()

    one = (long.groupby(["match_id", "pair_key", "date", "category"])
                .agg(home_team=("team", "first"),
                     away_team=("opponent", "first"),
                     home_yellow=("yellow cards", "first"),
                     away_yellow=("yellow cards", "last"),
                     home_red=("red cards", "first"),
                     away_red=("red cards", "last"))
                .reset_index())
    one["total_yellow"] = one["home_yellow"] + one["away_yellow"]
    one["total_red"] = one["home_red"] + one["away_red"]
    ok("matches", f"{len(one)} matches built")
    ok("mean_yellow", f"{one['total_yellow'].mean():.2f}  (README target 2.8-3.8)")
    ok("mean_red", f"{one['total_red'].mean():.3f}  (README target <0.25 avg)")

    def loo_yellow(team, exclude_id):
        m = (long["team"] == team) & (long["match_id"] != exclude_id)
        return float(long.loc[m, "yellow cards"].mean()) if m.any() else np.nan

    section("Computing LOO yellow averages")
    one["home_loo_y"] = one.apply(lambda r: loo_yellow(r["home_team"], r["match_id"]), axis=1)
    one["away_loo_y"] = one.apply(lambda r: loo_yellow(r["away_team"], r["match_id"]), axis=1)
    one = one.dropna(subset=["home_loo_y", "away_loo_y"]).copy()
    ok("loo_built", f"{len(one)} matches")

    team_features = pd.read_csv(PROCESSED / "team_features.csv")
    conf_map = dict(zip(team_features["team"], team_features["confederation"]))
    rank_map = dict(zip(team_features["team"], team_features["fifa_rank"]))

    one["home_conf"] = one["home_team"].map(conf_map)
    one["away_conf"] = one["away_team"].map(conf_map)
    one["home_rank"] = one["home_team"].map(rank_map).fillna(100)
    one["away_rank"] = one["away_team"].map(rank_map).fillna(100)

    one["home_card_mult"] = one["home_conf"].map(CONF_CARD_MULT).fillna(1.0)
    one["away_card_mult"] = one["away_conf"].map(CONF_CARD_MULT).fillna(1.0)
    one["conf_mult"] = (one["home_card_mult"] + one["away_card_mult"]) / 2

    one["sum_avg_y"] = one["home_loo_y"] + one["away_loo_y"]
    one["abs_rank_diff"] = (one["home_rank"] - one["away_rank"]).abs()
    feature_cols = ["sum_avg_y", "abs_rank_diff", "conf_mult"]

    X = one[feature_cols].to_numpy()
    y = one["total_yellow"].to_numpy()

    section("Training yellow-card Poisson regression")
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_maes = []
    for tr, te in kf.split(Xs):
        m = PoissonRegressor(alpha=0.5, max_iter=500).fit(Xs[tr], y[tr])
        fold_maes.append(float(np.mean(np.abs(m.predict(Xs[te]) - y[te]))))
    cv_mae = float(np.mean(fold_maes))
    ok("cv_mae_yellow", f"{cv_mae:.2f}  (README target <1.0)")

    yellow_model = PoissonRegressor(alpha=0.5, max_iter=500).fit(Xs, y)
    in_mae = float(np.mean(np.abs(yellow_model.predict(Xs) - y)))
    ok("in_sample_mae", f"{in_mae:.2f}")
    ok("intercept", f"{yellow_model.intercept_:+.3f}")
    for n, c in zip(feature_cols, yellow_model.coef_):
        ok(f"coef[{n}]", f"{c:+.3f}")

    section("Red cards: empirical rate model")
    red_rate = float(one["total_red"].mean())
    ok("global_red_rate", f"{red_rate:.3f} per match  (default prediction = 0)")

    section("Saving models")
    bundle = {
        "yellow_model": yellow_model,
        "yellow_scaler": scaler,
        "yellow_feature_cols": feature_cols,
        "yellow_cv_mae": cv_mae,
        "yellow_in_sample_mae": in_mae,
        "red_rate_per_match": red_rate,
        "confederation_mult": CONF_CARD_MULT,
        "training_matches": len(one),
    }
    with open(MODELS / "cards_model.pkl", "wb") as f:
        pickle.dump(bundle, f)
    ok("cards_model.pkl", "saved")

    section("Predicting WC 2026 fixtures")
    fixtures = pd.read_csv(PROCESSED / "fixture_features.csv")
    fx = fixtures.copy()

    fx["home_conf"] = fx["home_team"].map(conf_map)
    fx["away_conf"] = fx["away_team"].map(conf_map)
    fx["home_card_mult"] = fx["home_conf"].map(CONF_CARD_MULT).fillna(1.0)
    fx["away_card_mult"] = fx["away_conf"].map(CONF_CARD_MULT).fillna(1.0)
    fx["conf_mult"] = (fx["home_card_mult"] + fx["away_card_mult"]) / 2

    fx["sum_avg_y"] = fx["home_wc22_avg_yellows"] + fx["away_wc22_avg_yellows"]
    fx["abs_rank_diff"] = (fx["home_fifa_rank"].fillna(100)
                           - fx["away_fifa_rank"].fillna(100)).abs()

    Xfx = scaler.transform(fx[feature_cols].to_numpy())
    fx["pred_yellow_raw"] = yellow_model.predict(Xfx)
    fx["altitude_card_factor"] = fx["altitude_m"].map(altitude_card_adj)
    fx["pred_yellow"] = fx["pred_yellow_raw"] * fx["altitude_card_factor"]
    fx["pred_yellow_rounded"] = fx["pred_yellow"].round().astype(int)

    fx["pred_red_lambda"] = red_rate * fx["altitude_card_factor"]
    fx["pred_red_rounded"] = (fx["pred_red_lambda"] >= 0.5).astype(int)

    out = fx[["match_id", "home_team", "away_team", "altitude_m",
              "sum_avg_y", "conf_mult", "abs_rank_diff",
              "pred_yellow", "pred_yellow_rounded",
              "pred_red_lambda", "pred_red_rounded"]]
    out.to_csv(PROCESSED / "cards_predictions.csv", index=False)
    ok("cards_predictions.csv", f"{len(out)} group fixtures")

    section("Distribution of predicted yellow cards")
    print(out["pred_yellow"].describe().round(2).to_string())

    section("Red card sanity")
    n_pred_reds = int(out["pred_red_rounded"].sum())
    ok("matches_with_red", f"{n_pred_reds}/72  (sanity total reds across 104 <=20)")
    ok("avg_pred_red_lambda", f"{out['pred_red_lambda'].mean():.3f}")

    section("Sample (first 6 group matches)")
    show = out.head(6)[[
        "match_id", "home_team", "away_team", "altitude_m",
        "pred_yellow", "pred_yellow_rounded", "pred_red_rounded"
    ]].round(2)
    print(show.to_string(index=False))

    print("\n" + "=" * 60)
    print("Cards model training complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
