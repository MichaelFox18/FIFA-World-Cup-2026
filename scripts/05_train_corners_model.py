"""
05_train_corners_model.py
Trains a Poisson regression model for total corners per match using WC 2022
match data (64 matches). Predicts total corners for the 72 group + 32 knockout
WC 2026 fixtures.

The training set is small, so we use only a few highly-meaningful features and
L2 regularisation to avoid overfitting:
  - sum of each team's leave-one-out avg corners
  - |FIFA rank gap| (closer games tend to be more open)

Plus an altitude adjustment applied at prediction time (high-altitude WC
venues have systematically fewer corners due to lower-tempo play).

Run from worldcup2026/:  py scripts/05_train_corners_model.py

Outputs:
  models/corners_model.pkl
  data/processed/corners_predictions.csv
"""

import math
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


def altitude_adj(altitude_m) -> float:
    """High-altitude venues -> systematically fewer corners."""
    if pd.isna(altitude_m):
        return 1.0
    if altitude_m > 2000:
        return 0.92
    if altitude_m > 1500:
        return 0.95
    return 1.0


def main():
    print("=" * 60)
    print("05_train_corners_model.py  |  Poisson regression")
    print("=" * 60)

    section("Loading training data")
    long = pd.read_csv(PROCESSED / "match_stats_wc2022.csv", parse_dates=["date"])
    ok("wc22_long", f"{len(long)} team-match rows from WC 2022")

    # Build unique match_id from (sorted team pair, date)
    long["pair_key"] = long.apply(
        lambda r: tuple(sorted([str(r["team"]), str(r["opponent"])])), axis=1
    )
    long["match_id"] = long.groupby(["pair_key", "date"]).ngroup()

    # Pivot to one row per match for training
    one = (long.groupby(["match_id", "pair_key", "date", "category"])
                .agg(home_team=("team", "first"),
                     away_team=("opponent", "first"),
                     home_corners=("corners", "first"),
                     away_corners=("corners", "last"))
                .reset_index())
    one["total_corners"] = one["home_corners"] + one["away_corners"]
    ok("matches_built", f"{len(one)} matches with total_corners")
    ok("mean_total_corners", f"{one['total_corners'].mean():.2f}  (README expected 9-11.5)")

    # Leave-one-out team averages
    def loo_avg(team: str, exclude_match_id: int) -> float:
        m = (long["team"] == team) & (long["match_id"] != exclude_match_id)
        return float(long.loc[m, "corners"].mean()) if m.any() else np.nan

    section("Computing leave-one-out team averages")
    one["home_loo_avg"] = one.apply(lambda r: loo_avg(r["home_team"], r["match_id"]), axis=1)
    one["away_loo_avg"] = one.apply(lambda r: loo_avg(r["away_team"], r["match_id"]), axis=1)
    one = one.dropna(subset=["home_loo_avg", "away_loo_avg"]).copy()
    ok("loo_built", f"{len(one)} matches with LOO avgs")

    # FIFA rank for each team (most recent)
    team_features = pd.read_csv(PROCESSED / "team_features.csv")
    rank_lookup = dict(zip(team_features["team"], team_features["fifa_rank"]))
    one["home_rank"] = one["home_team"].map(rank_lookup).fillna(100)
    one["away_rank"] = one["away_team"].map(rank_lookup).fillna(100)

    one["sum_avg_corners"] = one["home_loo_avg"] + one["away_loo_avg"]
    one["min_avg_corners"] = one[["home_loo_avg", "away_loo_avg"]].min(axis=1)
    one["max_avg_corners"] = one[["home_loo_avg", "away_loo_avg"]].max(axis=1)
    one["abs_rank_diff"] = (one["home_rank"] - one["away_rank"]).abs()
    one["avg_rank"] = (one["home_rank"] + one["away_rank"]) / 2
    feature_cols = ["sum_avg_corners", "min_avg_corners", "max_avg_corners",
                    "abs_rank_diff", "avg_rank"]

    X = one[feature_cols].to_numpy()
    y = one["total_corners"].to_numpy()

    section("Training Poisson regression (L2 regularised)")
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    # 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_maes = []
    for tr, te in kf.split(Xs):
        m = PoissonRegressor(alpha=0.3, max_iter=500).fit(Xs[tr], y[tr])
        fold_maes.append(float(np.mean(np.abs(m.predict(Xs[te]) - y[te]))))
    cv_mae = float(np.mean(fold_maes))
    ok("cv_mae_total", f"{cv_mae:.2f}  (README target <2.5)")

    # Refit on full data
    model = PoissonRegressor(alpha=0.3, max_iter=500).fit(Xs, y)
    in_mae = float(np.mean(np.abs(model.predict(Xs) - y)))
    ok("in_sample_mae", f"{in_mae:.2f}")
    ok("intercept_log", f"{model.intercept_:+.3f}")
    for n, c in zip(feature_cols, model.coef_):
        ok(f"coef[{n}]", f"{c:+.3f}")

    section("Saving model")
    bundle = {
        "model": model, "scaler": scaler, "feature_cols": feature_cols,
        "cv_mae_total": cv_mae, "in_sample_mae": in_mae,
        "training_matches": len(one),
    }
    with open(MODELS / "corners_model.pkl", "wb") as f:
        pickle.dump(bundle, f)
    ok("corners_model.pkl", "saved")

    section("Predicting WC 2026 group fixtures")
    fixtures = pd.read_csv(PROCESSED / "fixture_features.csv")
    fx = fixtures.copy()
    fx["sum_avg_corners"] = fx["home_wc22_avg_corners"] + fx["away_wc22_avg_corners"]
    fx["min_avg_corners"] = fx[["home_wc22_avg_corners", "away_wc22_avg_corners"]].min(axis=1)
    fx["max_avg_corners"] = fx[["home_wc22_avg_corners", "away_wc22_avg_corners"]].max(axis=1)
    fx["abs_rank_diff"] = (fx["home_fifa_rank"].fillna(100)
                           - fx["away_fifa_rank"].fillna(100)).abs()
    fx["avg_rank"] = (fx["home_fifa_rank"].fillna(100)
                      + fx["away_fifa_rank"].fillna(100)) / 2

    Xfx = scaler.transform(fx[feature_cols].to_numpy())
    fx["pred_total_corners_raw"] = model.predict(Xfx)
    fx["altitude_factor"] = fx["altitude_m"].map(altitude_adj)
    fx["pred_total_corners"] = fx["pred_total_corners_raw"] * fx["altitude_factor"]
    fx["pred_total_corners_rounded"] = fx["pred_total_corners"].round().astype(int)

    out = fx[["match_id", "home_team", "away_team", "altitude_m",
              "sum_avg_corners", "abs_rank_diff",
              "pred_total_corners_raw", "altitude_factor",
              "pred_total_corners", "pred_total_corners_rounded"]]
    out.to_csv(PROCESSED / "corners_predictions.csv", index=False)
    ok("corners_predictions.csv", f"{len(out)} group fixtures")

    section("Distribution of predicted total corners")
    print(out["pred_total_corners"].describe().round(2).to_string())

    section("Sample (first 6 group matches)")
    show = out.head(6)[[
        "match_id", "home_team", "away_team", "altitude_m",
        "pred_total_corners", "pred_total_corners_rounded"
    ]].round(2)
    print(show.to_string(index=False))

    print("\n" + "=" * 60)
    print("Corners model training complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
