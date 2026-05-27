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


# Per-source sample weights when pooling tournaments.
# WC22 is the exact tournament type we predict, so it gets full weight.
# Continental tournaments are slightly downweighted (different selection,
# different referee pool, different prep windows).
SOURCE_WEIGHT = {"WC 2022": 1.0}
DEFAULT_EXTERNAL_WEIGHT = 0.8


def load_training_pool():
    """Concatenate WC22 + any external tournament stats present."""
    wc22 = pd.read_csv(PROCESSED / "match_stats_wc2022.csv", parse_dates=["date"])
    # Possession is stored as '42%' string in WC22 -- normalise to numeric here
    if "possession" in wc22.columns and wc22["possession"].dtype == object:
        wc22["possession"] = pd.to_numeric(
            wc22["possession"].astype(str).str.replace("%", "").str.strip(),
            errors="coerce"
        )
    parts = [wc22]
    ext_path = PROCESSED / "match_stats_external.csv"
    if ext_path.exists():
        ext = pd.read_csv(ext_path, parse_dates=["date"])
        # Align to WC22 schema; only the cols we use need exist
        parts.append(ext)
        ok("external_long", f"{len(ext)} rows from {ext_path.name}")
    else:
        warn("external_missing",
             f"{ext_path.name} not found -- run 01d to expand training data")
    combined = pd.concat(parts, ignore_index=True, sort=False)
    combined["source_weight"] = combined["tournament"].map(
        lambda t: SOURCE_WEIGHT.get(t, DEFAULT_EXTERNAL_WEIGHT)
    )
    return combined


def main():
    print("=" * 60)
    print("05_train_corners_model.py  |  Poisson regression")
    print("=" * 60)

    section("Loading training data")
    long = load_training_pool()
    ok("pooled_long", f"{len(long)} team-match rows across "
                      f"{long['tournament'].nunique()} tournament(s)")
    for t, n in long.groupby("tournament").size().sort_values(ascending=False).items():
        ok(f"  {t}", f"{n} team-rows")

    # Drop rows without corners data (external sources may have NaN)
    long = long.dropna(subset=["corners"]).copy()

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
                     away_corners=("corners", "last"),
                     tournament=("tournament", "first"),
                     source_weight=("source_weight", "first"))
                .reset_index())
    one["total_corners"] = one["home_corners"] + one["away_corners"]
    one = one.dropna(subset=["total_corners"]).copy()
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

    # NOTE: tried adding per-team possession (WC22 only, 32 teams covered) as
    # a corners-model feature on 2026-05-27. CV MAE got worse (2.70 -> 2.77),
    # likely because only ~65% of WC26 teams have possession history and the
    # signal is already captured by sum_avg_corners. Feature removed; the
    # per-team possession data is still extracted in match_stats_wc2022.csv
    # for future use.

    one["sum_avg_corners"] = one["home_loo_avg"] + one["away_loo_avg"]
    one["min_avg_corners"] = one[["home_loo_avg", "away_loo_avg"]].min(axis=1)
    one["max_avg_corners"] = one[["home_loo_avg", "away_loo_avg"]].max(axis=1)
    one["abs_rank_diff"] = (one["home_rank"] - one["away_rank"]).abs()
    one["avg_rank"] = (one["home_rank"] + one["away_rank"]) / 2
    feature_cols = ["sum_avg_corners", "min_avg_corners", "max_avg_corners",
                    "abs_rank_diff", "avg_rank"]

    X = one[feature_cols].to_numpy()
    y = one["total_corners"].to_numpy()
    sw = one["source_weight"].to_numpy()
    wc22_mask = (one["tournament"] == "WC 2022").to_numpy()

    section("Training Poisson regression (L2 regularised)")
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    # 5-fold CV (pooled)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_maes, fold_maes_wc22 = [], []
    for tr, te in kf.split(Xs):
        m = PoissonRegressor(alpha=0.3, max_iter=500).fit(
            Xs[tr], y[tr], sample_weight=sw[tr]
        )
        pred_te = m.predict(Xs[te])
        fold_maes.append(float(np.mean(np.abs(pred_te - y[te]))))
        # Restrict the test fold to WC22 rows for honest target-distribution MAE
        wc22_te = wc22_mask[te]
        if wc22_te.any():
            fold_maes_wc22.append(
                float(np.mean(np.abs(pred_te[wc22_te] - y[te][wc22_te])))
            )
    cv_mae = float(np.mean(fold_maes))
    ok("cv_mae_total", f"{cv_mae:.2f}  (pooled, all tournaments)")
    if fold_maes_wc22:
        ok("cv_mae_wc22_only", f"{float(np.mean(fold_maes_wc22)):.2f}  "
                                f"(README target <2.5)")

    # Refit on full data
    model = PoissonRegressor(alpha=0.3, max_iter=500).fit(Xs, y, sample_weight=sw)
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
    # Build per-team pooled corner averages from the same data the model was
    # trained on, so train- and predict-time inputs share a distribution.
    # Fall back to the wc22_avg column from fixture_features for teams that
    # didn't appear in the pool at all.
    pooled_avg_corners = long.groupby("team")["corners"].mean().to_dict()
    fixtures = pd.read_csv(PROCESSED / "fixture_features.csv")
    fx = fixtures.copy()
    fx["home_pool_corners"] = fx["home_team"].map(pooled_avg_corners)
    fx["away_pool_corners"] = fx["away_team"].map(pooled_avg_corners)
    n_home_pool = int(fx["home_pool_corners"].notna().sum())
    n_away_pool = int(fx["away_pool_corners"].notna().sum())
    fx["home_pool_corners"] = fx["home_pool_corners"].fillna(fx["home_wc22_avg_corners"])
    fx["away_pool_corners"] = fx["away_pool_corners"].fillna(fx["away_wc22_avg_corners"])
    ok("pooled_avg_home", f"{n_home_pool}/{len(fx)} fixtures got pooled home avg")
    ok("pooled_avg_away", f"{n_away_pool}/{len(fx)} fixtures got pooled away avg")
    fx["sum_avg_corners"] = fx["home_pool_corners"] + fx["away_pool_corners"]
    fx["min_avg_corners"] = fx[["home_pool_corners", "away_pool_corners"]].min(axis=1)
    fx["max_avg_corners"] = fx[["home_pool_corners", "away_pool_corners"]].max(axis=1)
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
