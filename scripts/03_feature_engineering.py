"""
03_feature_engineering.py
Builds match-level feature tables for model training and prediction.

Run from worldcup2026/:  py scripts/03_feature_engineering.py

Outputs:
  data/processed/match_features.csv          -- historical matches w/ leakage-free features
  data/processed/fixture_features.csv        -- 72 group matches w/ prediction features
  data/processed/fixture_features_knockout.csv -- 32 knockout slots (teams TBD)
  data/processed/confederation_baselines.csv -- corner/card averages by confederation
"""

import math
import numpy as np
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
EXTERNAL = Path("data/external")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)


def section(t): print(f"\n-- {t} " + "-" * (60 - len(t)))
def ok(name, detail=""): print(f"  [OK]   {name:<28} {detail}")
def warn(name, detail=""): print(f"  [WARN] {name:<28} {detail}")

# Time-decay half life: 18 months per README
DECAY_HALFLIFE_DAYS = 18 * 30.4  # ~547 days
# Friendly downweight per README
FRIENDLY_WEIGHT = 0.2


# -- Rolling time-aware team features ----------------------------------------

def add_rolling_team_features(matches: pd.DataFrame) -> pd.DataFrame:
    """For each match attach each team's rolling stats over their last N
    matches BEFORE that date. Leakage-free via shift(1) + rolling."""
    home = matches[["date", "home_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "home_score": "gf", "away_score": "ga"}
    ).copy()
    home["match_idx"] = matches.index
    home["was_home"] = True

    away = matches[["date", "away_team", "away_score", "home_score"]].rename(
        columns={"away_team": "team", "away_score": "gf", "home_score": "ga"}
    ).copy()
    away["match_idx"] = matches.index
    away["was_home"] = False

    long = pd.concat([home, away], ignore_index=True).sort_values(["team", "date"])
    long["result"] = np.where(long["gf"] > long["ga"], 1,
                       np.where(long["gf"] < long["ga"], -1, 0))

    g = long.groupby("team", sort=False)
    for w in (5, 10, 20):
        long[f"avg_gf_{w}"] = g["gf"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
        long[f"avg_ga_{w}"] = g["ga"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
    long["win_rate_10"] = g["result"].transform(
        lambda s: (s.shift(1) == 1).rolling(10, min_periods=1).mean()
    )
    long["draw_rate_10"] = g["result"].transform(
        lambda s: (s.shift(1) == 0).rolling(10, min_periods=1).mean()
    )
    long["days_rest"] = g["date"].transform(
        lambda s: (s - s.shift(1)).dt.days
    ).clip(upper=365)

    feature_cols = [c for c in long.columns
                    if c.startswith(("avg_gf_", "avg_ga_", "win_rate_", "draw_rate_"))
                    or c == "days_rest"]

    home_feats = (long[long["was_home"]]
                  .set_index("match_idx")[feature_cols]
                  .add_prefix("home_"))
    away_feats = (long[~long["was_home"]]
                  .set_index("match_idx")[feature_cols]
                  .add_prefix("away_"))

    return matches.join(home_feats).join(away_feats)


# -- FIFA rank at match date (time-aware) ------------------------------------

def add_fifa_rank_at_date(matches: pd.DataFrame, fifa: pd.DataFrame) -> pd.DataFrame:
    fifa = fifa[["rank_date", "country_full", "rank"]].dropna().copy()
    fifa["rank_date"] = pd.to_datetime(fifa["rank_date"], format="mixed", errors="coerce")
    fifa = fifa.dropna(subset=["rank_date"])
    fifa["team"] = fifa["country_full"].astype("object")
    fifa = fifa[["rank_date", "team", "rank"]].sort_values("rank_date")

    m = matches.copy()
    m["date"] = pd.to_datetime(m["date"])
    m["__row__"] = range(len(m))

    def lookup(side: str) -> pd.Series:
        df = (m[["date", f"{side}_team", "__row__"]]
              .rename(columns={f"{side}_team": "team"})
              .copy())
        df["team"] = df["team"].astype("object")
        df = df.sort_values("date")
        merged = pd.merge_asof(
            df, fifa, left_on="date", right_on="rank_date",
            by="team", direction="backward"
        )
        return merged.set_index("__row__")["rank"]

    m["home_fifa_rank"] = lookup("home")
    m["away_fifa_rank"] = lookup("away")
    return m.drop(columns="__row__")


# -- Head-to-head (cumulative, time-aware) -----------------------------------

# Per-match GD is clipped to keep blowouts from dominating. A consistent 3-goal
# h2h edge over many meetings is already a very strong signal.
H2H_AVG_CLIP = 3.0


def add_h2h(matches: pd.DataFrame) -> pd.DataFrame:
    m = matches.sort_values("date").copy()
    m["pair"] = m.apply(lambda r: tuple(sorted([r["home_team"], r["away_team"]])), axis=1)
    m["gd_first"] = np.where(
        m["home_team"] == m["pair"].str[0],
        m["home_score"] - m["away_score"],
        m["away_score"] - m["home_score"]
    )
    g = m.groupby("pair", sort=False)
    m["h2h_played"] = g.cumcount()
    m["h2h_cum_gd_first"] = g["gd_first"].cumsum().shift(1).fillna(0)
    first_is_home = (m["home_team"] == m["pair"].str[0])
    m["h2h_home_gd"] = np.where(first_is_home, m["h2h_cum_gd_first"], -m["h2h_cum_gd_first"])
    # Per-prior-meeting average -- consistent across training (leakage-free
    # via cumsum().shift(1)) and prediction (full historical lookup).
    m["h2h_avg_gd"] = (m["h2h_home_gd"] / m["h2h_played"].clip(lower=1)).clip(
        lower=-H2H_AVG_CLIP, upper=H2H_AVG_CLIP
    )
    return m.drop(columns=["pair", "gd_first", "h2h_cum_gd_first"]).sort_index()


def compute_fixture_h2h(fixtures: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """For each prediction fixture, look up cumulative h2h goal-diff favoring
    the home team across all historical meetings."""
    hist = matches[["home_team", "away_team", "home_score", "away_score"]].copy()
    hist["pair"] = hist.apply(
        lambda r: tuple(sorted([r["home_team"], r["away_team"]])), axis=1
    )
    hist["gd_first"] = np.where(
        hist["home_team"] == hist["pair"].str[0],
        hist["home_score"] - hist["away_score"],
        hist["away_score"] - hist["home_score"]
    )
    agg = hist.groupby("pair", sort=False).agg(
        h2h_played=("gd_first", "size"),
        cum_gd_first=("gd_first", "sum")
    )

    out = fixtures.copy()
    pair_keys = out.apply(
        lambda r: tuple(sorted([r["home_team"], r["away_team"]])), axis=1
    )
    out["h2h_played"]      = pair_keys.map(agg["h2h_played"]).fillna(0).astype(int)
    cum_gd_first           = pair_keys.map(agg["cum_gd_first"]).fillna(0)
    first_is_home          = out["home_team"] == pair_keys.str[0]
    out["h2h_home_gd"]     = np.where(first_is_home, cum_gd_first, -cum_gd_first)
    out["h2h_avg_gd"]      = (out["h2h_home_gd"] /
                              out["h2h_played"].clip(lower=1)).clip(
                                  lower=-H2H_AVG_CLIP, upper=H2H_AVG_CLIP)
    return out


# -- Confederation baselines (fallback for teams without WC22 data) ----------

def confederation_baselines(team_features: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in team_features.columns if c.startswith("wc22_")]
    base = (team_features.dropna(subset=cols, how="all")
                         .groupby("confederation")[cols]
                         .mean()
                         .reset_index())
    global_row = team_features[cols].mean().to_frame().T
    global_row["confederation"] = "_GLOBAL"
    return pd.concat([base, global_row], ignore_index=True)


def apply_confederation_fallback(tf: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    # Only fill *averages* -- not counts like wc22_matches
    wc_cols = [c for c in tf.columns if c.startswith("wc22_avg_")]
    tf = tf.copy()
    # Strip whitespace from join keys on both sides
    tf["confederation"] = tf["confederation"].astype(str).str.strip()
    baselines = baselines.copy()
    baselines["confederation"] = baselines["confederation"].astype(str).str.strip()
    b = baselines.set_index("confederation")

    for col in wc_cols:
        if col not in b.columns:
            continue
        # 1) confederation baseline
        conf_fill = tf["confederation"].map(b[col])
        tf[col] = tf[col].fillna(conf_fill)
        # 2) global fallback
        if "_GLOBAL" in b.index:
            tf[col] = tf[col].fillna(b.loc["_GLOBAL", col])
        # 3) hard zero fallback (guarantees no NaN survives)
        tf[col] = tf[col].fillna(0.0)
    return tf


# -- Build fixture feature table for prediction ------------------------------

def build_fixture_features(fixtures: pd.DataFrame, team_features: pd.DataFrame) -> pd.DataFrame:
    """Attach home/away team features and add pairwise diff features."""
    # Build home view: rename "team" -> "home_team", prefix everything else with "home_"
    home = team_features.copy()
    home.columns = ["home_team" if c == "team" else f"home_{c}" for c in home.columns]
    away = team_features.copy()
    away.columns = ["away_team" if c == "team" else f"away_{c}" for c in away.columns]

    out = (fixtures
           .merge(home, on="home_team", how="left")
           .merge(away, on="away_team", how="left"))

    # Pairwise diff features (positive = home advantage)
    out["rank_diff"]      = out["away_fifa_rank"].fillna(200) - out["home_fifa_rank"].fillna(200)
    out["points_diff"]    = out["home_fifa_points"].fillna(0) - out["away_fifa_points"].fillna(0)
    out["pm_prob_diff"]   = out["home_polymarket_win_prob"].fillna(0) - out["away_polymarket_win_prob"].fillna(0)
    out["form_gf_diff"]   = out["home_avg_goals_for"]     - out["away_avg_goals_for"]
    out["form_ga_diff"]   = out["home_avg_goals_against"] - out["away_avg_goals_against"]
    out["win_rate_diff"]  = out["home_win_rate"]          - out["away_win_rate"]
    out["expected_corners"] = out["home_wc22_avg_corners"] + out["away_wc22_avg_corners"]
    out["expected_yellows"] = out["home_wc22_avg_yellows"] + out["away_wc22_avg_yellows"]
    out["expected_reds"]    = out["home_wc22_avg_reds"]    + out["away_wc22_avg_reds"]

    if "altitude_m" in out.columns:
        out["high_altitude"]      = (out["altitude_m"] > 500).astype(int)
        out["very_high_altitude"] = (out["altitude_m"] > 1500).astype(int)

    return out


# -- Main ---------------------------------------------------------------------

def main():
    print("=" * 60)
    print("03_feature_engineering.py")
    print("=" * 60)

    section("Loading processed data")
    matches = pd.read_csv(PROCESSED / "matches_clean.csv", parse_dates=["date"])
    team_features = pd.read_csv(PROCESSED / "team_features.csv")
    fixtures_g = pd.read_csv(PROCESSED / "fixtures_group.csv")
    fixtures_k = pd.read_csv(PROCESSED / "fixtures_knockout.csv")
    fifa = pd.read_csv(RAW / "fifa_ranking_combined.csv", parse_dates=["rank_date"])
    ok("inputs", f"{len(matches):,} matches, {len(team_features)} teams, "
                 f"{len(fixtures_g)} group fixtures, {len(fixtures_k)} knockout slots")

    section("Confederation baselines + fallback fill")
    baselines = confederation_baselines(team_features)
    baselines.to_csv(PROCESSED / "confederation_baselines.csv", index=False)
    ok("baselines_saved", f"confederation_baselines.csv ({len(baselines)} rows)")
    team_features = apply_confederation_fallback(team_features, baselines)
    wc_cols = [c for c in team_features.columns if c.startswith("wc22_avg_")]
    remaining_na = team_features[wc_cols].isna().sum().sum()
    ok("fallback_filled", f"{remaining_na} remaining NaNs in wc22_avg_* (should be 0)")

    section("Rolling time-aware team features (training set)")
    matches = add_rolling_team_features(matches)
    ok("rolling", "home_/away_ avg_gf, avg_ga, win_rate, draw_rate, days_rest")

    section("FIFA rank at match date (merge_asof)")
    matches = add_fifa_rank_at_date(matches, fifa)
    n_with_rank = matches["home_fifa_rank"].notna().sum()
    ok("fifa_rank", f"{n_with_rank:,}/{len(matches):,} matches have home rank")

    section("Head-to-head (cumulative, leakage-free)")
    matches = add_h2h(matches)
    ok("h2h", "h2h_played + h2h_home_gd")

    section("Time decay + sample weights")
    today = pd.Timestamp.today().normalize()
    age_days = (today - matches["date"]).dt.days.clip(lower=0)
    matches["time_weight"] = np.exp(-age_days / DECAY_HALFLIFE_DAYS * math.log(2))
    matches["sample_weight"] = matches["time_weight"] * np.where(
        matches["is_competitive"], 1.0, FRIENDLY_WEIGHT
    )
    ok("weights", f"avg time_weight={matches['time_weight'].mean():.3f}, "
                  f"avg sample_weight={matches['sample_weight'].mean():.3f}")

    section("Pairwise diff features (training set)")
    matches["rank_diff"] = matches["away_fifa_rank"] - matches["home_fifa_rank"]
    matches["form_gf_diff"] = matches["home_avg_gf_10"] - matches["away_avg_gf_10"]
    matches["form_ga_diff"] = matches["home_avg_ga_10"] - matches["away_avg_ga_10"]
    matches["win_rate_diff"] = matches["home_win_rate_10"] - matches["away_win_rate_10"]
    ok("pairwise", "rank_diff, form_gf_diff, form_ga_diff, win_rate_diff")

    section("Building fixture feature table (prediction set)")
    group_features = build_fixture_features(fixtures_g, team_features)
    group_features["round"] = "Group"
    group_features["multiplier"] = 1
    # Historical h2h between the two teams (full record, not leakage-restricted
    # since this is future prediction).
    group_features = compute_fixture_h2h(group_features, matches)
    ok("h2h_in_fixtures",
       f"mean h2h_played={group_features['h2h_played'].mean():.1f}, "
       f"non-zero h2h: {int((group_features['h2h_played'] > 0).sum())}/"
       f"{len(group_features)}")
    ok("fixtures_group", f"{len(group_features)} matches, {len(group_features.columns)} cols")

    knock_features = fixtures_k.copy()
    if "altitude_m" in knock_features.columns:
        knock_features["high_altitude"] = (knock_features["altitude_m"] > 500).astype(int)
        knock_features["very_high_altitude"] = (knock_features["altitude_m"] > 1500).astype(int)
    ok("fixtures_knockout", f"{len(knock_features)} slots (teams TBD via MC sim)")

    section("Quick sanity on fixture features")
    sample = group_features.head(3)[[
        "match_id", "home_team", "away_team",
        "home_fifa_rank", "away_fifa_rank", "rank_diff",
        "expected_corners", "expected_yellows",
        "altitude_m", "home_is_host"
    ]]
    print(sample.to_string(index=False))

    section("Writing outputs")
    matches.to_csv(PROCESSED / "match_features.csv", index=False)
    ok("match_features.csv", f"{len(matches):,} rows × {len(matches.columns)} cols")

    group_features.to_csv(PROCESSED / "fixture_features.csv", index=False)
    ok("fixture_features.csv", f"{len(group_features)} group × {len(group_features.columns)} cols")

    knock_features.to_csv(PROCESSED / "fixture_features_knockout.csv", index=False)
    ok("fixture_features_knockout.csv", f"{len(knock_features)} slots × {len(knock_features.columns)} cols")

    print("\n" + "=" * 60)
    print("Feature engineering complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
