"""
08_generate_predictions.py
Combines outputs from every model into the final 104-match prediction set.

Group stage (1-72):  uses precomputed goals/corners/cards predictions
Knockout stage (73-104):
  - Predicted matchup from MC sim (pred_home_team, pred_away_team)
  - Re-predicts score for that matchup via Dixon-Coles (neutral venue)
  - Re-predicts corners + yellows + reds for that matchup
  - Winner: from score; if draw, fall back to MC's pred_winner / strength
  - Penalties = True only if MC's p_penalties > 0.5

Run AFTER all upstream scripts (02, 03, 04, 04b, 05, 06, 07).

Outputs:
  output/predictions_final.csv          -- submission file
  data/processed/predictions_combined.csv -- audit copy with helper columns
"""

import math
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import poisson

PROCESSED = Path("data/processed")
EXTERNAL  = Path("data/external")
MODELS    = Path("models")
OUTPUT    = Path("output")
OUTPUT.mkdir(parents=True, exist_ok=True)


def section(t): print(f"\n-- {t} " + "-" * (60 - len(t)))
def ok(name, detail=""): print(f"  [OK]   {name:<26} {detail}")
def warn(name, detail=""): print(f"  [WARN] {name:<26} {detail}")


def altitude_corners_adj(altitude_m):
    if pd.isna(altitude_m): return 1.0
    if altitude_m > 2000: return 0.92
    if altitude_m > 1500: return 0.95
    return 1.0


# Tournament-elevation: WC22 (our training set) averaged 8.94 corners/match,
# but WC18 and WC14 (summer tournaments in NA-like conditions) averaged
# ~10.5/match. WC26 conditions (summer, NA venues) line up with the older
# trend. Apply a small upward bump so our predictions match the historical
# long-run WC corner rate.
CORNERS_TOURNAMENT_BUMP = 0.6


def altitude_cards_adj(altitude_m):
    if pd.isna(altitude_m): return 1.0
    if altitude_m > 2000: return 1.05
    if altitude_m > 1500: return 1.03
    return 1.0


CONF_CARD_MULT = {"CONMEBOL": 1.15, "CAF": 1.10, "CONCACAF": 1.00,
                  "AFC": 0.95, "UEFA": 0.95, "OFC": 1.00}


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


def outcome_probs(P: np.ndarray) -> tuple:
    """Return (P_home_win, P_draw, P_away_win)."""
    p_home = float(np.tril(P, k=-1).sum())
    p_draw = float(np.diag(P).sum())
    p_away = float(np.triu(P, k=1).sum())
    return p_home, p_draw, p_away


def best_score_from_lambdas(lh: float, la: float, rho: float) -> tuple:
    """Score = rounded expected goals (best aggregate calibration).
    Winner probs come from the full Dixon-Coles distribution (independent
    of the rounded score). README explicitly endorses this as one of two
    valid prediction strategies; we pick rounded-expected because it lines
    up with realistic tournament averages (~2.5 goals/match) instead of
    underpredicting via the (0,0)-favouring distribution mode."""
    P = score_distribution(lh, la, rho)
    h = int(round(lh))
    a = int(round(la))
    p_home, p_draw, p_away = outcome_probs(P)
    return h, a, p_home, p_draw, p_away


def predict_score_neutral(home, away, gm):
    """Knockout prediction (neutral venue). Forces a non-draw result since
    knockout matches can't actually end in a tie -- either regulation/ET
    decides it, or pens do (and pens are reported via the penalties flag,
    not via the recorded score). Predicting (1,1) for a knockout while
    setting penalties=False is logically impossible, and (1,1)+pens=True
    over-predicts the real ~12% pens rate. So we force the favoured team
    (higher lambda) to win by 1 goal when rounding produces a tie."""
    team_to_idx = gm["team_to_idx"]
    if home not in team_to_idx or away not in team_to_idx:
        return 1, 0, 0.5, 0.0, 0.5
    hi, ai = team_to_idx[home], team_to_idx[away]
    alpha, attack, defense, rho = gm["alpha"], gm["attack"], gm["defense"], gm["rho"]
    lh = math.exp(alpha + attack[hi] - defense[ai])
    la = math.exp(alpha + attack[ai] - defense[hi])

    h = int(round(lh))
    a = int(round(la))
    if h == a:                       # force decisive
        if lh >= la:
            h += 1
        else:
            a += 1

    P = score_distribution(lh, la, rho)
    p_home, p_draw, p_away = outcome_probs(P)
    return h, a, p_home, p_draw, p_away


def build_feature_lookups(team_features: pd.DataFrame):
    f = team_features.copy()
    for col in ["wc22_avg_corners", "wc22_avg_yellows", "wc22_avg_reds"]:
        if col in f.columns:
            f[col] = f[col].fillna(0.0)
    lookups = {}
    for _, r in f.iterrows():
        lookups[r["team"]] = {
            "wc22_corners":  float(r.get("wc22_avg_corners", 0)),
            "wc22_yellows":  float(r.get("wc22_avg_yellows", 0)),
            "wc22_reds":     float(r.get("wc22_avg_reds", 0)),
            "confederation": r.get("confederation", "UEFA"),
            "fifa_rank":     float(r.get("fifa_rank", 100)),
        }
    return lookups


def predict_corners_for(home, away, altitude_m, team_lookup, corners_bundle):
    model = corners_bundle["model"]
    scaler = corners_bundle["scaler"]
    feature_cols = corners_bundle["feature_cols"]
    h = team_lookup.get(home, {})
    a = team_lookup.get(away, {})
    feats = {
        "sum_avg_corners": h.get("wc22_corners", 0) + a.get("wc22_corners", 0),
        "min_avg_corners": min(h.get("wc22_corners", 0), a.get("wc22_corners", 0)),
        "max_avg_corners": max(h.get("wc22_corners", 0), a.get("wc22_corners", 0)),
        "abs_rank_diff":   abs(h.get("fifa_rank", 100) - a.get("fifa_rank", 100)),
        "avg_rank":        (h.get("fifa_rank", 100) + a.get("fifa_rank", 100)) / 2,
    }
    X = np.array([[feats[c] for c in feature_cols]])
    Xs = scaler.transform(X)
    raw = float(model.predict(Xs)[0])
    return raw * altitude_corners_adj(altitude_m)


def predict_cards_for(home, away, altitude_m, team_lookup, cards_bundle):
    model = cards_bundle["yellow_model"]
    scaler = cards_bundle["yellow_scaler"]
    feature_cols = cards_bundle["yellow_feature_cols"]
    h = team_lookup.get(home, {})
    a = team_lookup.get(away, {})
    h_mult = CONF_CARD_MULT.get(h.get("confederation", "UEFA"), 1.0)
    a_mult = CONF_CARD_MULT.get(a.get("confederation", "UEFA"), 1.0)
    feats = {
        "sum_avg_y":     h.get("wc22_yellows", 0) + a.get("wc22_yellows", 0),
        "abs_rank_diff": abs(h.get("fifa_rank", 100) - a.get("fifa_rank", 100)),
        "conf_mult":     (h_mult + a_mult) / 2,
    }
    X = np.array([[feats[c] for c in feature_cols]])
    Xs = scaler.transform(X)
    raw = float(model.predict(Xs)[0])
    yellow = raw * altitude_cards_adj(altitude_m)
    red_rate = cards_bundle["red_rate_per_match"] * altitude_cards_adj(altitude_m)
    return yellow, red_rate


def main():
    print("=" * 60)
    print("08_generate_predictions.py")
    print("=" * 60)

    section("Loading inputs")
    group_lambdas = pd.read_csv(PROCESSED / "group_lambdas.csv")
    corners_preds = pd.read_csv(PROCESSED / "corners_predictions.csv")
    cards_preds   = pd.read_csv(PROCESSED / "cards_predictions.csv")
    mc_knockout   = pd.read_csv(PROCESSED / "mc_knockout.csv")
    matchup_details_path = PROCESSED / "mc_matchup_details.csv"
    matchup_details = pd.read_csv(matchup_details_path) if matchup_details_path.exists() else None
    fixtures_g    = pd.read_csv(PROCESSED / "fixtures_group.csv")
    fixtures_k    = pd.read_csv(PROCESSED / "fixtures_knockout.csv")
    team_features = pd.read_csv(PROCESSED / "team_features.csv")

    with open(MODELS / "goals_model.pkl", "rb") as f:
        gm = pickle.load(f)
    with open(MODELS / "corners_model.pkl", "rb") as f:
        corners_bundle = pickle.load(f)
    with open(MODELS / "cards_model.pkl", "rb") as f:
        cards_bundle = pickle.load(f)
    ok("inputs", "all loaded")

    team_lookup = build_feature_lookups(team_features)

    section("Building group-stage predictions (1-72)")
    rho = gm["rho"]
    group_rows = []
    for _, r in fixtures_g.iterrows():
        mid = int(r["match_id"])
        g = group_lambdas[group_lambdas["match_id"] == mid].iloc[0]
        c = corners_preds[corners_preds["match_id"] == mid].iloc[0]
        y = cards_preds[cards_preds["match_id"] == mid].iloc[0]

        lh, la = float(g["lambda_home"]), float(g["lambda_away"])
        hs, as_, p_home, p_draw, p_away = best_score_from_lambdas(lh, la, rho)

        # winning_team: independent max-probability prediction (not score-implied)
        if p_home >= p_draw and p_home >= p_away:
            winning = "home"
        elif p_draw >= p_away:
            winning = "draw"
        else:
            winning = "away"

        group_rows.append({
            "match_id":     mid,
            "home_team":    r["home_team"],
            "away_team":    r["away_team"],
            "home_score":   hs,
            "away_score":   as_,
            "corners":      int(round(float(c["pred_total_corners"]) + CORNERS_TOURNAMENT_BUMP)),
            "yellow_cards": int(y["pred_yellow_rounded"]),
            "red_cards":    int(y["pred_red_rounded"]),
            "winning_team": winning,
            "match_winner": np.nan,
            "penalties":    np.nan,
        })
    ok("group_predictions", f"{len(group_rows)} matches")

    section("Reconciling bracket consistency (Final / 3rd-place)")
    # Pull the predicted Final matchup
    final_row = mc_knockout[mc_knockout["round"] == "Final"].iloc[0]
    final_teams = {final_row["pred_home_team"], final_row["pred_away_team"]}
    ok("final_teams", f"{final_row['pred_home_team']} vs {final_row['pred_away_team']}")

    # 3rd-place playoff teams must NOT overlap with Final teams (they're the
    # losers of the semi-finals). If MC's top pick conflicts, swap to the
    # most-common matchup whose teams aren't in the Final.
    overrides = {}
    if matchup_details is not None:
        third_row = mc_knockout[mc_knockout["round"] == "Third-place playoff"].iloc[0]
        third_mid = int(third_row["match_id"])
        if third_row["pred_home_team"] in final_teams or third_row["pred_away_team"] in final_teams:
            candidates = matchup_details[
                (matchup_details["match_id"] == third_mid)
                & ~matchup_details["home_team"].isin(final_teams)
                & ~matchup_details["away_team"].isin(final_teams)
            ].sort_values("count", ascending=False)
            if not candidates.empty:
                top = candidates.iloc[0]
                overrides[third_mid] = (top["home_team"], top["away_team"], float(top["prob"]))
                warn("third_place_conflict",
                     f"original {third_row['pred_home_team']} vs {third_row['pred_away_team']} "
                     f"-> replaced with {top['home_team']} vs {top['away_team']} (p={top['prob']:.3f})")
            else:
                warn("third_place_conflict", "no non-conflicting matchup found")
        else:
            ok("third_place", f"{third_row['pred_home_team']} vs {third_row['pred_away_team']}  (no conflict)")
    else:
        warn("matchup_details", "mc_matchup_details.csv not found -- re-run script 07")

    section("Building knockout-stage predictions (73-104)")
    knockout_lookup = fixtures_k.set_index("match_id")
    ko_rows = []
    for _, r in mc_knockout.iterrows():
        mid = int(r["match_id"])
        if mid in overrides:
            home, away, _ = overrides[mid]
        else:
            home = r["pred_home_team"]
            away = r["pred_away_team"]
        fx = knockout_lookup.loc[mid]
        altitude_m = fx.get("altitude_m", np.nan)

        hs, as_, p_home, p_draw, p_away = predict_score_neutral(home, away, gm)
        corners = round(predict_corners_for(home, away, altitude_m, team_lookup, corners_bundle)
                        + CORNERS_TOURNAMENT_BUMP)
        yellow, red_rate = predict_cards_for(home, away, altitude_m, team_lookup, cards_bundle)
        yellow_r = int(round(yellow))
        red_r    = 1 if red_rate >= 0.5 else 0

        # Knockout match_winner is always home or away (draws decided by pens).
        # Use win probabilities directly (split the draw probability proportionally).
        p_home_eff = p_home + p_draw * (p_home / max(p_home + p_away, 1e-6))
        p_away_eff = p_away + p_draw * (p_away / max(p_home + p_away, 1e-6))
        match_winner = "home" if p_home_eff >= p_away_eff else "away"

        # Logical consistency: if predicted score (after 90+ET) is a draw,
        # penalties MUST be True (knockouts can't end in draws); if decisive,
        # the match was settled in regulation/ET so penalties=False.
        penalties = (hs == as_)

        ko_rows.append({
            "match_id":     mid,
            "home_team":    home,
            "away_team":    away,
            "home_score":   hs,
            "away_score":   as_,
            "corners":      corners,
            "yellow_cards": yellow_r,
            "red_cards":    red_r,
            "winning_team": np.nan,
            "match_winner": match_winner,
            "penalties":    penalties,
        })
    ok("knockout_predictions", f"{len(ko_rows)} matches")

    section("Sanity checks (per README)")
    all_preds = pd.DataFrame(group_rows + ko_rows).sort_values("match_id").reset_index(drop=True)
    all_preds["total_goals"] = all_preds["home_score"] + all_preds["away_score"]

    avg_goals    = all_preds["total_goals"].mean()
    avg_corners  = all_preds["corners"].mean()
    avg_yellow   = all_preds["yellow_cards"].mean()
    avg_red      = all_preds["red_cards"].mean()
    sum_red      = int(all_preds["red_cards"].sum())
    n_matches    = len(all_preds)

    checks = [
        ("avg_goals",     2.2 <= avg_goals    <= 2.8, f"{avg_goals:.2f}  (target 2.2-2.8)"),
        ("avg_corners",   9.0 <= avg_corners  <= 11.5, f"{avg_corners:.2f} (target 9.0-11.5)"),
        ("avg_yellow",    2.8 <= avg_yellow   <= 3.8, f"{avg_yellow:.2f}  (target 2.8-3.8)"),
        ("avg_red",       avg_red < 0.25,             f"{avg_red:.2f}  (target <0.25)"),
        ("sum_red_total", sum_red <= 20,              f"{sum_red}  (target <=20)"),
        ("match_count",   n_matches == 104,           f"{n_matches}  (target 104)"),
    ]
    for name, passed, val in checks:
        (ok if passed else warn)(name, val)

    null_total = int(all_preds[["home_score", "away_score", "corners",
                                "yellow_cards", "red_cards"]].isnull().sum().sum())
    (ok if null_total == 0 else warn)("nulls_in_core", f"{null_total}")

    section("Writing outputs")
    all_preds.to_csv(PROCESSED / "predictions_combined.csv", index=False)
    ok("predictions_combined.csv", f"{len(all_preds)} rows (audit copy)")

    submission_cols = ["match_id", "home_team", "away_team", "home_score",
                       "away_score", "corners", "yellow_cards", "red_cards",
                       "winning_team", "match_winner", "penalties"]
    submission = all_preds[submission_cols]
    submission.to_csv(OUTPUT / "predictions_final.csv", index=False)
    ok("predictions_final.csv", f"-> {OUTPUT / 'predictions_final.csv'}")

    section("Sample group-stage predictions (first 5)")
    print(all_preds.head(5)[[
        "match_id", "home_team", "away_team",
        "home_score", "away_score", "corners",
        "yellow_cards", "red_cards", "winning_team"
    ]].to_string(index=False))

    section("Sample knockout predictions (first 5)")
    print(all_preds[all_preds["match_id"] >= 73].head(5)[
        ["match_id", "home_team", "away_team", "home_score", "away_score",
         "corners", "yellow_cards", "match_winner", "penalties"]
    ].to_string(index=False))

    section("Final + 3rd-place playoff predictions")
    print(all_preds[all_preds["match_id"] >= 103][
        ["match_id", "home_team", "away_team", "home_score", "away_score",
         "corners", "yellow_cards", "match_winner", "penalties"]
    ].to_string(index=False))

    print("\n" + "=" * 60)
    print("Predictions complete.")
    print(f"Submission file: {OUTPUT / 'predictions_final.csv'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
