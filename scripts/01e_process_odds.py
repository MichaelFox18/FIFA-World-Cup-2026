"""
01e_process_odds.py
Converts the raw Odds API pulls (data/raw/odds_h2h.csv -- now holds h2h +
spreads + totals despite the historical filename) into per-match consensus
data for both the winning_team blend AND the goals-lambda blend.

For each (match, bookmaker) triple in the h2h market:
  1. Convert decimal odds -> raw implied probability (1/price)
  2. Normalise the three outcomes (home/draw/away) so they sum to 1 -- this
     strips the bookmaker's overround (typical ~1.05-1.10 sum before).

For spreads / totals: take the median line across bookmakers (robust to a
single book mispricing or stale line). The spread line equals the expected
home_goals - away_goals; the total line equals the expected home_goals +
away_goals. Together these triangulate market-implied Poisson lambdas.

Output:
  data/processed/odds_consensus.csv
    match_id, home_team, away_team, commence_time, n_bookmakers,
    p_home, p_draw, p_away,
    expected_total, expected_diff,         # from totals & spreads markets
    lambda_market_home, lambda_market_away  # derived: (total +/- diff) / 2
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path("data/raw")
EXTERNAL = Path("data/external")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)


def load_name_map() -> dict:
    """Return {variant: canonical} mapping. Used to normalise Odds-API team
    names ('Turkey', 'Czech Republic', 'Ivory Coast', etc.) to the canonical
    names used in fixtures_group.csv and the rest of the pipeline."""
    nm_path = EXTERNAL / "team_name_map.csv"
    if not nm_path.exists():
        return {}
    nm = pd.read_csv(nm_path)
    return dict(zip(nm["variant"].astype(str).str.strip(),
                    nm["canonical"].astype(str).str.strip()))


def section(t): print(f"\n-- {t} " + "-" * (60 - len(t)))
def ok(name, detail=""): print(f"  [OK]   {name:<26} {detail}")
def warn(name, detail=""): print(f"  [WARN] {name:<26} {detail}")


def main():
    print("=" * 60)
    print("01e_process_odds.py  |  Odds API h2h -> consensus probabilities")
    print("=" * 60)

    src = RAW / "odds_h2h.csv"
    if not src.exists():
        warn("missing", f"{src} not found; run 01_collect_data.py first")
        return

    section("Loading raw odds")
    raw_all = pd.read_csv(src)
    ok("rows", f"{len(raw_all):,}")
    ok("matches", f"{raw_all['match_id'].nunique()} distinct fixtures")
    ok("bookmakers", f"{raw_all['bookmaker'].nunique()} distinct")
    ok("markets", str(raw_all["market"].value_counts().to_dict()))

    # Normalise team names so the Odds-API spellings (Turkey, Czech Republic,
    # Ivory Coast, Bosnia & Herzegovina) match the canonical names used in
    # fixtures_group.csv. Without this, ~14 of 72 fixtures fail to match in
    # the downstream lambda blend.
    name_map = load_name_map()
    if name_map:
        for col in ("home_team", "away_team", "outcome"):
            raw_all[col] = raw_all[col].map(lambda n: name_map.get(str(n).strip(), n))
        ok("name_normalised", f"applied {len(name_map)} variants from team_name_map.csv")

    # Split by market for separate processing
    raw = raw_all[raw_all["market"] == "h2h"].copy()
    spreads_raw = raw_all[raw_all["market"] == "spreads"].copy()
    totals_raw = raw_all[raw_all["market"] == "totals"].copy()

    # Map outcome -> which side it is for this match
    def side_of(row):
        if row["outcome"] == row["home_team"]: return "home"
        if row["outcome"] == row["away_team"]: return "away"
        if str(row["outcome"]).lower() == "draw": return "draw"
        return None

    raw["side"] = raw.apply(side_of, axis=1)
    n_dropped = int(raw["side"].isna().sum())
    if n_dropped:
        warn("unmappable_outcomes", f"{n_dropped} rows with outcome not matching home/away/Draw -- dropped")
    raw = raw.dropna(subset=["side"]).copy()

    raw["implied_p"] = 1.0 / raw["price"]

    section("Per-bookmaker normalisation (strip overround)")
    # Sum implied across the three sides per (match_id, bookmaker)
    overround = raw.groupby(["match_id", "bookmaker"])["implied_p"].sum()
    raw["bookmaker_sum"] = raw.set_index(["match_id", "bookmaker"]).index.map(overround)
    raw["fair_p"] = raw["implied_p"] / raw["bookmaker_sum"]
    avg_overround = float(overround.mean())
    ok("avg_overround", f"{avg_overround:.3f} (1.00 = fair, real books typically 1.05-1.10)")

    section("Cross-bookmaker consensus (median per outcome)")
    consensus = (raw.groupby(["match_id", "home_team", "away_team",
                              "commence_time", "side"])["fair_p"]
                    .median()
                    .reset_index())

    # Pivot side -> wide
    wide = consensus.pivot_table(
        index=["match_id", "home_team", "away_team", "commence_time"],
        columns="side", values="fair_p"
    ).reset_index()
    wide.columns.name = None
    # Force expected column order
    for col in ("home", "draw", "away"):
        if col not in wide.columns:
            wide[col] = np.nan
    wide = wide.rename(columns={"home": "p_home", "draw": "p_draw", "away": "p_away"})

    # Bookmaker count per match
    n_bm = (raw.groupby(["match_id"])["bookmaker"].nunique()
                .rename("n_bookmakers").reset_index())
    out = wide.merge(n_bm, on="match_id", how="left")

    # Some matches may be missing the draw price -> renormalise to 1 over present sides
    s = out[["p_home", "p_draw", "p_away"]].sum(axis=1)
    out["p_home"] = out["p_home"] / s
    out["p_draw"] = out["p_draw"] / s
    out["p_away"] = out["p_away"] / s

    out["commence_time"] = pd.to_datetime(out["commence_time"], errors="coerce")
    ok("matches_with_consensus", f"{len(out)}")
    if len(out):
        ok("median_n_bookmakers", f"{out['n_bookmakers'].median():.0f}")
        ok("probs_check", f"row sums {out[['p_home','p_draw','p_away']].sum(axis=1).mean():.3f}")

    # ---- Spreads: expected goal differential (home - away) ----
    section("Spreads -> expected goal differential")
    if len(spreads_raw):
        # Only the HOME team's spread row tells us the expected diff (negative
        # if home favoured). For each (match, bookmaker), there should be one
        # row per outcome (home + away); we filter to the home row.
        spreads_raw = spreads_raw.copy()
        # Drop missing-point rows (some bookmakers may not offer this market)
        spreads_raw = spreads_raw.dropna(subset=["point"])
        home_spread = spreads_raw[spreads_raw["outcome"] == spreads_raw["home_team"]]
        # Median spread line per match across bookmakers
        # spread.point is from the home team's perspective: -3.5 means home
        # favoured by 3.5. Expected diff (home - away) = -point.
        diff_per_match = (home_spread.groupby("match_id")["point"]
                                      .median()
                                      .rename("expected_diff")) * -1
        n_spr_books = (home_spread.groupby("match_id")["bookmaker"]
                                   .nunique()
                                   .rename("n_spread_books"))
        spreads_df = pd.concat([diff_per_match, n_spr_books], axis=1).reset_index()
        ok("matches_with_spreads", f"{len(spreads_df)}")
        out = out.merge(spreads_df, on="match_id", how="left")
    else:
        out["expected_diff"] = np.nan
        out["n_spread_books"] = 0
        warn("no_spreads", "no spread rows in raw data")

    # ---- Totals: expected total goals ----
    section("Totals -> expected total goals")
    if len(totals_raw):
        totals_raw = totals_raw.copy()
        totals_raw = totals_raw.dropna(subset=["point"])
        # Both Over and Under outcomes share the same point value per
        # (match, bookmaker), so just dedupe and take the line.
        line_per_book = (totals_raw.drop_duplicates(["match_id", "bookmaker"])
                                    .groupby("match_id")["point"]
                                    .median()
                                    .rename("expected_total"))
        n_tot_books = (totals_raw.groupby("match_id")["bookmaker"]
                                  .nunique()
                                  .rename("n_totals_books"))
        totals_df = pd.concat([line_per_book, n_tot_books], axis=1).reset_index()
        ok("matches_with_totals", f"{len(totals_df)}")
        out = out.merge(totals_df, on="match_id", how="left")
    else:
        out["expected_total"] = np.nan
        out["n_totals_books"] = 0
        warn("no_totals", "no totals rows in raw data")

    # ---- Derive market-implied Poisson lambdas ----
    # When BOTH expected_total and expected_diff are present:
    #   lambda_home = (total + diff) / 2
    #   lambda_away = (total - diff) / 2
    # When only h2h is available, leave the market lambdas NaN -- script 08
    # then falls back to the model's lambdas alone.
    section("Deriving market-implied lambdas")
    have_both = out["expected_total"].notna() & out["expected_diff"].notna()
    out["lambda_market_home"] = np.where(
        have_both, (out["expected_total"] + out["expected_diff"]) / 2, np.nan
    )
    out["lambda_market_away"] = np.where(
        have_both, (out["expected_total"] - out["expected_diff"]) / 2, np.nan
    )
    # Guard against negative lambdas (shouldn't happen if the data is sane)
    out["lambda_market_home"] = out["lambda_market_home"].clip(lower=0.05)
    out["lambda_market_away"] = out["lambda_market_away"].clip(lower=0.05)
    ok("matches_with_market_lambdas", f"{int(have_both.sum())} of {len(out)}")
    if have_both.any():
        sub = out.loc[have_both, ["lambda_market_home", "lambda_market_away"]]
        ok("avg_lambda_home_market", f"{sub['lambda_market_home'].mean():.2f}")
        ok("avg_lambda_away_market", f"{sub['lambda_market_away'].mean():.2f}")

    out_path = PROCESSED / "odds_consensus.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved {len(out)} consensus rows -> {out_path}")

    if len(out):
        section("Sample (10 strongest favourites)")
        out["max_p"] = out[["p_home", "p_draw", "p_away"]].max(axis=1)
        show = out.sort_values("max_p", ascending=False).head(10)[
            ["home_team", "away_team", "n_bookmakers", "p_home", "p_draw", "p_away"]
        ].round(3)
        print(show.to_string(index=False))

    print("=" * 60)


if __name__ == "__main__":
    main()
