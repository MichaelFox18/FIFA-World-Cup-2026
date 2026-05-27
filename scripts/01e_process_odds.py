"""
01e_process_odds.py
Converts the raw Odds API h2h pulls (data/raw/odds_h2h.csv) into a per-match
consensus probability table.

For each (match, bookmaker) triple:
  1. Convert decimal odds -> raw implied probability (1/price)
  2. Normalise the three outcomes (home/draw/away) so they sum to 1 -- this
     strips the bookmaker's overround (typical ~1.05-1.10 sum before).

Across bookmakers we take the median per outcome (robust to a single book
mispricing). Output:

  data/processed/odds_consensus.csv
    match_id, home_team, away_team, commence_time, n_bookmakers,
    p_home, p_draw, p_away
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)


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
    raw = pd.read_csv(src)
    ok("rows", f"{len(raw):,}")
    ok("matches", f"{raw['match_id'].nunique()} distinct fixtures")
    ok("bookmakers", f"{raw['bookmaker'].nunique()} distinct")

    # Only h2h market
    raw = raw[raw["market"] == "h2h"].copy()

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
