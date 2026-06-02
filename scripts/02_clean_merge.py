"""
02_clean_merge.py
Cleans raw data, normalises team names, resolves UEFA playoff placeholders,
parses market signals, and builds per-team feature aggregates.
Run from worldcup2026/:  py scripts/02_clean_merge.py
"""

import re
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
EXTERNAL = Path("data/external")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)


def section(title: str) -> None:
    print(f"\n-- {title} " + "-" * (60 - len(title)))


def ok(name: str, detail: str = "") -> None:
    print(f"  [OK]   {name:<24} {detail}")


def warn(name: str, detail: str = "") -> None:
    print(f"  [WARN] {name:<24} {detail}")


# -- Team name normalisation --------------------------------------------------

def load_name_map() -> dict:
    df = pd.read_csv(EXTERNAL / "team_name_map.csv")
    return dict(zip(df["variant"].str.strip(), df["canonical"].str.strip()))


def normalise(name, mapping: dict):
    if pd.isna(name):
        return name
    s = str(name).strip()
    return mapping.get(s, s)


# -- Match competitiveness ----------------------------------------------------

COMPETITIVE_PATTERNS = [
    r"FIFA World Cup",
    r"UEFA Euro",
    r"UEFA Nations League",
    r"Copa Am(é|e)rica",
    r"African Cup of Nations",
    r"Africa Cup of Nations",
    r"AFC Asian Cup",
    r"CONCACAF Gold Cup",
    r"CONCACAF Nations League",
    r"OFC Nations Cup",
    r"Confederations Cup",
    r"qualification",
]
_COMP_RE = re.compile("|".join(COMPETITIVE_PATTERNS), re.IGNORECASE)


def is_competitive(tournament: str) -> bool:
    return bool(_COMP_RE.search(str(tournament)))


# -- Polymarket parsing -------------------------------------------------------

_PM_TEAM_RE = re.compile(r"Will\s+(.+?)\s+win", re.IGNORECASE)


def extract_pm_team(market_question: str):
    m = _PM_TEAM_RE.match(str(market_question))
    return m.group(1).strip() if m else None


# -- Venue parsing ------------------------------------------------------------

def split_venue(venue_str: str):
    """'Estadio Azteca, Mexico City' -> ('Estadio Azteca', 'Mexico City')."""
    if pd.isna(venue_str) or "," not in str(venue_str):
        return None, None
    parts = [p.strip() for p in str(venue_str).split(",", 1)]
    return parts[0], parts[1]


# -- Main pipeline ------------------------------------------------------------

def main():
    print("=" * 60)
    print("02_clean_merge.py  |  Cleaning + merging raw data")
    print("=" * 60)

    name_map = load_name_map()
    unmapped = set()

    def norm(x):
        if pd.isna(x):
            return x
        s = str(x).strip()
        if s in name_map:
            return name_map[s]
        unmapped.add(s)
        return s

    # ---- Load raw -----------------------------------------------------------
    section("Loading raw data")
    results = pd.read_csv(RAW / "results.csv")
    # results.csv has MIXED date formats -- early rows are ISO (1872-11-30),
    # most rows are US (2/3/1900). format='mixed' parses each row individually.
    results["date"] = pd.to_datetime(results["date"], format="mixed", errors="coerce")
    ok("results", f"{len(results):,} rows  (latest played: "
                   f"{results.dropna(subset=['home_score'])['date'].max().date()})")

    # Optional hand-curated supplement of matches the Kaggle source hasn't
    # picked up yet (typically April–early-June pre-WC friendlies). Same
    # schema as results.csv; deduped on (date, home_team, away_team) with
    # the supplemental winning if a duplicate exists.
    supp_path = RAW / "recent_results_supplemental.csv"
    if supp_path.exists():
        supp = pd.read_csv(supp_path)
        supp["date"] = pd.to_datetime(supp["date"], format="mixed", errors="coerce")
        # Only keep rows that have actual scores -- ignore future-fixture rows
        supp = supp.dropna(subset=["home_score", "away_score"]).copy()
        if len(supp):
            before = len(results)
            results = pd.concat([results, supp], ignore_index=True)
            results = results.drop_duplicates(
                subset=["date", "home_team", "away_team"], keep="last"
            ).reset_index(drop=True)
            added = len(results) - before
            ok("supplemental_results",
               f"+{added} from {supp_path.name} (latest played now: "
               f"{results.dropna(subset=['home_score'])['date'].max().date()})")
        else:
            ok("supplemental_results", f"{supp_path.name} present but empty")
    else:
        ok("supplemental_results",
           f"{supp_path.name} not present (optional)")

    shootouts = pd.read_csv(RAW / "shootouts.csv")
    shootouts["date"] = pd.to_datetime(shootouts["date"], errors="coerce")
    ok("shootouts", f"{len(shootouts):,} rows")

    fifa = pd.read_csv(RAW / "fifa_ranking_combined.csv")
    fifa["rank_date"] = pd.to_datetime(fifa["rank_date"], errors="coerce")
    ok("fifa_ranking", f"{len(fifa):,} rows")

    polymarket = pd.read_csv(RAW / "polymarket.csv")
    ok("polymarket", f"{len(polymarket):,} rows")

    odds = pd.read_csv(RAW / "odds_h2h.csv")
    ok("odds_h2h", f"{len(odds):,} rows")

    group_fix = pd.read_csv(RAW / "group_fixtures.csv")
    ok("group_fixtures", f"{len(group_fix)} rows")

    knockout = pd.read_csv(RAW / "knockout_slots.csv")
    ok("knockout_slots", f"{len(knockout)} rows")

    venues = pd.read_csv(EXTERNAL / "venues.csv")
    uefa_playoffs = pd.read_csv(EXTERNAL / "uefa_playoff_qualifiers.csv")
    fifa_playoffs_path = EXTERNAL / "fifa_intercontinental_qualifiers.csv"
    fifa_playoffs = pd.read_csv(fifa_playoffs_path) if fifa_playoffs_path.exists() else None

    # Current FIFA rankings (manually maintained -- pick most recent file)
    # Naming convention: fifa_<month>_rankings_manual.csv  (e.g. fifa_april_rankings_manual.csv)
    manual_ranks_files = sorted(EXTERNAL.glob("fifa_*_rankings_manual.csv"),
                                key=lambda p: p.stat().st_mtime)
    if manual_ranks_files:
        latest_manual = manual_ranks_files[-1]
        current_fifa = pd.read_csv(latest_manual)
        current_fifa["team"] = current_fifa["team"].map(norm)
        ok("fifa_current", f"{len(current_fifa)} teams from {latest_manual.name}")
    else:
        current_fifa = None
        warn("fifa_current",
             "no fifa_*_rankings_manual.csv found -- falling back to latest snapshot")

    # ---- Resolve playoff placeholders (UEFA + FIFA intercontinental) -------
    section("Resolving playoff placeholders")
    playoff_map = {f"UEFA Playoff {row.pathway}": row.qualifier
                   for row in uefa_playoffs.itertuples()}
    if fifa_playoffs is not None:
        playoff_map.update({row.placeholder: row.qualifier
                            for row in fifa_playoffs.itertuples()})
    for col in ["home_team", "away_team"]:
        before = group_fix[col].isin(playoff_map.keys()).sum()
        group_fix[col] = group_fix[col].replace(playoff_map)
        ok(f"group_fix.{col}", f"{before} placeholders resolved")

    # ---- Normalise team names across all sources ---------------------------
    section("Normalising team names")
    for col in ["home_team", "away_team"]:
        results[col] = results[col].map(norm)
        shootouts[col] = shootouts[col].map(norm)
        odds[col] = odds[col].map(norm)
        group_fix[col] = group_fix[col].map(norm)
    shootouts["winner"] = shootouts["winner"].map(norm)
    fifa["country_full"] = fifa["country_full"].map(norm)
    ok("normalised", "team columns across 5 sources")

    # ---- Polymarket: extract team + implied probability --------------------
    section("Parsing Polymarket markets")
    polymarket["team_raw"] = polymarket["market"].map(extract_pm_team)
    polymarket["team"] = polymarket["team_raw"].map(lambda t: norm(t) if t else None)
    pm_yes = polymarket[(polymarket["outcome"].str.lower() == "yes") &
                        (polymarket["team"].notna())].copy()
    pm_yes = pm_yes[["team", "price", "volume", "liquidity"]].rename(
        columns={"price": "polymarket_win_prob"}
    )
    pm_yes = pm_yes.sort_values("liquidity", ascending=False).drop_duplicates("team")
    ok("polymarket_winners", f"{len(pm_yes)} teams with win probabilities")
    if len(pm_yes):
        top = pm_yes.sort_values("polymarket_win_prob", ascending=False).head(5)
        print(f"          Top 5: {top[['team', 'polymarket_win_prob']].to_dict('records')}")

    # ---- Filter & flag historical matches ----------------------------------
    section("Filtering historical matches")
    results = results[results["date"] >= "2010-01-01"].copy()
    results["is_competitive"] = results["tournament"].map(is_competitive)
    results["total_goals"] = results["home_score"] + results["away_score"]
    results["goal_diff"] = results["home_score"] - results["away_score"]
    results["winner"] = results.apply(
        lambda r: r["home_team"] if r["home_score"] > r["away_score"]
        else (r["away_team"] if r["away_score"] > r["home_score"] else "Draw"),
        axis=1
    )
    n_comp = int(results["is_competitive"].sum())
    ok("matches_filtered", f"{len(results):,} matches from 2010+ ({n_comp:,} competitive)")

    # ---- Resolve fixtures: split venue + flag host advantage ---------------
    section("Resolving fixtures")
    group_fix["round"] = "Group"
    group_fix["multiplier"] = 1
    group_fix["date_utc"] = pd.to_datetime(group_fix["date_utc"], errors="coerce")
    group_fix[["stadium", "city"]] = group_fix["venue"].apply(
        lambda v: pd.Series(split_venue(v))
    )

    knockout["date_utc"] = pd.to_datetime(knockout["date_utc"], errors="coerce")
    knockout[["stadium", "city"]] = knockout["venue"].apply(
        lambda v: pd.Series(split_venue(v))
    )

    venue_cols = ["stadium", "country", "altitude_m", "avg_june_temp_c",
                  "avg_june_humidity_pct", "roof_covered"]
    group_fix = group_fix.merge(venues[venue_cols], on="stadium", how="left")
    knockout = knockout.merge(venues[venue_cols], on="stadium", how="left")

    missing_venues = group_fix["altitude_m"].isna().sum()
    if missing_venues:
        warn("venue_join", f"{missing_venues} group fixtures failed venue join")
        warn("missing_stadiums",
             str(group_fix.loc[group_fix["altitude_m"].isna(), "stadium"].unique()))

    # Host advantage flag (team's country matches venue country)
    host_country = {"United States": "US", "Mexico": "MX", "Canada": "CA"}

    def is_host(team, venue_country):
        return int(host_country.get(str(team), "") == str(venue_country))

    group_fix["home_is_host"] = group_fix.apply(
        lambda r: is_host(r["home_team"], r["country"]), axis=1
    )
    group_fix["away_is_host"] = group_fix.apply(
        lambda r: is_host(r["away_team"], r["country"]), axis=1
    )
    ok("fixtures_group", f"{len(group_fix)} fixtures with venue + host data")
    ok("fixtures_knockout", f"{len(knockout)} slots with venue data")
    n_host = int(group_fix["home_is_host"].sum() + group_fix["away_is_host"].sum())
    ok("host_matches", f"{n_host} group fixtures involve a host nation")

    # ---- Per-team feature table --------------------------------------------
    section("Building per-team features")

    # Latest historical snapshot from Kaggle (always built as fallback)
    historical_latest = (
        fifa.sort_values("rank_date")
            .groupby("country_full")
            .tail(1)
            [["country_full", "rank", "total_points", "confederation", "rank_date"]]
            .rename(columns={
                "country_full": "team",
                "rank": "fifa_rank",
                "total_points": "fifa_points",
                "rank_date": "fifa_rank_date",
            })
    )

    if current_fifa is not None:
        manual = current_fifa[["team", "rank", "points", "confederation"]].rename(
            columns={"rank": "fifa_rank", "points": "fifa_points"}
        )
        manual["fifa_rank_date"] = pd.to_datetime(
            manual_ranks_files[-1].stat().st_mtime, unit="s"
        ).normalize()
        # Manual file is primary; historical fills any teams missing from the manual file
        gap_teams = historical_latest[~historical_latest["team"].isin(manual["team"])]
        latest_fifa = pd.concat([manual, gap_teams], ignore_index=True)
        ok("fifa_source",
           f"{len(manual)} from {manual_ranks_files[-1].name} + "
           f"{len(gap_teams)} from June 2024 snapshot (fallback)")
    else:
        latest_fifa = historical_latest
        warn("fifa_source", "using historical snapshot only (June 2024)")

    # Recent form: last 20 competitive matches per team
    comp = results[results["is_competitive"]].copy()
    home_view = comp[["date", "home_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "home_score": "gf", "away_score": "ga"}
    )
    away_view = comp[["date", "away_team", "away_score", "home_score"]].rename(
        columns={"away_team": "team", "away_score": "gf", "home_score": "ga"}
    )
    team_matches = pd.concat([home_view, away_view], ignore_index=True).sort_values("date")
    team_matches["result"] = team_matches.apply(
        lambda r: "W" if r["gf"] > r["ga"] else ("L" if r["gf"] < r["ga"] else "D"),
        axis=1
    )

    def last_n_stats(group, n=20):
        g = group.tail(n)
        return pd.Series({
            "matches_played":    len(g),
            "avg_goals_for":     g["gf"].mean(),
            "avg_goals_against": g["ga"].mean(),
            "win_rate":          (g["result"] == "W").mean(),
            "draw_rate":         (g["result"] == "D").mean(),
            "last_match":        g["date"].max(),
        })

    form = (
        team_matches.groupby("team", group_keys=False)
                    .apply(last_n_stats, n=20)
                    .reset_index()
    )

    # Shootout record since 2005
    s = shootouts[shootouts["date"] >= "2005-01-01"].copy()
    shootout_rows = []
    for _, r in s.iterrows():
        for team in [r["home_team"], r["away_team"]]:
            shootout_rows.append({"team": team, "won": int(r["winner"] == team)})
    if shootout_rows:
        sdf = pd.DataFrame(shootout_rows)
        shootout_record = sdf.groupby("team").agg(
            shootouts_played=("won", "count"),
            shootouts_won=("won", "sum"),
        ).reset_index()
        shootout_record["shootout_win_rate"] = (
            shootout_record["shootouts_won"] / shootout_record["shootouts_played"]
        )
    else:
        shootout_record = pd.DataFrame(
            columns=["team", "shootouts_played", "shootouts_won", "shootout_win_rate"]
        )

    teams_in_wc = pd.unique(pd.concat([group_fix["home_team"], group_fix["away_team"]]))
    team_features = pd.DataFrame({"team": teams_in_wc})
    team_features = (
        team_features
            .merge(latest_fifa, on="team", how="left")
            .merge(form, on="team", how="left")
            .merge(pm_yes, on="team", how="left")
            .merge(shootout_record, on="team", how="left")
    )

    # WC 2022 detailed match stats (corners/cards/possession) if 01c was run
    wc22_path = PROCESSED / "match_stats_wc2022.csv"
    if wc22_path.exists():
        wc22 = pd.read_csv(wc22_path)
        agg_targets = {
            "corners":         "wc22_avg_corners",
            "yellow cards":    "wc22_avg_yellows",
            "red cards":       "wc22_avg_reds",
            "fouls against":   "wc22_avg_fouled",
            "possession":      "wc22_avg_possession",
            "total attempts":  "wc22_avg_shots",
            "offsides":        "wc22_avg_offsides",
        }
        available = {src: dst for src, dst in agg_targets.items() if src in wc22.columns}
        if available:
            # Defensive: force numeric on the columns we're aggregating
            for c in available:
                wc22[c] = pd.to_numeric(wc22[c], errors="coerce")
            wc22_avg = (
                wc22.groupby("team")[list(available.keys())]
                    .mean()
                    .rename(columns=available)
                    .reset_index()
            )
            wc22_avg["wc22_matches"] = wc22.groupby("team").size().values
            team_features = team_features.merge(wc22_avg, on="team", how="left")
            ok("wc22_stats_merged", f"{len(wc22_avg)} teams enriched with WC 2022 data")
    else:
        warn("wc22_stats", "match_stats_wc2022.csv not found -- run 01c first")

    ok("team_features", f"{len(team_features)} WC 2026 teams")
    missing_rank = int(team_features["fifa_rank"].isna().sum())
    missing_form = int(team_features["matches_played"].isna().sum())
    if missing_rank:
        warn("team_features.fifa", f"{missing_rank} teams missing FIFA rank")
        warn("missing_teams",
             str(team_features.loc[team_features["fifa_rank"].isna(), "team"].tolist()))
    if missing_form:
        warn("team_features.form", f"{missing_form} teams missing recent form")

    # ---- Save outputs ------------------------------------------------------
    section("Writing processed files")
    results.to_csv(PROCESSED / "matches_clean.csv", index=False)
    ok("matches_clean.csv", f"{len(results):,} rows")

    group_fix.to_csv(PROCESSED / "fixtures_group.csv", index=False)
    ok("fixtures_group.csv", f"{len(group_fix)} rows")

    knockout.to_csv(PROCESSED / "fixtures_knockout.csv", index=False)
    ok("fixtures_knockout.csv", f"{len(knockout)} rows")

    team_features.to_csv(PROCESSED / "team_features.csv", index=False)
    ok("team_features.csv", f"{len(team_features)} rows")

    pm_yes.to_csv(PROCESSED / "polymarket_clean.csv", index=False)
    ok("polymarket_clean.csv", f"{len(pm_yes)} rows")

    audit_df = pd.DataFrame({"unmapped_name": sorted(unmapped)})
    audit_df.to_csv(PROCESSED / "name_audit.csv", index=False)
    ok("name_audit.csv",
       f"{len(audit_df)} distinct names encountered (most are already canonical)")

    print("\n" + "=" * 60)
    print("Done.  Processed files in data/processed/")
    print("=" * 60)


if __name__ == "__main__":
    main()
