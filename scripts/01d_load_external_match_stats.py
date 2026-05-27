"""
01d_load_external_match_stats.py
Loads additional international-tournament match stats (Euro 2024, Copa America
2024, AFCON 2023/2025, etc.) from CSVs the user drops into
data/raw/external_stats/, reshapes them to the same long format as
match_stats_wc2022.csv, and saves to:

  data/processed/match_stats_external.csv

Scripts 05 and 06 concatenate this with the WC22 data to grow the corners and
cards training pool from 64 matches to ~250+.

Run from worldcup2026/:  py scripts/01d_load_external_match_stats.py

The loader auto-detects three common Kaggle CSV layouts:
  A) Match-level wide:   home_team, away_team, date, home_corners, away_corners,
                         home_yellow_cards, away_yellow_cards, home_red_cards,
                         away_red_cards  (any case/spacing/underscore variant)
  B) Football-data.co.uk:  HomeTeam, AwayTeam, Date, HC, AC, HY, AY, HR, AR
  C) team1/team2 wide:   columns ending in `team1` / `team2`  (WC22 Kaggle layout)

The filename's stem becomes the `tournament` tag (e.g.,
`euro_2024.csv` -> `tournament="Euro 2024"`). Each file is also expected to
have a `category` column or the script falls back to `"Group"`. Files whose
schema can't be parsed are skipped with a warning.
"""

import re
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
EXTERNAL_DIR = RAW / "external_stats"
EXTERNAL = Path("data/external")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)


def section(t): print(f"\n-- {t} " + "-" * (60 - len(t)))
def ok(name, detail=""): print(f"  [OK]   {name:<28} {detail}")
def warn(name, detail=""): print(f"  [WARN] {name:<28} {detail}")
def skip(name, detail=""): print(f"  [SKIP] {name:<28} {detail}")


# Some all-caps labels seen in tournament datasets that need explicit mapping
# before the standard team_name_map normalization is applied.
CAPS_MAP = {
    "USA":              "United States",
    "UAE":              "United Arab Emirates",
    "KSA":              "Saudi Arabia",
    "DR CONGO":         "DR Congo",
    "DRC":              "DR Congo",
    "RSA":              "South Africa",
}


def load_name_map():
    df = pd.read_csv(EXTERNAL / "team_name_map.csv")
    return dict(zip(df["variant"].astype(str).str.strip(),
                    df["canonical"].astype(str).str.strip()))


def normalize_team(name, name_map):
    if pd.isna(name):
        return name
    s = str(name).strip()
    if s in CAPS_MAP:
        s = CAPS_MAP[s]
    else:
        s = s.title()
    return name_map.get(s, s)


# Column-name aliases. Lowercase, alphanumeric-only keys are matched against
# similarly-stripped column names so spaces / underscores / case don't matter.
HOME_TEAM_ALIASES = {"hometeam", "home", "homename", "team1", "homeside",
                     "team_1", "team_home"}
AWAY_TEAM_ALIASES = {"awayteam", "away", "awayname", "team2", "awayside",
                     "team_2", "team_away"}
DATE_ALIASES      = {"date", "matchdate", "kickoff", "kickoffdate"}
CATEGORY_ALIASES  = {"category", "stage", "round", "phase"}
HOME_CORNERS      = {"homecorners", "hc", "corners1", "homecorner",
                     "h_corners", "cornersteam1",
                     "team1corners", "team1cornerkicks"}
AWAY_CORNERS      = {"awaycorners", "ac", "corners2", "awaycorner",
                     "a_corners", "cornersteam2",
                     "team2corners", "team2cornerkicks"}
HOME_YELLOW       = {"homeyellowcards", "homeyellows", "hy", "yellow1",
                     "homeyellow", "h_yellow", "yellowcardsteam1",
                     "yellowcardteam1",
                     "team1yellowcards", "team1yellows"}
AWAY_YELLOW       = {"awayyellowcards", "awayyellows", "ay", "yellow2",
                     "awayyellow", "a_yellow", "yellowcardsteam2",
                     "yellowcardteam2",
                     "team2yellowcards", "team2yellows"}
HOME_RED          = {"homeredcards", "homereds", "hr", "red1", "homered",
                     "h_red", "redcardsteam1", "redcardteam1",
                     "team1redcards", "team1reds"}
AWAY_RED          = {"awayredcards", "awayreds", "ar", "red2", "awayred",
                     "a_red", "redcardsteam2", "redcardteam2",
                     "team2redcards", "team2reds"}
# xG (expected goals) - only present in some datasets (Euro 2024, Copa 2024).
# Multiple Kaggle variants of the column name. Pick the first found.
HOME_XG           = {"homeexpectedgoalsxg", "homexg", "homenonpenaltyxg",
                     "team1xg", "homeexpectedgoals"}
AWAY_XG           = {"awayexpectedgoalsxg", "awayxg", "awaynonpenaltyxg",
                     "team2xg", "awayexpectedgoals"}

# Many Kaggle tournament datasets omit the match date entirely. We fall back
# to a tournament-start date and assign sequential day offsets so each match
# gets a unique date (which the downstream LOO uses to disambiguate matches).
# Tournament tag (from filename) -> ISO start date.
TOURNAMENT_START_DATES = {
    "Euro 2024":          "2024-06-14",
    "Copa America 2024":  "2024-06-20",
    "Afcon 2023":         "2024-01-13",  # AFCON "2023" was played Jan-Feb 2024
    "Afcon 2025 2026":    "2025-12-21",  # AFCON "2025" runs Dec 2025 - Jan 2026
}


def normkey(s: str) -> str:
    """Lowercase + strip everything except a-z0-9."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def find_col(cols_normmap: dict, aliases: set) -> str | None:
    """Return the original column name whose normkey is in aliases."""
    for norm, orig in cols_normmap.items():
        if norm in aliases:
            return orig
    return None


def parse_match_level(df: pd.DataFrame) -> pd.DataFrame | None:
    """Format A or B: one row per match with home_*/away_* columns.

    Date column is optional -- if missing, the caller will synthesize dates from
    the tournament's known start date.
    """
    cols_normmap = {normkey(c): c for c in df.columns}
    home_team = find_col(cols_normmap, HOME_TEAM_ALIASES)
    away_team = find_col(cols_normmap, AWAY_TEAM_ALIASES)
    date_col  = find_col(cols_normmap, DATE_ALIASES)
    if not (home_team and away_team):
        return None
    hc = find_col(cols_normmap, HOME_CORNERS)
    ac = find_col(cols_normmap, AWAY_CORNERS)
    hy = find_col(cols_normmap, HOME_YELLOW)
    ay = find_col(cols_normmap, AWAY_YELLOW)
    hr = find_col(cols_normmap, HOME_RED)
    ar = find_col(cols_normmap, AWAY_RED)
    hxg = find_col(cols_normmap, HOME_XG)
    axg = find_col(cols_normmap, AWAY_XG)
    cat = find_col(cols_normmap, CATEGORY_ALIASES)
    # Need at least corners or yellows to be useful
    if not ((hc and ac) or (hy and ay)):
        return None

    if date_col:
        date_series = pd.to_datetime(df[date_col], errors="coerce", dayfirst=False)
        # If dates failed to parse with default order, try day-first
        if date_series.isna().mean() > 0.5:
            date_series = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    else:
        date_series = pd.Series([pd.NaT] * len(df))

    out = pd.DataFrame({
        "home_team":    df[home_team].astype(str),
        "away_team":    df[away_team].astype(str),
        "date":         date_series,
        "category":     df[cat].astype(str) if cat else "Group",
        "home_corners": df[hc] if hc else pd.NA,
        "away_corners": df[ac] if ac else pd.NA,
        "home_yellow":  df[hy] if hy else pd.NA,
        "away_yellow":  df[ay] if ay else pd.NA,
        "home_red":     df[hr] if hr else 0,
        "away_red":     df[ar] if ar else 0,
        "home_xg":      df[hxg] if hxg else pd.NA,
        "away_xg":      df[axg] if axg else pd.NA,
    })
    return out


def synthesize_dates(match_df: pd.DataFrame, tournament: str) -> pd.DataFrame:
    """If date column is missing or empty, assign sequential days starting from
    the known tournament start. Used for Kaggle datasets that omit kickoff dates."""
    if match_df["date"].notna().any():
        return match_df
    start = TOURNAMENT_START_DATES.get(tournament)
    if not start:
        warn("no_start_date", f"no synthetic date known for {tournament}")
        return match_df
    base = pd.Timestamp(start)
    match_df = match_df.copy()
    match_df["date"] = [base + pd.Timedelta(days=i) for i in range(len(match_df))]
    return match_df


def parse_team1_team2_wide(df: pd.DataFrame) -> pd.DataFrame | None:
    """Format C: WC22-style wide layout with *team1/*team2 suffix columns."""
    cols = [c.strip().lower() for c in df.columns]
    df = df.copy()
    df.columns = cols
    if not ({"team1", "team2"} <= set(cols)):
        return None
    # We need at least corners team1/team2 or yellow cards team1/team2
    needed = ["corners team1", "corners team2",
              "yellow cards team1", "yellow cards team2"]
    if not any(c in cols for c in needed):
        return None
    out = pd.DataFrame({
        "home_team":    df["team1"].astype(str),
        "away_team":    df["team2"].astype(str),
        "date":         pd.to_datetime(df.get("date"), errors="coerce", dayfirst=False),
        "category":     df["category"].astype(str) if "category" in cols else "Group",
        "home_corners": df.get("corners team1"),
        "away_corners": df.get("corners team2"),
        "home_yellow":  df.get("yellow cards team1"),
        "away_yellow":  df.get("yellow cards team2"),
        "home_red":     df.get("red cards team1", 0),
        "away_red":     df.get("red cards team2", 0),
        "home_xg":      pd.NA,
        "away_xg":      pd.NA,
    })
    return out


def reshape_to_long(match_df: pd.DataFrame, tournament: str,
                    name_map: dict) -> pd.DataFrame:
    """Wide match-level -> two long-format rows (one per team perspective)."""
    side_a = pd.DataFrame({
        "team":         match_df["home_team"].map(lambda n: normalize_team(n, name_map)),
        "opponent":     match_df["away_team"].map(lambda n: normalize_team(n, name_map)),
        "date":         match_df["date"],
        "hour":         None,
        "category":     match_df["category"],
        "corners":      pd.to_numeric(match_df["home_corners"], errors="coerce"),
        "yellow cards": pd.to_numeric(match_df["home_yellow"], errors="coerce"),
        "red cards":    pd.to_numeric(match_df["home_red"],    errors="coerce").fillna(0),
        "xg_for":       pd.to_numeric(match_df.get("home_xg"), errors="coerce"),
        "xg_against":   pd.to_numeric(match_df.get("away_xg"), errors="coerce"),
        "tournament":   tournament,
    })
    side_b = pd.DataFrame({
        "team":         match_df["away_team"].map(lambda n: normalize_team(n, name_map)),
        "opponent":     match_df["home_team"].map(lambda n: normalize_team(n, name_map)),
        "date":         match_df["date"],
        "hour":         None,
        "category":     match_df["category"],
        "corners":      pd.to_numeric(match_df["away_corners"], errors="coerce"),
        "yellow cards": pd.to_numeric(match_df["away_yellow"], errors="coerce"),
        "red cards":    pd.to_numeric(match_df["away_red"],    errors="coerce").fillna(0),
        "xg_for":       pd.to_numeric(match_df.get("away_xg"), errors="coerce"),
        "xg_against":   pd.to_numeric(match_df.get("home_xg"), errors="coerce"),
        "tournament":   tournament,
    })
    return pd.concat([side_a, side_b], ignore_index=True)


def tournament_from_filename(fname: str) -> str:
    """euro_2024.csv -> 'Euro 2024'.  afcon-2025.csv -> 'Afcon 2025'."""
    stem = Path(fname).stem
    parts = re.split(r"[_\-\s]+", stem)
    return " ".join(p.capitalize() for p in parts if p)


def main():
    print("=" * 60)
    print("01d_load_external_match_stats.py  |  international tournament expansion")
    print("=" * 60)

    if not EXTERNAL_DIR.exists():
        warn("missing_dir", f"create {EXTERNAL_DIR} and drop CSVs there")
        return
    files = sorted([p for p in EXTERNAL_DIR.glob("*.csv")])
    if not files:
        warn("no_files", f"drop tournament CSVs into {EXTERNAL_DIR} (see README)")
        return
    ok("found", f"{len(files)} CSV(s) in {EXTERNAL_DIR}")

    name_map = load_name_map()

    all_long = []
    per_file_summary = []
    for f in files:
        section(f"Parsing {f.name}")
        try:
            raw = pd.read_csv(f)
        except pd.errors.ParserError:
            # Some Kaggle exports have stray commas in unquoted goal-scorer fields.
            # Retry with the python engine, which is more tolerant and lets us
            # skip malformed rows individually.
            try:
                raw = pd.read_csv(f, engine="python", on_bad_lines="skip")
                warn("malformed_csv", "skipped bad rows (used python engine)")
            except (pd.errors.ParserError, UnicodeDecodeError) as e:
                warn("read_error", str(e))
                continue
        except UnicodeDecodeError as e:
            warn("read_error", str(e))
            continue
        ok("raw_shape", f"{len(raw)} rows x {len(raw.columns)} cols")

        match_df = parse_match_level(raw)
        layout = "match_level"
        if match_df is None:
            match_df = parse_team1_team2_wide(raw)
            layout = "team1_team2_wide"
        if match_df is None:
            skip("unrecognised", "no known column layout matched -- see header docs")
            continue
        ok("layout", layout)

        tournament = tournament_from_filename(f.name)

        # If the source file omits dates, synthesize them from a known
        # tournament start date so each match has a unique (pair, date) key.
        had_real_dates = match_df["date"].notna().any()
        match_df = synthesize_dates(match_df, tournament)
        if not had_real_dates and match_df["date"].notna().any():
            ok("dates_synthesized", f"start={match_df['date'].min().date()}")

        # Drop rows missing the essentials
        before = len(match_df)
        match_df = match_df.dropna(subset=["home_team", "away_team", "date"]).copy()
        if len(match_df) < before:
            warn("dropped_incomplete", f"{before - len(match_df)} rows missing team/date")

        long = reshape_to_long(match_df, tournament, name_map)

        # Keep only rows where at least corners or yellow cards is present
        keep = long["corners"].notna() | long["yellow cards"].notna()
        long = long.loc[keep].copy()

        if long.empty:
            skip("no_useable_rows", "all rows missing both corners and yellows")
            continue

        n_matches = long.groupby(["date", "team", "opponent"]).ngroups // 2
        ok("tournament_tag", tournament)
        ok("long_rows", f"{len(long)} ({n_matches} matches)")
        ok("teams", f"{long['team'].nunique()} distinct")
        if long["corners"].notna().any():
            ok("mean_corners", f"{long['corners'].mean():.2f} per team")
        if long["yellow cards"].notna().any():
            ok("mean_yellow", f"{long['yellow cards'].mean():.2f} per team")

        per_file_summary.append((f.name, tournament, n_matches,
                                 long["corners"].mean(),
                                 long["yellow cards"].mean()))
        all_long.append(long)

    if not all_long:
        warn("nothing_parsed", "no files produced usable long-format data")
        return

    section("Combining")
    combined = pd.concat(all_long, ignore_index=True)
    # Dedupe (sometimes the same match appears in two source files)
    combined = combined.drop_duplicates(
        subset=["team", "opponent", "date"], keep="first"
    ).reset_index(drop=True)
    ok("combined_rows", f"{len(combined)}")
    n_matches_total = combined.groupby(["date", "team", "opponent"]).ngroups // 2
    ok("combined_matches", f"~{n_matches_total} matches")

    out = PROCESSED / "match_stats_external.csv"
    combined.to_csv(out, index=False)
    print(f"\nSaved {len(combined)} rows -> {out}")

    section("Per-file summary")
    summary = pd.DataFrame(per_file_summary,
                           columns=["file", "tournament", "matches",
                                    "mean_corners", "mean_yellow"]).round(2)
    print(summary.to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
