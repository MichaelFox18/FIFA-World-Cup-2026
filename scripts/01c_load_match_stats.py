"""
01c_load_match_stats.py
Loads the WC 2022 detailed match stats dataset (Kaggle), reshapes from the
team1/team2 wide format into one-row-per-team-per-match, normalises team
names, and saves to data/processed/ for use by feature engineering.

Run after dropping Fifa_world_cup_complete_data.csv in data/raw/.
Run from worldcup2026/:  py scripts/01c_load_match_stats.py

Outputs:
  data/processed/match_stats_wc2022.csv  -- long format (team-match rows)
"""

import re
import pandas as pd
from pathlib import Path

_TEAM_SUFFIX_RE = re.compile(r"\s*team[12]$")


def strip_team_suffix(col: str) -> str:
    """'corners team1' -> 'corners',  'completed line breaksteam1' -> 'completed line breaks'."""
    return _TEAM_SUFFIX_RE.sub("", col).strip()

RAW = Path("data/raw")
EXTERNAL = Path("data/external")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)


def section(title):
    print(f"\n-- {title} " + "-" * (60 - len(title)))


def ok(name, detail=""):
    print(f"  [OK]   {name:<24} {detail}")


def warn(name, detail=""):
    print(f"  [WARN] {name:<24} {detail}")


# Common all-caps team labels in this dataset that need explicit mapping
# (others get title-cased then run through the standard name_map)
CAPS_MAP = {
    "USA":              "United States",
    "UAE":              "United Arab Emirates",
    "KSA":              "Saudi Arabia",
    "DR CONGO":         "DR Congo",
}


def load_name_map():
    df = pd.read_csv(EXTERNAL / "team_name_map.csv")
    return dict(zip(df["variant"].str.strip(), df["canonical"].str.strip()))


def normalize_team(name, name_map):
    if pd.isna(name):
        return name
    s = str(name).strip()
    if s in CAPS_MAP:
        s = CAPS_MAP[s]
    else:
        s = s.title()
    return name_map.get(s, s)


def reshape_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """team1/team2 wide -> one row per team per match (long)."""
    # Normalise column names: collapse double spaces, lowercase
    df = df.copy()
    df.columns = [c.strip().replace("  ", " ").lower() for c in df.columns]

    # Common (non-side-specific) columns
    common = ["date", "hour", "category"]
    # Exclude the team-name columns themselves -- those are handled separately
    team1_cols = [c for c in df.columns if c.endswith("team1") and c != "team1"]
    team2_cols = [c for c in df.columns if c.endswith("team2") and c != "team2"]

    if not team1_cols or not team2_cols:
        raise ValueError("Could not find team1/team2 stat columns")

    # team1 perspective
    t1 = df[["team1", "team2"] + common + team1_cols].copy()
    t1 = t1.rename(columns={"team1": "team", "team2": "opponent"})
    t1.columns = [strip_team_suffix(c) if c not in ("team", "opponent") else c
                  for c in t1.columns]

    # team2 perspective
    t2 = df[["team2", "team1"] + common + team2_cols].copy()
    t2 = t2.rename(columns={"team2": "team", "team1": "opponent"})
    t2.columns = [strip_team_suffix(c) if c not in ("team", "opponent") else c
                  for c in t2.columns]

    # Ensure column order matches before concat
    t2 = t2[t1.columns]
    return pd.concat([t1, t2], ignore_index=True)


def clean_numeric_with_pct(df: pd.DataFrame) -> pd.DataFrame:
    """possession columns have '%' suffix -- strip and convert to numeric."""
    for col in df.columns:
        if col in ("team", "opponent", "date", "hour", "category"):
            continue
        if df[col].dtype == object:
            # Try strip-and-convert; leave alone if it fails
            try:
                df[col] = (
                    df[col].astype(str)
                    .str.replace("%", "", regex=False)
                    .str.strip()
                    .replace({"": None})
                    .astype(float)
                )
            except (ValueError, TypeError):
                pass
    return df


def main():
    print("=" * 60)
    print("01c_load_match_stats.py  |  WC 2022 detailed match stats")
    print("=" * 60)

    section("Loading raw dataset")
    src = RAW / "Fifa_world_cup_complete_data.csv"
    if not src.exists():
        warn("missing", f"{src} not found")
        return
    raw = pd.read_csv(src)
    ok("raw", f"{len(raw)} matches × {len(raw.columns)} cols")

    section("Reshaping wide -> long")
    long = reshape_to_long(raw)
    long = clean_numeric_with_pct(long)
    ok("long_format", f"{len(long)} team-match rows")

    section("Normalising team names")
    name_map = load_name_map()
    long["team"] = long["team"].map(lambda n: normalize_team(n, name_map))
    long["opponent"] = long["opponent"].map(lambda n: normalize_team(n, name_map))
    long["date"] = pd.to_datetime(long["date"], errors="coerce", dayfirst=False)
    long["tournament"] = "WC 2022"
    ok("normalised", f"{long['team'].nunique()} distinct teams")
    teams_seen = sorted(long["team"].unique().tolist())
    print(f"          Teams: {teams_seen}")

    section("Sanity check: key stat columns present")
    key_cols = ["corners", "yellow cards", "red cards", "possession",
                "fouls against", "number of goals", "offsides", "total attempts"]
    found = [c for c in key_cols if c in long.columns]
    missing = [c for c in key_cols if c not in long.columns]
    ok("found", ", ".join(found))
    if missing:
        warn("missing", ", ".join(missing))

    section("Per-team averages preview")
    agg_cols = [c for c in ["corners", "yellow cards", "red cards"]
                if c in long.columns]
    if agg_cols:
        preview = (
            long.groupby("team")[agg_cols]
                .mean().round(2)
                .sort_values(agg_cols[0], ascending=False)
                .head(8)
        )
        print(preview.to_string())

    out = PROCESSED / "match_stats_wc2022.csv"
    long.to_csv(out, index=False)
    print(f"\nSaved {len(long)} rows -> {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
