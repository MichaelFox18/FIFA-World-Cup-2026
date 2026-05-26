"""
01b_collect_match_details.py
Scrapes FBRef tournament hub pages for per-team season stats covering recent
international tournaments. Fills the corners/cards data gap that the Kaggle
results.csv leaves us with.

Run once:  py scripts/01b_collect_match_details.py
Takes 30-60 seconds with polite rate limiting.

Output:
  data/raw/fbref_tournament_stats.csv  -- per-team-per-tournament aggregates
"""

import time
import re
import cloudscraper
import pandas as pd
from io import StringIO
from pathlib import Path

BASE = "https://fbref.com"
RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

# cloudscraper handles Cloudflare's JS challenge that plain requests trips on.
SCRAPER = cloudscraper.create_scraper(
    browser={"browser": "chrome", "platform": "windows", "mobile": False}
)

REQUEST_DELAY = 3.5  # be polite

# Recent international tournaments (FBRef comp IDs + slug used in URL)
TOURNAMENTS = [
    {"id": 1,   "year": 2022, "slug": "World-Cup",                 "tag": "wc2022"},
    {"id": 676, "year": 2024, "slug": "European-Championship",      "tag": "euro2024"},
    {"id": 685, "year": 2024, "slug": "Copa-America",               "tag": "copa2024"},
    {"id": 656, "year": 2024, "slug": "Africa-Cup-of-Nations",      "tag": "afcon2024"},
    {"id": 664, "year": 2023, "slug": "Asian-Cup",                  "tag": "asian2023"},
]


def section(title):
    print(f"\n-- {title} " + "-" * (60 - len(title)))


def ok(name, detail=""):
    print(f"  [OK]   {name:<24} {detail}")


def fail(name, detail=""):
    print(f"  [FAIL] {name:<24} {detail}")


def fetch_hub(comp_id, year, slug):
    """Fetch the tournament hub page and uncomment hidden tables."""
    url = f"{BASE}/en/comps/{comp_id}/{year}/{year}-{slug}-Stats"
    time.sleep(REQUEST_DELAY)
    resp = SCRAPER.get(url, timeout=45)
    resp.raise_for_status()
    # FBRef hides some stat tables in HTML comments
    return resp.text.replace("<!--", "").replace("-->", ""), url


def flatten_cols(df):
    """Flatten pandas MultiIndex columns from FBRef's nested headers."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(c for c in col if c and not str(c).startswith("Unnamed"))
            for col in df.columns
        ]
    df.columns = [str(c).strip("_").lower() for c in df.columns]
    return df


def find_team_col(cols):
    """Identify the team-name column in a table (varies by table type)."""
    for c in cols:
        if "squad" in c.lower() or "team" in c.lower():
            return c
    return None


def clean_team_name(name):
    """Strip FBRef prefixes/suffixes like 'eng ENG' or 'vs Mexico'."""
    if pd.isna(name):
        return name
    s = str(name).strip()
    # Remove leading country code prefixes like "eng ENG" or trailing codes
    s = re.sub(r"^[a-z]{2,3}\s+", "", s)
    s = re.sub(r"\s+[A-Z]{3}$", "", s)
    return s.strip()


def extract_team_stats(html):
    """Parse all team-stats tables on a hub page, merge by team."""
    tables = pd.read_html(StringIO(html))
    keep_keys = {
        # Map of canonical_name -> set of partial column matches
        "matches_played":  ["mp", "matchesplayed"],
        "wins":            ["w_w", "wins_w", "_w_"],
        "draws":           ["d_d", "draws_d", "_d_"],
        "losses":          ["l_l", "losses_l", "_l_"],
        "goals_for":       ["gf"],
        "goals_against":   ["ga"],
        "xg":              ["xg_xg", "_xg_for"],
        "xga":             ["xga"],
        "yellow_cards":    ["crdy"],
        "red_cards":       ["crdr"],
        "corners":         ["ck"],
        "fouls":           ["fls"],
        "offsides":        ["off_misc"],
        "possession":      ["poss"],
    }

    merged = None
    for t in tables:
        if len(t) < 4:
            continue
        t = flatten_cols(t.copy())
        team_col = find_team_col(t.columns)
        if not team_col:
            continue
        t = t.rename(columns={team_col: "team"})
        t["team"] = t["team"].map(clean_team_name)
        t = t[t["team"].notna() & ~t["team"].str.contains("Squad", case=False, na=False)]
        if merged is None:
            merged = t
        else:
            new_cols = [c for c in t.columns if c == "team" or c not in merged.columns]
            merged = merged.merge(t[new_cols], on="team", how="outer")
    return merged


def main():
    print("=" * 60)
    print("01b_collect_match_details.py  |  FBRef team tournament stats")
    print("=" * 60)

    all_rows = []
    for tn in TOURNAMENTS:
        section(f"{tn['tag']}  ({tn['year']}-{tn['slug']})")
        try:
            html, url = fetch_hub(tn["id"], tn["year"], tn["slug"])
            df = extract_team_stats(html)
            if df is None or df.empty:
                fail(tn["tag"], "no team tables found on hub page")
                continue
            df["tournament"] = tn["tag"]
            df["tournament_year"] = tn["year"]
            all_rows.append(df)
            ok(tn["tag"], f"{len(df)} teams, {len(df.columns)} cols")

            # Surface useful columns we managed to extract
            interesting = [c for c in df.columns if any(
                k in c for k in ["mp", "gf", "ga", "ck", "crdy", "crdr", "xg", "fls", "poss"]
            )]
            if interesting:
                ok("found_cols", ", ".join(interesting[:10]))
        except Exception as e:
            fail(tn["tag"], str(e)[:120])

    if not all_rows:
        print("\nNo data collected. Verify FBRef URLs by visiting:")
        for tn in TOURNAMENTS:
            print(f"  https://fbref.com/en/comps/{tn['id']}/{tn['year']}/"
                  f"{tn['year']}-{tn['slug']}-Stats")
        return

    combined = pd.concat(all_rows, ignore_index=True, sort=False)
    out = RAW / "fbref_tournament_stats.csv"
    combined.to_csv(out, index=False)

    print("\n" + "=" * 60)
    print(f"Saved {len(combined)} rows × {len(combined.columns)} cols to {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
