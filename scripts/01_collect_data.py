"""
01_collect_data.py
Pulls every external data source into data/raw/ and audits data/external/.
Run from worldcup2026/:  py scripts/01_collect_data.py
"""

import os
import json
import requests
import pandas as pd
from io import StringIO
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

RAW = Path("data/raw")
EXTERNAL = Path("data/external")
RAW.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def section(title: str) -> None:
    print(f"\n-- {title} " + "-" * (60 - len(title)))


def ok(name: str, detail: str = "") -> None:
    print(f"  [OK]   {name:<22} {detail}")


def miss(name: str, detail: str = "") -> None:
    print(f"  [MISS] {name:<22} {detail}")


def fail(name: str, detail: str = "") -> None:
    print(f"  [FAIL] {name:<22} {detail}")


def save(df: pd.DataFrame, filename: str, name: str, extra: str = "") -> None:
    path = RAW / filename
    df.to_csv(path, index=False)
    ok(name, f"{len(df):,} rows  ->  {filename}{('  ' + extra) if extra else ''}")


# -- 1. Competition + Kaggle files (manually placed) --------------------------

def check_raw_inputs() -> None:
    section("Manually-placed raw files")
    expected = {
        "group_fixtures.csv":  ("DataLab: 72 group stage fixtures",                72),
        "knockout_slots.csv":  ("DataLab: 32 knockout slots + multipliers",        32),
        "results.csv":         ("Kaggle: international match results",            None),
        "goalscorers.csv":     ("Kaggle: per-goal detail",                        None),
        "shootouts.csv":       ("Kaggle: penalty shootout history",               None),
        "former_names.csv":    ("Kaggle: team name changes",                      None),
    }
    for filename, (description, expected_rows) in expected.items():
        path = RAW / filename
        if path.exists():
            rows = sum(1 for _ in open(path, encoding="utf-8", errors="ignore")) - 1
            if expected_rows and rows != expected_rows:
                fail(filename, f"only {rows} rows -- expected {expected_rows}!  {description}")
            else:
                ok(filename, f"{rows:,} rows -- {description}")
        else:
            miss(filename, description)


# -- 2. FIFA rankings: combine all snapshot files ---------------------------

def consolidate_fifa_rankings() -> None:
    section("FIFA rankings  (consolidating snapshots)")
    snapshots = sorted(RAW.glob("fifa_ranking-*.csv"))
    if not snapshots:
        miss("fifa_ranking", "no fifa_ranking-*.csv files found in data/raw/")
        return
    frames = []
    for snap in snapshots:
        date_str = snap.stem.replace("fifa_ranking-", "")
        df = pd.read_csv(snap)
        df["snapshot_date"] = pd.to_datetime(date_str, errors="coerce")
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    save(combined, "fifa_ranking_combined.csv", "fifa_ranking",
         extra=f"from {len(snapshots)} snapshots, latest {snapshots[-1].stem.replace('fifa_ranking-', '')}")


# -- 3. Current FIFA rankings from Wikipedia (most up-to-date public source) --

def fetch_current_fifa_rankings() -> None:
    section("Current FIFA rankings  (Wikipedia)")
    try:
        url = "https://en.wikipedia.org/wiki/FIFA_Men%27s_World_Ranking"
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))

        def looks_like_rankings(t: pd.DataFrame) -> bool:
            cols = [str(c).lower() for c in t.columns]
            has_rank = any(("rank" in c or "position" in c or c == "#") for c in cols)
            has_team = any(("team" in c or "nation" in c or "country" in c) for c in cols)
            has_points = any(("points" in c or "pts" in c) for c in cols)
            return has_rank and has_team and has_points and len(t) >= 10

        candidates = [t for t in tables if looks_like_rankings(t)]
        if not candidates:
            raise ValueError(
                f"No rankings table found among {len(tables)} tables on page"
            )
        df = max(candidates, key=len)
        df["scraped_on"] = datetime.now().date().isoformat()
        save(df, "fifa_ranking_current.csv", "fifa_current",
             extra=f"top {len(df)} teams")
    except Exception as e:
        fail("fifa_current", str(e)[:120])


# -- 4. Polymarket outright odds (free public API) ----------------------------

def fetch_polymarket() -> None:
    section("Polymarket  (gamma-api.polymarket.com)")
    try:
        rows = []

        # Polymarket has reorganised the WC 2026 surface over time:
        #   * Early on, championship outrights were grouped under a single
        #     "FIFA World Cup 2026 Winner" event in /events.
        #   * Later they got split into individual "Will X win the 2026 FIFA
        #     World Cup?" markets that live in /markets, not /events.
        # We fetch BOTH and dedupe -- this stays robust if Polymarket reverts.

        # Path A: /events with WC 2026 in title/slug
        evt_resp = requests.get("https://gamma-api.polymarket.com/events",
                                params={"limit": 200, "active": "true", "closed": "false"},
                                headers=HEADERS, timeout=30)
        evt_resp.raise_for_status()
        events = evt_resp.json()
        wc_events = [
            e for e in events
            if "world cup" in (e.get("title", "") + e.get("slug", "")).lower()
            and "2026" in (e.get("title", "") + e.get("slug", ""))
        ]
        for ev in wc_events:
            for m in ev.get("markets", []):
                outcomes = json.loads(m.get("outcomes", "[]"))
                prices = json.loads(m.get("outcomePrices", "[]"))
                for outcome, price in zip(outcomes, prices):
                    rows.append({
                        "event":     ev.get("title", ""),
                        "market":    m.get("question", ""),
                        "outcome":   outcome,
                        "price":     float(price) if price else None,
                        "volume":    m.get("volume", 0),
                        "liquidity": m.get("liquidity", 0),
                        "end_date":  m.get("endDate", ""),
                    })

        # Path B: /markets directly for "Will X win the 2026 FIFA World Cup?"
        # These are individual binary markets; the championship outrights
        # currently live here, not in /events.
        mkt_resp = requests.get("https://gamma-api.polymarket.com/markets",
                                params={"limit": 500, "active": "true", "closed": "false"},
                                headers=HEADERS, timeout=30)
        mkt_resp.raise_for_status()
        markets = mkt_resp.json()
        wc_markets = [
            m for m in markets
            if "world cup" in m.get("question", "").lower()
            and "2026" in m.get("question", "")
            and "win" in m.get("question", "").lower()
        ]
        for m in wc_markets:
            outcomes = json.loads(m.get("outcomes", "[]"))
            prices = json.loads(m.get("outcomePrices", "[]"))
            for outcome, price in zip(outcomes, prices):
                rows.append({
                    "event":     "FIFA World Cup 2026 Championship",
                    "market":    m.get("question", ""),
                    "outcome":   outcome,
                    "price":     float(price) if price else None,
                    "volume":    m.get("volume", 0),
                    "liquidity": m.get("liquidity", 0),
                    "end_date":  m.get("endDate", ""),
                })

        if not rows:
            fail("polymarket", "no WC 2026 markets found via /events or /markets")
            return

        df = pd.DataFrame(rows).drop_duplicates(subset=["market", "outcome"], keep="first")
        save(df, "polymarket.csv", "polymarket",
             extra=f"{len(wc_events)} events + {len(wc_markets)} standalone markets")
    except Exception as e:
        fail("polymarket", str(e))


# -- 5. The Odds API h2h match odds -------------------------------------------

def fetch_odds() -> None:
    section("The Odds API  (the-odds-api.com)")
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        miss("odds_api", "ODDS_API_KEY not set in .env")
        return
    try:
        url = "https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup/odds/"
        params = {
            "apiKey":     api_key,
            "regions":    "eu",
            # h2h = moneyline (winner / draw / loser)
            # spreads = Asian handicap (home -3.5 etc.) -> expected goal diff
            # totals = over/under total goals (4.5) -> expected total goals
            # Together these triangulate match-level Poisson lambdas straight
            # from bookmaker consensus, much sharper than h2h alone for
            # blowout matchups where the favourite piles on.
            "markets":    "h2h,spreads,totals",
            "oddsFormat": "decimal",
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            miss("odds_api", "No WC 2026 match odds available yet -- re-run closer to tournament")
            return

        rows = []
        for match in data:
            for bk in match.get("bookmakers", []):
                for mkt in bk.get("markets", []):
                    for o in mkt.get("outcomes", []):
                        rows.append({
                            "match_id":      match["id"],
                            "home_team":     match["home_team"],
                            "away_team":     match["away_team"],
                            "commence_time": match["commence_time"],
                            "bookmaker":     bk["key"],
                            "market":        mkt["key"],
                            "outcome":       o["name"],
                            "price":         o["price"],
                            # point = the spread/total line (e.g. -3.5, 4.5).
                            # Null for h2h outcomes.
                            "point":         o.get("point"),
                        })

        df = pd.DataFrame(rows)
        remaining = resp.headers.get("x-requests-remaining", "?")
        n_h2h = (df["market"] == "h2h").sum()
        n_spr = (df["market"] == "spreads").sum()
        n_tot = (df["market"] == "totals").sum()
        save(df, "odds_h2h.csv", "odds_api",
             extra=f"{remaining} requests left; rows: h2h={n_h2h}, "
                   f"spreads={n_spr}, totals={n_tot}")
    except Exception as e:
        fail("odds_api", str(e))


# -- 6. External lookup tables (committed) ------------------------------------

def check_external() -> None:
    section("External lookup tables (data/external/)")
    for filename in ["venues.csv", "uefa_playoff_qualifiers.csv", "team_name_map.csv"]:
        path = EXTERNAL / filename
        if path.exists():
            df = pd.read_csv(path)
            ok(filename, f"{len(df):,} rows")
        else:
            miss(filename, "expected in data/external/")


# -- Main ---------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("01_collect_data.py  |  FIFA World Cup 2026 Prediction Model")
    print("=" * 60)

    check_raw_inputs()
    check_external()
    consolidate_fifa_rankings()
    fetch_current_fifa_rankings()
    fetch_polymarket()
    fetch_odds()

    print("\n" + "=" * 60)
    print("Done. Review any [MISS]/[FAIL] entries above.")
    print("Raw files in data/raw/  |  Lookups in data/external/")
    print("=" * 60)
