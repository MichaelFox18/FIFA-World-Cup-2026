"""
01f_compute_player_form.py
Aggregates data/raw/goalscorers.csv into per-player goal-scoring form, then
enriches data/external/team_key_players.csv with recent-goals columns for
the T-2 review.

The Kaggle goalscorers dataset has 47k+ per-goal records dating back to
1916. We mostly care about the last 12 months as a "form" signal: a
star striker who hasn't scored for his country in a year is meaningfully
different from one who's bagging hat-tricks.

Output:
  data/processed/player_form.csv          -- per-(team, player) aggregation
  data/external/team_key_players_enriched.csv  -- watchlist + form columns

Run from worldcup2026/:  py scripts/01f_compute_player_form.py
"""

import re
import sys
import pandas as pd
import unicodedata
from pathlib import Path

# Windows console defaults to cp1252 which can't print "Ćirković" / "Mbappé"
# etc. Reconfigure stdout to UTF-8 so the debug prints don't crash.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, Exception):
    pass

RAW = Path("data/raw")
EXTERNAL = Path("data/external")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

# Look-back windows for "form" vs "baseline" goal counts.
TODAY = pd.Timestamp.today().normalize()
WINDOW_12MO = TODAY - pd.Timedelta(days=365)
WINDOW_24MO = TODAY - pd.Timedelta(days=730)


def section(t): print(f"\n-- {t} " + "-" * (60 - len(t)))
def ok(name, detail=""): print(f"  [OK]   {name:<28} {detail}")
def warn(name, detail=""): print(f"  [WARN] {name:<28} {detail}")


def load_name_map() -> dict:
    df = pd.read_csv(EXTERNAL / "team_name_map.csv")
    return dict(zip(df["variant"].astype(str).str.strip(),
                    df["canonical"].astype(str).str.strip()))


def normalize_team(name, name_map: dict) -> str:
    if pd.isna(name):
        return name
    s = str(name).strip()
    return name_map.get(s, s)


def strip_accents(s: str) -> str:
    """For fuzzy player-name matching: 'Mbappé' -> 'Mbappe'."""
    if pd.isna(s):
        return ""
    return "".join(
        c for c in unicodedata.normalize("NFD", str(s))
        if unicodedata.category(c) != "Mn"
    )


_SUFFIX_RE = re.compile(r"\b(jr|junior|sr|senior|iii|ii)\b\.?", re.IGNORECASE)


def player_key(name: str) -> str:
    """Lowercase, accent-stripped, alphanumeric-only -- robust join key.
    Strips Jr./Sr./Junior/Senior/II/III suffixes so 'Vinicius Jr.' and
    'Vinicius Junior' produce the same key."""
    cleaned = _SUFFIX_RE.sub("", strip_accents(name).lower())
    return re.sub(r"[^a-z0-9]", "", cleaned)


def fuzzy_match_per_team(watchlist_key: str, candidates: pd.DataFrame) -> int | None:
    """Given an unmatched watchlist key, look through this team's goalscorers
    for substring matches (either direction). Returns row index in candidates
    or None. Skips keys shorter than 4 chars to avoid false positives like
    'son' matching too eagerly."""
    if len(watchlist_key) < 4:
        return None
    for idx, c_key in candidates["_pkey"].items():
        if not isinstance(c_key, str) or len(c_key) < 4:
            continue
        if watchlist_key in c_key or c_key in watchlist_key:
            return idx
    return None


def main():
    print("=" * 60)
    print("01f_compute_player_form.py  |  goalscorers -> per-player form")
    print("=" * 60)

    section("Loading goalscorers")
    src = RAW / "goalscorers.csv"
    if not src.exists():
        warn("missing", f"{src} not found -- nothing to aggregate")
        return
    gs = pd.read_csv(src)
    gs["date"] = pd.to_datetime(gs["date"], format="mixed", errors="coerce")
    gs = gs.dropna(subset=["date", "team", "scorer"]).copy()
    ok("loaded", f"{len(gs):,} goal records, dates {gs['date'].min().date()} "
                  f"to {gs['date'].max().date()}")
    # Boolean coercion: source has TRUE/FALSE/True/False strings
    for c in ("own_goal", "penalty"):
        if c in gs.columns:
            gs[c] = gs[c].astype(str).str.lower().isin(("true", "1", "t", "yes"))

    section("Normalising team names")
    name_map = load_name_map()
    gs["team"] = gs["team"].map(lambda n: normalize_team(n, name_map))

    # Exclude own goals -- a goal you scored INTO YOUR OWN net should not count
    # as scoring form for "you". The opposing team gets the goal in our data
    # under their `team` column (the goal counts FOR them), so we filter on the
    # individual player record.
    gs = gs[~gs["own_goal"]].copy()

    section("Aggregating per (team, player)")
    # Slice windows
    g12 = gs[gs["date"] >= WINDOW_12MO]
    g24 = gs[gs["date"] >= WINDOW_24MO]
    ok("window_12mo", f"{len(g12):,} goals since {WINDOW_12MO.date()}")
    ok("window_24mo", f"{len(g24):,} goals since {WINDOW_24MO.date()}")

    agg12 = (g12.groupby(["team", "scorer"])
                .agg(goals_12mo=("scorer", "size"),
                     n_pens_12mo=("penalty", "sum"),
                     last_goal_date=("date", "max"))
                .reset_index())
    agg24 = (g24.groupby(["team", "scorer"])
                .agg(goals_24mo=("scorer", "size"))
                .reset_index())
    form = agg12.merge(agg24, on=["team", "scorer"], how="outer")
    form["goals_12mo"] = form["goals_12mo"].fillna(0).astype(int)
    form["goals_24mo"] = form["goals_24mo"].fillna(0).astype(int)
    form["n_pens_12mo"] = form["n_pens_12mo"].fillna(0).astype(int)
    form = form.rename(columns={"scorer": "player"})
    form = form.sort_values(["goals_12mo", "goals_24mo"], ascending=False)

    out = PROCESSED / "player_form.csv"
    form.to_csv(out, index=False)
    ok("player_form.csv", f"{len(form):,} (team, player) rows -> {out}")

    section("Top 15 scorers (last 12 months)")
    print(form.head(15)[["team", "player", "goals_12mo", "goals_24mo",
                          "n_pens_12mo"]].to_string(index=False))

    # ---- Enrich the key-player watchlist ----------------------------------
    section("Enriching key-player watchlist")
    kp_path = EXTERNAL / "team_key_players.csv"
    if not kp_path.exists():
        warn("missing", f"{kp_path} not found -- skipping enrichment")
        return
    kp = pd.read_csv(kp_path)
    ok("watchlist", f"{len(kp)} players across {kp['team'].nunique()} teams")

    # Build a join key on both sides
    kp["_pkey"] = kp["player"].map(player_key)
    form["_pkey"] = form["player"].map(player_key)

    # Pass 1: exact key match
    form_lookup = form[["team", "_pkey", "goals_12mo", "goals_24mo",
                         "n_pens_12mo", "last_goal_date"]]
    enriched = kp.merge(form_lookup, on=["team", "_pkey"], how="left")
    enriched["match_kind"] = enriched["goals_24mo"].notna().map(
        {True: "exact", False: "miss"}
    )

    # Pass 2: substring-based fuzzy match within the same team for misses
    n_fuzzy = 0
    for idx, row in enriched[enriched["match_kind"] == "miss"].iterrows():
        team_candidates = form_lookup[form_lookup["team"] == row["team"]].copy()
        if team_candidates.empty:
            continue
        match_idx = fuzzy_match_per_team(row["_pkey"], team_candidates)
        if match_idx is not None:
            matched_row = team_candidates.loc[match_idx]
            enriched.loc[idx, "goals_12mo"] = matched_row["goals_12mo"]
            enriched.loc[idx, "goals_24mo"] = matched_row["goals_24mo"]
            enriched.loc[idx, "n_pens_12mo"] = matched_row["n_pens_12mo"]
            enriched.loc[idx, "last_goal_date"] = matched_row["last_goal_date"]
            enriched.loc[idx, "match_kind"] = "fuzzy"
            n_fuzzy += 1

    enriched["name_matched"] = enriched["match_kind"].isin(["exact", "fuzzy"])
    enriched["goals_12mo"] = enriched["goals_12mo"].fillna(0).astype(int)
    enriched["goals_24mo"] = enriched["goals_24mo"].fillna(0).astype(int)
    enriched["n_pens_12mo"] = enriched["n_pens_12mo"].fillna(0).astype(int)
    enriched = enriched.drop(columns="_pkey")

    n_exact = int((enriched["match_kind"] == "exact").sum())
    ok("matched_exact", f"{n_exact}/{len(enriched)} via exact key")
    ok("matched_fuzzy", f"{n_fuzzy} additional via substring fallback")
    ok("matched_total", f"{int(enriched['name_matched'].sum())}/{len(enriched)} "
                         f"({100*enriched['name_matched'].mean():.0f}%)")

    # Sort by team, then weight + recent goals for visual scan
    enriched = enriched.sort_values(
        ["team", "importance_weight", "goals_12mo"],
        ascending=[True, False, False]
    )

    out_kp = EXTERNAL / "team_key_players_enriched.csv"
    enriched.to_csv(out_kp, index=False)
    ok("enriched_watchlist", f"{out_kp}")

    section("Sample of enriched entries (top 8 teams alphabetically)")
    sample = enriched.head(20)[["team", "player", "importance_weight",
                                 "goals_12mo", "goals_24mo", "name_matched"]]
    print(sample.to_string(index=False))

    section("Watchlist entries with NO match (review for typos / alt spellings)")
    unmatched = enriched[~enriched["name_matched"]]
    if len(unmatched) == 0:
        ok("clean", "all watchlist players found in goalscorers dataset")
    else:
        ok("unmatched_count", f"{len(unmatched)} entries")
        print(unmatched.head(20)[["team", "player",
                                    "importance_weight"]].to_string(index=False))
        if len(unmatched) > 20:
            print(f"  ... and {len(unmatched) - 20} more")

    print("\n" + "=" * 60)
    print("Player form aggregation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
