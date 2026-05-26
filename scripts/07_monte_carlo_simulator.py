"""
07_monte_carlo_simulator.py
Simulates the WC 2026 tournament N_SIMS times to derive probabilistic
knockout matchups, qualifier probabilities, and championship odds.

For each iteration:
  1. Sample (home_goals, away_goals) for all 72 group matches from Dixon-Coles
  2. Compute standings with FIFA tiebreakers (P > GD > GF > random)
  3. Determine qualifiers: 12 group winners, 12 runners-up, 8 best 3rds
  4. Resolve qualifiers into 32 knockout slots
  5. Simulate every knockout match (90 min, ET if tied, penalties if still tied)
  6. Track matchups, winners, penalty rates

Run from worldcup2026/:  py scripts/07_monte_carlo_simulator.py

Outputs:
  data/processed/mc_qualifiers.csv  -- per-team probabilities (round reached, champion)
  data/processed/mc_knockout.csv    -- per-slot most likely matchup, winner, p_penalties
"""

import re
import math
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter

PROCESSED = Path("data/processed")
MODELS = Path("models")

N_SIMS = 10000
SEED = 42

EXTRA_TIME_LAMBDA_FACTOR = 1 / 3
PENALTY_HOME_BIAS = 0.0


def section(t): print(f"\n-- {t} " + "-" * (60 - len(t)))
def ok(name, detail=""): print(f"  [OK]   {name:<26} {detail}")
def warn(name, detail=""): print(f"  [WARN] {name:<26} {detail}")


# ── Setup ─────────────────────────────────────────────────────────────────────

def build_groups(group_fix: pd.DataFrame) -> dict:
    groups = defaultdict(list)
    for _, r in group_fix.iterrows():
        for t in (r["home_team"], r["away_team"]):
            if t not in groups[r["group"]]:
                groups[r["group"]].append(t)
    return dict(groups)


def build_group_lambdas(group_lambdas: pd.DataFrame) -> dict:
    out = {}
    for _, r in group_lambdas.iterrows():
        out[(r["home_team"], r["away_team"])] = (float(r["lambda_home"]),
                                                 float(r["lambda_away"]))
    return out


def precompute_knockout_lambdas(all_teams: list, gm: dict) -> tuple:
    """Precompute Dixon-Coles lambdas for every (home, away) pair (neutral venue)."""
    n = len(all_teams)
    team_to_idx = {t: i for i, t in enumerate(all_teams)}
    lh = np.zeros((n, n))
    la = np.zeros((n, n))
    alpha, attack, defense = gm["alpha"], gm["attack"], gm["defense"]
    for i, h in enumerate(all_teams):
        if h not in gm["team_to_idx"]:
            continue
        hi = gm["team_to_idx"][h]
        for j, a in enumerate(all_teams):
            if i == j or a not in gm["team_to_idx"]:
                continue
            ai = gm["team_to_idx"][a]
            lh[i, j] = math.exp(alpha + attack[hi] - defense[ai])
            la[i, j] = math.exp(alpha + attack[ai] - defense[hi])
    return lh, la, team_to_idx


# ── Group stage ───────────────────────────────────────────────────────────────

def simulate_group_stage(group_fixtures: pd.DataFrame,
                         lambda_lookup: dict,
                         rng: np.random.Generator) -> list:
    out = []
    for _, r in group_fixtures.iterrows():
        lh, la = lambda_lookup.get((r["home_team"], r["away_team"]), (1.0, 1.0))
        hg = int(rng.poisson(lh))
        ag = int(rng.poisson(la))
        out.append({"group": r["group"], "home": r["home_team"],
                    "away": r["away_team"], "hg": hg, "ag": ag})
    return out


def compute_standings(groups: dict, group_results: list,
                      rng: np.random.Generator) -> dict:
    standings = {}
    for grp, teams in groups.items():
        stats = {t: {"P": 0, "GD": 0, "GF": 0} for t in teams}
        for r in group_results:
            if r["group"] != grp:
                continue
            h, a, hg, ag = r["home"], r["away"], r["hg"], r["ag"]
            stats[h]["GF"] += hg
            stats[h]["GD"] += hg - ag
            stats[a]["GF"] += ag
            stats[a]["GD"] += ag - hg
            if hg > ag:
                stats[h]["P"] += 3
            elif ag > hg:
                stats[a]["P"] += 3
            else:
                stats[h]["P"] += 1
                stats[a]["P"] += 1
        sorted_teams = sorted(
            teams,
            key=lambda t: (-stats[t]["P"], -stats[t]["GD"], -stats[t]["GF"], rng.random())
        )
        standings[grp] = [(t, stats[t]) for t in sorted_teams]
    return standings


def determine_qualifiers(standings: dict) -> dict:
    qualifiers = {}
    third_place_teams = []
    for grp, sorted_list in standings.items():
        qualifiers[f"W-{grp}"] = sorted_list[0][0]
        qualifiers[f"RU-{grp}"] = sorted_list[1][0]
        team, stats = sorted_list[2]
        third_place_teams.append((grp, team, stats["P"], stats["GD"], stats["GF"]))
    third_place_teams.sort(key=lambda x: (-x[2], -x[3], -x[4]))
    best_thirds = third_place_teams[:8]
    qualified_third_groups = set()
    for grp, team, *_ in best_thirds:
        qualifiers[f"3-{grp}"] = team
        qualified_third_groups.add(grp)
    qualifiers["__best_third_groups__"] = qualified_third_groups
    return qualifiers


# ── Knockout ──────────────────────────────────────────────────────────────────

_RE_WINNER_GROUP = re.compile(r"Winner Group (\w)")
_RE_RUNNERUP    = re.compile(r"Runner-up Group (\w)")
_RE_BEST_3RD    = re.compile(r"Best 3rd \(Groups ([\w/]+)\)")
_RE_WIN_MATCH   = re.compile(r"Winner Match (\d+)")
_RE_LOSE_MATCH  = re.compile(r"Loser Match (\d+)")


def resolve_slot(descriptor, qualifiers, results, best_third_assigned) -> str:
    s = str(descriptor).strip()
    if m := _RE_WINNER_GROUP.match(s):
        return qualifiers.get(f"W-{m.group(1)}", "TBD")
    if m := _RE_RUNNERUP.match(s):
        return qualifiers.get(f"RU-{m.group(1)}", "TBD")
    if m := _RE_BEST_3RD.match(s):
        allowed = m.group(1).split("/")
        for g in allowed:
            if g in qualifiers.get("__best_third_groups__", set()):
                t = qualifiers[f"3-{g}"]
                if t not in best_third_assigned:
                    best_third_assigned.add(t)
                    return t
        # Fallback: any qualifying 3rd not yet assigned
        for k, v in qualifiers.items():
            if k.startswith("3-") and not k.startswith("__") and v not in best_third_assigned:
                best_third_assigned.add(v)
                return v
        return "TBD"
    if m := _RE_WIN_MATCH.match(s):
        return results.get(int(m.group(1)), {}).get("winner", "TBD")
    if m := _RE_LOSE_MATCH.match(s):
        return results.get(int(m.group(1)), {}).get("loser", "TBD")
    return "TBD"


def simulate_knockout_match(home, away, lh_table, la_table, team_to_idx, rng):
    if home not in team_to_idx or away not in team_to_idx:
        hg = int(rng.poisson(1.2))
        ag = int(rng.poisson(1.2))
    else:
        hi, ai = team_to_idx[home], team_to_idx[away]
        hg = int(rng.poisson(lh_table[hi, ai]))
        ag = int(rng.poisson(la_table[hi, ai]))

    if hg != ag:
        return hg, ag, (home if hg > ag else away), False

    # Extra time
    if home in team_to_idx and away in team_to_idx:
        hi, ai = team_to_idx[home], team_to_idx[away]
        et_lh = lh_table[hi, ai] * EXTRA_TIME_LAMBDA_FACTOR
        et_la = la_table[hi, ai] * EXTRA_TIME_LAMBDA_FACTOR
    else:
        et_lh = et_la = 0.4
    hg += int(rng.poisson(et_lh))
    ag += int(rng.poisson(et_la))

    if hg != ag:
        return hg, ag, (home if hg > ag else away), False

    winner = home if rng.random() < 0.5 + PENALTY_HOME_BIAS else away
    return hg, ag, winner, True


def simulate_knockout_bracket(knockout, qualifiers, lh_table, la_table,
                              team_to_idx, rng):
    results = {}
    best_third_assigned = set()
    for _, row in knockout.iterrows():
        mid = int(row["match_id"])
        home = resolve_slot(row["slot_home"], qualifiers, results, best_third_assigned)
        away = resolve_slot(row["slot_away"], qualifiers, results, best_third_assigned)
        hg, ag, winner, pens = simulate_knockout_match(
            home, away, lh_table, la_table, team_to_idx, rng
        )
        loser = away if winner == home else home
        results[mid] = {
            "home": home, "away": away,
            "home_goals": hg, "away_goals": ag,
            "winner": winner, "loser": loser, "penalties": pens,
            "round": row["round"], "multiplier": row["multiplier"],
        }
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(f"07_monte_carlo_simulator.py  |  N_SIMS = {N_SIMS:,}")
    print("=" * 60)

    section("Loading inputs")
    group_fix = pd.read_csv(PROCESSED / "fixtures_group.csv")
    knockout = pd.read_csv(PROCESSED / "fixtures_knockout.csv").sort_values("match_id")
    group_lambdas = pd.read_csv(PROCESSED / "group_lambdas.csv")
    with open(MODELS / "goals_model.pkl", "rb") as f:
        gm = pickle.load(f)
    ok("group_fixtures", f"{len(group_fix)} matches")
    ok("knockout_slots", f"{len(knockout)} matches")
    ok("group_lambdas", f"{len(group_lambdas)} fixtures with lambdas")
    ok("goals_model", f"alpha={gm['alpha']:.3f}, gamma={gm['gamma']:.3f}, rho={gm['rho']:.3f}")

    groups = build_groups(group_fix)
    lambda_lookup = build_group_lambdas(group_lambdas)

    all_wc_teams = sorted(set(group_fix["home_team"]) | set(group_fix["away_team"]))
    lh_table, la_table, team_to_idx = precompute_knockout_lambdas(all_wc_teams, gm)
    missing = [t for t in all_wc_teams if t not in gm["team_to_idx"]]
    if missing:
        warn("teams_not_in_goals_model", str(missing))
    ok("knockout_lambdas", f"precomputed {len(all_wc_teams)}x{len(all_wc_teams)} table")

    final_match_id = int(knockout[knockout["round"] == "Final"]["match_id"].iloc[0])
    ok("final_match_id", str(final_match_id))

    section(f"Running {N_SIMS:,} simulations")
    qualify_count = Counter()
    group_winner_count = Counter()
    runner_up_count = Counter()
    matchup_count = defaultdict(Counter)
    home_in_slot = defaultdict(Counter)
    away_in_slot = defaultdict(Counter)
    winner_count = defaultdict(Counter)
    penalties_count = Counter()
    round_reached_count = defaultdict(Counter)
    champion_count = Counter()

    for sim in range(N_SIMS):
        rng = np.random.default_rng(SEED + sim)
        group_results = simulate_group_stage(group_fix, lambda_lookup, rng)
        standings = compute_standings(groups, group_results, rng)
        qualifiers = determine_qualifiers(standings)

        for grp, sorted_list in standings.items():
            group_winner_count[sorted_list[0][0]] += 1
            runner_up_count[sorted_list[1][0]] += 1
        for k, v in qualifiers.items():
            if k.startswith("__"):
                continue
            if isinstance(v, str):
                qualify_count[v] += 1

        ko = simulate_knockout_bracket(knockout, qualifiers,
                                       lh_table, la_table, team_to_idx, rng)
        for mid, info in ko.items():
            matchup_count[mid][(info["home"], info["away"])] += 1
            home_in_slot[mid][info["home"]] += 1
            away_in_slot[mid][info["away"]] += 1
            winner_count[mid][info["winner"]] += 1
            if info["penalties"]:
                penalties_count[mid] += 1
            round_reached_count[info["winner"]][info["round"]] += 1

        champion = ko[final_match_id]["winner"]
        champion_count[champion] += 1

        if (sim + 1) % max(1, N_SIMS // 10) == 0:
            print(f"  ... {sim + 1:,} / {N_SIMS:,}")

    section("Aggregating qualifier probabilities")
    qual_rows = []
    for t in all_wc_teams:
        qual_rows.append({
            "team": t,
            "p_qualify_ko": qualify_count[t] / N_SIMS,
            "p_group_winner": group_winner_count[t] / N_SIMS,
            "p_runner_up": runner_up_count[t] / N_SIMS,
            "p_reach_r16": round_reached_count[t]["Round of 32"] / N_SIMS,
            "p_reach_qf": round_reached_count[t]["Round of 16"] / N_SIMS,
            "p_reach_sf": round_reached_count[t]["Quarter-final"] / N_SIMS,
            "p_reach_final": round_reached_count[t]["Semi-final"] / N_SIMS,
            "p_champion": champion_count[t] / N_SIMS,
        })
    qual_df = pd.DataFrame(qual_rows).sort_values("p_champion", ascending=False)
    qual_df.to_csv(PROCESSED / "mc_qualifiers.csv", index=False)
    ok("mc_qualifiers.csv", f"{len(qual_df)} teams")

    section("Top 10 championship probabilities")
    print(qual_df.head(10)[[
        "team", "p_qualify_ko", "p_reach_qf", "p_reach_sf", "p_champion"
    ]].round(3).to_string(index=False))

    section("Aggregating per-slot knockout predictions")
    ko_rows = []
    for _, row in knockout.iterrows():
        mid = int(row["match_id"])
        top_home, _ = home_in_slot[mid].most_common(1)[0]
        top_away, _ = away_in_slot[mid].most_common(1)[0]
        top_match, top_match_count = matchup_count[mid].most_common(1)[0]
        winner_team, win_n = winner_count[mid].most_common(1)[0]
        ko_rows.append({
            "match_id": mid,
            "round": row["round"],
            "multiplier": row["multiplier"],
            "slot_home_desc": row["slot_home"],
            "slot_away_desc": row["slot_away"],
            "pred_home_team": top_home,
            "pred_away_team": top_away,
            "p_matchup_exact": top_match_count / N_SIMS,
            "top_pair_match": f"{top_match[0]} vs {top_match[1]}",
            "pred_winner": winner_team,
            "p_pred_winner": win_n / N_SIMS,
            "p_penalties": penalties_count[mid] / N_SIMS,
        })
    ko_df = pd.DataFrame(ko_rows)
    ko_df.to_csv(PROCESSED / "mc_knockout.csv", index=False)
    ok("mc_knockout.csv", f"{len(ko_df)} slots")

    # Save the full matchup-count detail per slot so script 08 can use
    # alternatives to enforce bracket consistency (e.g. 3rd-place teams
    # cannot overlap with Final teams).
    detail_rows = []
    for mid, counter in matchup_count.items():
        for (h, a), n in counter.items():
            detail_rows.append({
                "match_id": mid, "home_team": h, "away_team": a,
                "count": n, "prob": n / N_SIMS
            })
    detail_df = pd.DataFrame(detail_rows).sort_values(
        ["match_id", "count"], ascending=[True, False]
    )
    detail_df.to_csv(PROCESSED / "mc_matchup_details.csv", index=False)
    ok("mc_matchup_details.csv", f"{len(detail_df)} matchup candidates")

    section("Sample knockout predictions (first 5 R32 matches)")
    show = ko_df.head(5)[[
        "match_id", "round", "pred_home_team", "pred_away_team",
        "p_matchup_exact", "pred_winner", "p_pred_winner", "p_penalties"
    ]].round(3)
    print(show.to_string(index=False))

    section("Predicted Final + Third-place playoff")
    final_rows = ko_df[ko_df["round"].isin(["Final", "Third-place playoff"])]
    print(final_rows[[
        "match_id", "round", "pred_home_team", "pred_away_team",
        "p_matchup_exact", "pred_winner", "p_pred_winner", "p_penalties"
    ]].round(3).to_string(index=False))

    print("\n" + "=" * 60)
    print("Monte Carlo simulation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
