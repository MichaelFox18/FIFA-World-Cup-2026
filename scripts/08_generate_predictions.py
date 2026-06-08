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
import re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy.stats import poisson
from scipy.optimize import brentq

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

# Blend weight when combining Dixon-Coles + bookmaker consensus for the
# winning_team prediction. 1.0 = pure model, 0.0 = pure market. The market is
# usually sharper at the top of the bracket but noisier on long-shots.
MARKET_BLEND_WEIGHT = 0.25       # 75% model, 25% market

# Separate blend weight for the GOALS-LAMBDA blend (totals + spreads markets).
# Dixon-Coles systematically underpredicts blowouts because Poisson with a
# capped lambda undercounts how much a much-stronger team will pile on.
# Bookmaker totals + spreads incorporate the right tail. 50/50 split per
# user's call on 2026-06-02.
LAMBDA_BLEND_WEIGHT = 0.5        # 50% model, 50% market

# Blend weight for the MANUAL corners/cards markets (manual_cards_corners_odds.csv).
# The Odds API carries no corner/card markets, so these hand-collected bookmaker
# totals are our only market signal for the two weakest models. Applied to group
# fixtures only (knockout matchups aren't known pre-tournament). 0.5 = even split.
MANUAL_ODDS_BLEND_WEIGHT = 0.5


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


def dc_winner_neutral(home, away, gm, h2h_lookup=None):
    """Deterministic Dixon-Coles winner at a neutral venue: whichever team has
    the larger expected goals. Reused by predict_score_neutral and the bracket
    reconciler so a chosen matchup yields the same winner everywhere."""
    team_to_idx = gm["team_to_idx"]
    if home not in team_to_idx or away not in team_to_idx:
        return home
    hi, ai = team_to_idx[home], team_to_idx[away]
    alpha, attack, defense = gm["alpha"], gm["attack"], gm["defense"]
    delta_h2h = float(gm.get("delta", 0.0))
    h2h_ha = float((h2h_lookup or {}).get((home, away), 0.0))
    lh = math.exp(alpha + attack[hi] - defense[ai] + delta_h2h * h2h_ha)
    la = math.exp(alpha + attack[ai] - defense[hi] - delta_h2h * h2h_ha)
    return home if lh >= la else away


_SLOT_DEP_RE = re.compile(r"^(Winner|Loser) Match (\d+)$")


def parse_bracket_dependencies(knockout_fixtures: pd.DataFrame) -> dict:
    """Map match_id -> (home_dep, away_dep), where each dep is either None
    (group-stage slot like 'Winner Group A') or a tuple (kind, parent_match_id)
    such as ('Winner', 73) or ('Loser', 101)."""
    deps = {}
    for _, r in knockout_fixtures.iterrows():
        mid = int(r["match_id"])
        h_match = _SLOT_DEP_RE.match(str(r["slot_home"]))
        a_match = _SLOT_DEP_RE.match(str(r["slot_away"]))
        deps[mid] = (
            (h_match.group(1), int(h_match.group(2))) if h_match else None,
            (a_match.group(1), int(a_match.group(2))) if a_match else None,
        )
    return deps


def reconcile_bracket(matchup_details: pd.DataFrame, mc_knockout: pd.DataFrame,
                      knockout_fixtures: pd.DataFrame, gm: dict,
                      h2h_lookup: dict | None) -> tuple[dict, list[str]]:
    """Top-down propagation: pick the Final matchup, then for each upstream
    round, pick the matchup whose Dixon-Coles winner matches the required
    downstream input. Returns (chosen, log) where chosen[match_id] = (home, away)
    and log is a list of human-readable notes.

    Enforces two invariants:
      1. Each match's DC winner must equal the team needed in the downstream slot.
      2. Within a round, each team appears in at most one matchup (no team can be
         in two semis, two QFs, etc., at the same level)."""
    deps = parse_bracket_dependencies(knockout_fixtures)
    round_of = dict(zip(knockout_fixtures["match_id"], knockout_fixtures["round"]))
    chosen: dict[int, tuple[str, str]] = {}
    required_winner: dict[int, str] = {}
    # Teams already assigned to ANY slot in a given round (e.g., a semi-final
    # team can't also be the other semi's loser). Built up as we go.
    assigned_in_round: dict[str, set[str]] = {}
    log: list[str] = []

    def pick_top_for(mid: int) -> tuple[str, str] | None:
        sub = matchup_details[matchup_details["match_id"] == mid]
        if sub.empty:
            row = mc_knockout[mc_knockout["match_id"] == mid]
            if row.empty:
                return None
            r = row.iloc[0]
            return (r["pred_home_team"], r["pred_away_team"])
        top = sub.sort_values("count", ascending=False).iloc[0]
        return (top["home_team"], top["away_team"])

    def _other_round_assignments(mid: int) -> set[str]:
        """Teams committed elsewhere in mid's round -- candidates for mid must
        not include any of these."""
        r = round_of.get(mid)
        return assigned_in_round.get(r, set())

    def pick_with_winner(mid: int, required: str) -> tuple[str, str] | None:
        """Pick the highest-count matchup whose Dixon-Coles winner matches
        `required` AND whose other team isn't already committed to this round.
        Falls back progressively: relax the round-uniqueness constraint first,
        then accept any matchup containing the required team."""
        sub = matchup_details[matchup_details["match_id"] == mid].sort_values(
            "count", ascending=False
        )
        if sub.empty:
            return pick_top_for(mid)
        forbidden = _other_round_assignments(mid)
        # 1) winner match AND round-unique
        for _, c in sub.iterrows():
            h, a = c["home_team"], c["away_team"]
            if h in forbidden or a in forbidden:
                continue
            if dc_winner_neutral(h, a, gm, h2h_lookup) == required:
                return (h, a)
        # 2) winner match (ignore round-uniqueness if no candidate satisfies it)
        for _, c in sub.iterrows():
            if dc_winner_neutral(c["home_team"], c["away_team"], gm, h2h_lookup) == required:
                log.append(f"match {mid}: best winner-match conflicts with already-"
                           f"assigned round teams {sorted(forbidden)}; accepting "
                           f"{c['home_team']} vs {c['away_team']} anyway")
                return (c["home_team"], c["away_team"])
        # 3) team appears at all (round-unique preferred)
        for _, c in sub.iterrows():
            h, a = c["home_team"], c["away_team"]
            if (h == required or a == required) and h not in forbidden and a not in forbidden:
                log.append(f"match {mid}: required winner {required!r} never wins in MC, "
                           f"took round-unique matchup containing them: {h} vs {a}")
                return (h, a)
        # 4) absolute fallback
        top = sub.iloc[0]
        log.append(f"match {mid}: required team {required!r} not in MC's matchups, "
                   f"took global top: {top['home_team']} vs {top['away_team']}")
        return (top["home_team"], top["away_team"])

    def commit(mid: int, pick: tuple[str, str]):
        """Record the chosen matchup AND mark its teams as committed for the
        round so later picks in the same round can't reuse them."""
        chosen[mid] = pick
        r = round_of.get(mid)
        if r is not None:
            assigned_in_round.setdefault(r, set()).update(pick)

    final_mid_row = knockout_fixtures[knockout_fixtures["round"] == "Final"]
    if final_mid_row.empty:
        return chosen, ["no Final match found in fixtures"]
    final_mid = int(final_mid_row.iloc[0]["match_id"])

    # Step 1: Final picks freely
    pick = pick_top_for(final_mid)
    if pick is None:
        return chosen, ["matchup_details empty for Final"]
    commit(final_mid, pick)
    log.append(f"Final ({final_mid}): {pick[0]} vs {pick[1]}")

    # Propagate constraints
    home_dep, away_dep = deps[final_mid]
    if home_dep and home_dep[0] == "Winner":
        required_winner[home_dep[1]] = pick[0]
    if away_dep and away_dep[0] == "Winner":
        required_winner[away_dep[1]] = pick[1]

    # Step 2: BFS down the bracket (semis -> QFs -> R16 -> R32). Process by
    # descending match_id so each match is handled after the rounds that
    # depend on it. Skip 3rd-place -- handled separately after Semis are set.
    third_row = knockout_fixtures[knockout_fixtures["round"] == "Third-place playoff"]
    third_mid = int(third_row.iloc[0]["match_id"]) if not third_row.empty else None

    order = sorted(deps.keys(), reverse=True)
    for mid in order:
        if mid in chosen or mid == third_mid:
            continue
        req = required_winner.get(mid)
        pick = pick_with_winner(mid, req) if req else pick_top_for(mid)
        if pick is None:
            continue
        commit(mid, pick)
        # Propagate constraints upward (to earlier rounds)
        h_dep, a_dep = deps[mid]
        if h_dep and h_dep[0] == "Winner":
            required_winner.setdefault(h_dep[1], pick[0])
        if a_dep and a_dep[0] == "Winner":
            required_winner.setdefault(a_dep[1], pick[1])

    # Step 3: 3rd-place uses losers of the two Semis (whose matchups are now fixed)
    if third_mid is not None:
        h_dep, a_dep = deps[third_mid]
        third_home = third_away = None
        if h_dep and h_dep[0] == "Loser" and h_dep[1] in chosen:
            ph, pa = chosen[h_dep[1]]
            w = dc_winner_neutral(ph, pa, gm, h2h_lookup)
            third_home = pa if w == ph else ph
        if a_dep and a_dep[0] == "Loser" and a_dep[1] in chosen:
            ph, pa = chosen[a_dep[1]]
            w = dc_winner_neutral(ph, pa, gm, h2h_lookup)
            third_away = pa if w == ph else ph
        if third_home and third_away:
            commit(third_mid, (third_home, third_away))
            log.append(f"3rd-place ({third_mid}): {third_home} vs {third_away} "
                       f"(derived from Semi losers)")

    return chosen, log


def reconcile_bracket_bottom_up(matchup_details, mc_knockout, knockout_fixtures,
                                gm, h2h_lookup):
    """Bottom-up deterministic bracket reconciliation.

    The MC sim emits per-slot *marginal* modes (most common home, away and
    winner). Each is individually reasonable but they are jointly inconsistent: a
    slot's modal winner need not appear in its modal matchup, and one team can be
    the modal occupant of two different slots. The old top-down reconciler tried to
    repair this by forcing each match's Dixon-Coles winner to equal the team needed
    downstream, which cascaded into *dropping* strong sides -- England, France and
    Netherlands all vanished from the 2026 knockouts despite ~84-93% qualify odds.

    This version is bottom-up and consistent by construction:
      1. Seed the Round of 32 from the MC per-slot modal matchups, de-duplicated so
         all 32 participants are distinct (greedy, most-confident position first).
      2. Play every later round upward: each match's two teams are the Dixon-Coles
         winners of its feeder matches; the 3rd-place match takes the semi losers.
    A team therefore advances only where the model favours it, and no team can
    occupy two slots in the same round.

    Returns (chosen, log) where chosen[match_id] = (home, away) for all matches.
    """
    deps = parse_bracket_dependencies(knockout_fixtures)
    log: list[str] = []

    mc_home = dict(zip(mc_knockout["match_id"], mc_knockout["pred_home_team"]))
    mc_away = dict(zip(mc_knockout["match_id"], mc_knockout["pred_away_team"]))

    # Per-slot occupant counts (marginalised from the matchup distribution).
    home_counts: dict[int, pd.Series] = {}
    away_counts: dict[int, pd.Series] = {}
    if matchup_details is not None:
        for mid, sub in matchup_details.groupby("match_id"):
            home_counts[int(mid)] = (sub.groupby("home_team")["count"].sum()
                                        .sort_values(ascending=False))
            away_counts[int(mid)] = (sub.groupby("away_team")["count"].sum()
                                        .sort_values(ascending=False))

    r32_ids = sorted(m for m in deps if deps[m] == (None, None))

    # --- 1. Seed R32 with globally-distinct occupants ----------------------
    # Process the 32 positions in descending confidence (top candidate's share of
    # the slot) so the most certain slots claim their team before contested ones.
    positions = []   # (confidence, match_id, "home"/"away", ranked_team_list)
    for mid in r32_ids:
        for pos, counts, fallback in (
            ("home", home_counts.get(mid), mc_home.get(mid)),
            ("away", away_counts.get(mid), mc_away.get(mid)),
        ):
            if counts is not None and len(counts):
                conf = float(counts.iloc[0] / counts.sum())
                ranked = counts.index.tolist()
            else:
                conf, ranked = 0.0, ([fallback] if fallback else [])
            positions.append((conf, mid, pos, ranked))

    positions.sort(key=lambda p: p[0], reverse=True)
    r32_home: dict[int, str] = {}
    r32_away: dict[int, str] = {}
    used: set[str] = set()
    for conf, mid, pos, ranked in positions:
        pick = next((t for t in ranked if t not in used), None)
        if pick is None:                       # exhausted -> accept modal even if dup
            pick = mc_home.get(mid) if pos == "home" else mc_away.get(mid)
        used.add(pick)
        (r32_home if pos == "home" else r32_away)[mid] = pick

    n_subbed = sum(1 for mid in r32_ids
                   if r32_home[mid] != mc_home.get(mid)
                   or r32_away[mid] != mc_away.get(mid))
    if n_subbed:
        log.append(f"R32 de-dup: {n_subbed} of {len(r32_ids)} slots had an occupant "
                   f"substituted to keep all 32 teams distinct")

    # --- 2. Play the bracket upward ----------------------------------------
    chosen: dict[int, tuple[str, str]] = {}
    winner_of: dict[int, str] = {}
    loser_of: dict[int, str] = {}

    def resolve(dep):
        kind, pid = dep
        return winner_of.get(pid) if kind == "Winner" else loser_of.get(pid)

    for mid in sorted(deps):                   # ascending -> feeders resolved first
        h_dep, a_dep = deps[mid]
        if h_dep is None and a_dep is None:
            home, away = r32_home[mid], r32_away[mid]
        else:
            home = resolve(h_dep) if h_dep else mc_home.get(mid)
            away = resolve(a_dep) if a_dep else mc_away.get(mid)
        chosen[mid] = (home, away)
        w = dc_winner_neutral(home, away, gm, h2h_lookup)
        winner_of[mid] = w
        loser_of[mid] = away if w == home else home

    fin = knockout_fixtures[knockout_fixtures["round"] == "Final"]
    if not fin.empty:
        fmid = int(fin.iloc[0]["match_id"])
        log.append(f"Final ({fmid}): {chosen[fmid][0]} vs {chosen[fmid][1]} "
                   f"-> winner {winner_of[fmid]}")
    return chosen, log


def predict_score_neutral(home, away, gm, h2h_lookup=None):
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
    delta_h2h = float(gm.get("delta", 0.0))
    h2h_ha = float((h2h_lookup or {}).get((home, away), 0.0))
    lh = math.exp(alpha + attack[hi] - defense[ai] + delta_h2h * h2h_ha)
    la = math.exp(alpha + attack[ai] - defense[hi] - delta_h2h * h2h_ha)

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


def _implied_total_mean(line, p_over):
    """Back out a Poisson mean for a total (corners or cards) from a de-vigged
    over probability at a given line. 'Over L.5' == total >= ceil(L)."""
    k = int(math.floor(float(line))) + 1          # over = total >= k
    p_over = min(max(float(p_over), 1e-4), 1 - 1e-4)
    f = lambda mu: (1.0 - poisson.cdf(k - 1, mu)) - p_over
    try:
        return brentq(f, 1e-3, 40.0)
    except ValueError:
        return None


def load_manual_market(fixtures_g):
    """Parse data/raw/manual_cards_corners_odds.csv (hand-collected corner/card
    totals) into {match_id: {'corners': mu, 'yellow_cards': mu}}. Returns
    (lookup, n_corner_fixtures, n_yellow_fixtures, unmatched_team_strings).
    Missing file -> empty lookup, so the blend is simply skipped."""
    path = Path("data/raw/manual_cards_corners_odds.csv")
    if not path.exists():
        return {}, 0, 0, []
    raw = pd.read_csv(path)
    key_to_mid = {(str(r.home_team), str(r.away_team)): int(r.match_id)
                  for r in fixtures_g.itertuples()}
    acc = defaultdict(lambda: defaultdict(list))
    unmatched = []
    for r in raw.itertuples():
        mid = key_to_mid.get((str(r.home_team), str(r.away_team)))
        if mid is None:
            unmatched.append(f"{r.home_team} vs {r.away_team}")
            continue
        try:
            io, iu = 1.0 / float(r.over_price), 1.0 / float(r.under_price)
            p_over = io / (io + iu)
            mu = _implied_total_mean(r.line, p_over)
        except (ValueError, ZeroDivisionError, TypeError):
            mu = None
        if mu is None:
            continue
        mk = str(r.market).strip().lower()
        if mk in ("corners", "corner"):
            acc[mid]["corners"].append(mu)
        elif mk in ("yellow_cards", "yellow_card", "yellows", "yellow", "cards"):
            acc[mid]["yellow_cards"].append(mu)
    lookup = {mid: {m: float(np.mean(v)) for m, v in d.items()}
              for mid, d in acc.items()}
    n_c = sum(1 for d in lookup.values() if "corners" in d)
    n_y = sum(1 for d in lookup.values() if "yellow_cards" in d)
    return lookup, n_c, n_y, sorted(set(unmatched))


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

    # Build a (home, away) -> h2h_avg_gd lookup for knockout neutral-venue
    # predictions. Uses the same definition as scripts 03/04 (avg per prior
    # meeting, clipped to +/-3).
    h2h_lookup = {}
    try:
        hist = pd.read_csv(PROCESSED / "matches_clean.csv", parse_dates=["date"])
        hist = hist.dropna(subset=["home_score", "away_score"])
        hist["pair"] = hist.apply(
            lambda r: tuple(sorted([r["home_team"], r["away_team"]])), axis=1
        )
        hist["gd_first"] = np.where(
            hist["home_team"] == hist["pair"].str[0],
            hist["home_score"] - hist["away_score"],
            hist["away_score"] - hist["home_score"]
        )
        agg = hist.groupby("pair", sort=False).agg(
            n=("gd_first", "size"), s=("gd_first", "sum")
        )
        for (ta, tb), row in agg.iterrows():
            avg = max(-3.0, min(3.0, row["s"] / max(row["n"], 1)))
            h2h_lookup[(ta, tb)] = avg
            h2h_lookup[(tb, ta)] = -avg
        ok("h2h_lookup", f"{len(h2h_lookup)//2} team pairs with history")
    except FileNotFoundError:
        warn("h2h_lookup", "matches_clean.csv not found, knockouts will use h2h=0")

    # Build bookmaker consensus lookup for the winning_team blend AND the
    # goals-lambda blend. odds_consensus.csv now carries both h2h probs and
    # market-derived lambdas (from spreads + totals markets).
    market_lookup = {}
    market_lambda_lookup = {}    # (home, away) -> (lambda_market_home, lambda_market_away)
    odds_path = PROCESSED / "odds_consensus.csv"
    if odds_path.exists():
        odds = pd.read_csv(odds_path)
        for _, o in odds.iterrows():
            market_lookup[(o["home_team"], o["away_team"])] = (
                float(o["p_home"]), float(o["p_draw"]), float(o["p_away"])
            )
            lh_m, la_m = o.get("lambda_market_home"), o.get("lambda_market_away")
            if pd.notna(lh_m) and pd.notna(la_m):
                market_lambda_lookup[(o["home_team"], o["away_team"])] = (float(lh_m), float(la_m))
        ok("market_lambdas",
           f"{len(market_lambda_lookup)} fixtures with market lambdas from spreads+totals "
           f"(blend={1-LAMBDA_BLEND_WEIGHT:.2f}*model + {LAMBDA_BLEND_WEIGHT:.2f}*market)")
        ok("market_lookup", f"{len(market_lookup)} fixtures with bookmaker consensus "
                            f"(blend={1-MARKET_BLEND_WEIGHT:.2f}*model + "
                            f"{MARKET_BLEND_WEIGHT:.2f}*market)")
    else:
        warn("market_lookup", "odds_consensus.csv not found; run 01e to enable market blend")

    manual_odds_lookup, n_manual_c, n_manual_y, manual_unmatched = load_manual_market(fixtures_g)
    if manual_odds_lookup:
        ok("manual_odds", f"corners on {n_manual_c} fixtures, yellow-cards on {n_manual_y} "
                          f"fixtures (blend={1-MANUAL_ODDS_BLEND_WEIGHT:.2f}*model + "
                          f"{MANUAL_ODDS_BLEND_WEIGHT:.2f}*market)")
        if manual_unmatched:
            warn("manual_odds_unmatched",
                 f"{len(manual_unmatched)} row-team(s) didn't match a group fixture: "
                 + ", ".join(manual_unmatched[:5]) + (" ..." if len(manual_unmatched) > 5 else ""))
    else:
        ok("manual_odds", "no manual_cards_corners_odds.csv (corner/card market blend skipped)")

    section("Building group-stage predictions (1-72)")
    rho = gm["rho"]
    group_rows = []
    n_winner_flipped = 0
    n_lambda_blended = 0
    n_corners_blended = 0
    n_yellow_blended = 0
    for _, r in fixtures_g.iterrows():
        mid = int(r["match_id"])
        g = group_lambdas[group_lambdas["match_id"] == mid].iloc[0]
        c = corners_preds[corners_preds["match_id"] == mid].iloc[0]
        y = cards_preds[cards_preds["match_id"] == mid].iloc[0]

        # Lambda blend: model (Dixon-Coles, Polymarket-calibrated) + market
        # (bookmaker totals + spreads). Market is sharper on blowouts.
        lh_model, la_model = float(g["lambda_home"]), float(g["lambda_away"])
        mkt_lambdas = market_lambda_lookup.get((r["home_team"], r["away_team"]))
        if mkt_lambdas is not None:
            w = LAMBDA_BLEND_WEIGHT
            lh = (1 - w) * lh_model + w * mkt_lambdas[0]
            la = (1 - w) * la_model + w * mkt_lambdas[1]
            n_lambda_blended += 1
        else:
            lh, la = lh_model, la_model
        hs, as_, p_home, p_draw, p_away = best_score_from_lambdas(lh, la, rho)

        # winning_team: blend with bookmaker consensus if available, then argmax
        mkt = market_lookup.get((r["home_team"], r["away_team"]))
        if mkt is not None:
            w = MARKET_BLEND_WEIGHT
            p_h_b = (1 - w) * p_home + w * mkt[0]
            p_d_b = (1 - w) * p_draw + w * mkt[1]
            p_a_b = (1 - w) * p_away + w * mkt[2]
        else:
            p_h_b, p_d_b, p_a_b = p_home, p_draw, p_away

        if p_h_b >= p_d_b and p_h_b >= p_a_b:
            winning = "home"
        elif p_d_b >= p_a_b:
            winning = "draw"
        else:
            winning = "away"

        # Track how often the market changes the model's call
        model_pick = ("home" if p_home >= p_draw and p_home >= p_away
                      else "draw" if p_draw >= p_away else "away")
        if winning != model_pick:
            n_winner_flipped += 1

        # Corners / yellow-cards: blend the model estimate with hand-collected
        # bookmaker totals where available (group fixtures only).
        manual = manual_odds_lookup.get(mid, {})
        corners_val = float(c["pred_total_corners"]) + CORNERS_TOURNAMENT_BUMP
        if "corners" in manual:
            corners_val = ((1 - MANUAL_ODDS_BLEND_WEIGHT) * corners_val
                           + MANUAL_ODDS_BLEND_WEIGHT * manual["corners"])
            n_corners_blended += 1
        yellow_val = (float(y["pred_yellow"]) if "pred_yellow" in y
                      else float(y["pred_yellow_rounded"]))
        if "yellow_cards" in manual:
            yellow_val = ((1 - MANUAL_ODDS_BLEND_WEIGHT) * yellow_val
                          + MANUAL_ODDS_BLEND_WEIGHT * manual["yellow_cards"])
            n_yellow_blended += 1

        group_rows.append({
            "match_id":     mid,
            "home_team":    r["home_team"],
            "away_team":    r["away_team"],
            "home_score":   hs,
            "away_score":   as_,
            "corners":      int(round(corners_val)),
            "yellow_cards": int(round(yellow_val)),
            "red_cards":    int(y["pred_red_rounded"]),
            "winning_team": winning,
            "match_winner": np.nan,
            "penalties":    np.nan,
        })
    ok("group_predictions", f"{len(group_rows)} matches")
    if market_lookup:
        ok("market_flips", f"{n_winner_flipped} of {len(group_rows)} group winners "
                            f"changed by market blend")
    if market_lambda_lookup:
        ok("lambda_blends",
           f"{n_lambda_blended} of {len(group_rows)} group fixtures had goals "
           f"lambdas blended with market totals+spreads")
    if manual_odds_lookup:
        ok("manual_blends", f"{n_corners_blended} corners + {n_yellow_blended} yellow-card "
                            f"group fixtures blended with manual bookmaker totals")

    section("Bottom-up bracket reconciliation")
    # Seed the Round of 32 from the MC's per-slot modal matchups (de-duplicated)
    # and play every later round upward via Dixon-Coles. Keeps the teams the model
    # expects in the bracket instead of the old top-down pass that evicted them.
    mc_home_map = dict(zip(mc_knockout["match_id"], mc_knockout["pred_home_team"]))
    mc_away_map = dict(zip(mc_knockout["match_id"], mc_knockout["pred_away_team"]))
    chosen: dict[int, tuple[str, str]] = {}
    if matchup_details is not None:
        chosen, log = reconcile_bracket_bottom_up(
            matchup_details, mc_knockout, fixtures_k, gm, h2h_lookup
        )
        n_diff = sum(1 for mid, (h, a) in chosen.items()
                     if h != mc_home_map.get(mid) or a != mc_away_map.get(mid))
        for entry in log[:6]:
            ok("reconcile", entry)
        ok("bracket_built", f"{len(chosen)} matches; {n_diff} differ from MC raw "
                            f"per-slot modes")
        if len(log) > 6:
            ok("more_reconcile_notes", f"... {len(log) - 6} additional notes (suppressed)")
    else:
        warn("matchup_details", "mc_matchup_details.csv not found -- re-run script 07")

    section("Building knockout-stage predictions (73-104)")
    knockout_lookup = fixtures_k.set_index("match_id")
    ko_rows = []
    for _, r in mc_knockout.iterrows():
        mid = int(r["match_id"])
        if mid in chosen:
            home, away = chosen[mid]
        else:
            home = r["pred_home_team"]
            away = r["pred_away_team"]
        fx = knockout_lookup.loc[mid]
        altitude_m = fx.get("altitude_m", np.nan)

        hs, as_, p_home, p_draw, p_away = predict_score_neutral(home, away, gm, h2h_lookup)
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
