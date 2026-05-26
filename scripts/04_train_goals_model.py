"""
04_train_goals_model.py
Fits a Dixon-Coles Poisson goals model via weighted MLE.

The model estimates per-team attack and defense parameters plus a global
home-advantage and the Dixon-Coles low-score correction (rho). Training uses
the sample weights produced by 03 (time decay x friendly downweight).

Run from worldcup2026/:  py scripts/04_train_goals_model.py

Outputs:
  models/goals_model.pkl              -- fitted parameters + team index
  data/processed/group_lambdas.csv    -- per-fixture (lambda_home, lambda_away)
"""

import math
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import poisson

PROCESSED = Path("data/processed")
MODELS = Path("models")
MODELS.mkdir(exist_ok=True)


def section(t): print(f"\n-- {t} " + "-" * (60 - len(t)))
def ok(name, detail=""): print(f"  [OK]   {name:<26} {detail}")
def warn(name, detail=""): print(f"  [WARN] {name:<26} {detail}")


TRAIN_FROM = "2018-01-01"
MIN_MATCHES_PER_TEAM = 10


# -- Dixon-Coles correction (vectorised) --------------------------------------

def dc_tau(x, y, lh, la, rho):
    tau = np.ones_like(lh)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    tau = np.where(m00, 1 - lh * la * rho, tau)
    tau = np.where(m01, 1 + lh * rho, tau)
    tau = np.where(m10, 1 + la * rho, tau)
    tau = np.where(m11, 1 - rho, tau)
    return tau


def make_nll(home_idx, away_idx, x, y, w, n_teams):

    def nll(params):
        attack = params[:n_teams]
        defense = params[n_teams:2 * n_teams]
        alpha = params[2 * n_teams]
        gamma = params[2 * n_teams + 1]
        rho = params[2 * n_teams + 2]

        lh = np.exp(alpha + attack[home_idx] - defense[away_idx] + gamma)
        la = np.exp(alpha + attack[away_idx] - defense[home_idx])

        log_ph = poisson.logpmf(x, lh)
        log_pa = poisson.logpmf(y, la)
        tau = dc_tau(x, y, lh, la, rho)
        log_tau = np.log(np.clip(tau, 1e-10, None))

        penalty = 100.0 * (attack.sum() ** 2 + defense.sum() ** 2)
        return -np.sum(w * (log_ph + log_pa + log_tau)) + penalty

    return nll


def score_distribution(lh, la, rho, max_goals=8):
    grid = np.arange(max_goals + 1)
    p_h = poisson.pmf(grid, lh)
    p_a = poisson.pmf(grid, la)
    P = np.outer(p_h, p_a)
    P[0, 0] *= max(1 - lh * la * rho, 0)
    P[0, 1] *= max(1 + lh * rho, 0)
    P[1, 0] *= max(1 + la * rho, 0)
    P[1, 1] *= max(1 - rho, 0)
    s = P.sum()
    if s > 0:
        P /= s
    return P


def best_scoreline(P):
    return np.unravel_index(np.argmax(P), P.shape)


def expected_goals_from_P(P):
    grid = np.arange(P.shape[0])
    eh = float((P.sum(axis=1) * grid).sum())
    ea = float((P.sum(axis=0) * grid).sum())
    return eh, ea


def win_draw_loss_probs(P):
    home_win = float(np.tril(P, k=-1).sum())
    draw     = float(np.diag(P).sum())
    away_win = float(np.triu(P, k=1).sum())
    return home_win, draw, away_win


def main():
    print("=" * 60)
    print("04_train_goals_model.py  |  Dixon-Coles Poisson")
    print("=" * 60)

    section("Loading training data")
    matches = pd.read_csv(PROCESSED / "match_features.csv", parse_dates=["date"])
    fixtures = pd.read_csv(PROCESSED / "fixture_features.csv")
    ok("matches_total", f"{len(matches):,}")

    train = matches[matches["date"] >= TRAIN_FROM].dropna(
        subset=["home_score", "away_score", "sample_weight"]
    ).copy()
    ok("matches_recent", f"{len(train):,} since {TRAIN_FROM}")

    team_counts = pd.concat([train["home_team"], train["away_team"]]).value_counts()
    keep_teams = set(team_counts[team_counts >= MIN_MATCHES_PER_TEAM].index)
    train = train[train["home_team"].isin(keep_teams) & train["away_team"].isin(keep_teams)]
    ok("teams_kept", f"{len(keep_teams)} teams (>={MIN_MATCHES_PER_TEAM} matches)")
    ok("matches_kept", f"{len(train):,} after team filter")

    teams = sorted(keep_teams)
    team_to_idx = {t: i for i, t in enumerate(teams)}
    home_idx = train["home_team"].map(team_to_idx).to_numpy()
    away_idx = train["away_team"].map(team_to_idx).to_numpy()
    x = train["home_score"].to_numpy().astype(int)
    y = train["away_score"].to_numpy().astype(int)
    w = train["sample_weight"].to_numpy()

    n_teams = len(teams)
    section("Fitting parameters")
    alpha0 = math.log(max((x.mean() + y.mean()) / 2.0, 0.1))
    p0 = np.concatenate([
        np.zeros(n_teams),
        np.zeros(n_teams),
        [alpha0, 0.3, -0.13],   # rho ~ -0.13 per README (low-score boost)
    ])
    bounds = (
        [(-3, 3)] * n_teams +
        [(-3, 3)] * n_teams +
        [(-2, 2), (0, 1), (-0.5, 0.5)]
    )

    nll = make_nll(home_idx, away_idx, x, y, w, n_teams)
    ok("optimizer", f"L-BFGS-B, {2 * n_teams + 3} free params")

    res = minimize(nll, p0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 2000})
    if not res.success:
        warn("convergence", res.message)
    ok("converged", f"final NLL = {res.fun:,.1f}")

    params = res.x
    attack = params[:n_teams]
    defense = params[n_teams:2 * n_teams]
    alpha = params[2 * n_teams]
    gamma = params[2 * n_teams + 1]
    rho = params[2 * n_teams + 2]
    ok("home_advantage", f"gamma = {gamma:.3f}  (~{(math.exp(gamma) - 1) * 100:.1f}% more home goals)")
    ok("dc_rho", f"rho = {rho:.3f}")
    ok("intercept", f"alpha = {alpha:.3f}  (~{math.exp(alpha):.2f} avg goals/team)")

    section("Top attacking sides (largest attack coef)")
    rank_a = sorted(zip(teams, attack), key=lambda t: -t[1])[:8]
    for t, v in rank_a:
        print(f"  {t:<28} {v:+.3f}")
    section("Strongest defences (largest positive defense coef = fewest goals conceded)")
    rank_d = sorted(zip(teams, defense), key=lambda t: -t[1])[:8]
    for t, v in rank_d:
        print(f"  {t:<28} {v:+.3f}")
    section("Weakest defences (largest negative = most goals conceded)")
    worst_d = sorted(zip(teams, defense), key=lambda t: t[1])[:5]
    for t, v in worst_d:
        print(f"  {t:<28} {v:+.3f}")

    section("In-sample goodness of fit")
    lh_in = np.exp(alpha + attack[home_idx] - defense[away_idx] + gamma)
    la_in = np.exp(alpha + attack[away_idx] - defense[home_idx])
    mae_home = float(np.mean(np.abs(lh_in - x)))
    mae_away = float(np.mean(np.abs(la_in - y)))
    ok("MAE home", f"{mae_home:.3f}  (README target <0.8)")
    ok("MAE away", f"{mae_away:.3f}  (README target <0.8)")

    section("Saving model")
    model = {
        "teams": teams, "team_to_idx": team_to_idx,
        "attack": attack, "defense": defense,
        "alpha": alpha, "gamma": gamma, "rho": rho,
        "training_window_from": TRAIN_FROM,
        "training_matches": int(len(train)),
        "in_sample_mae_home": mae_home,
        "in_sample_mae_away": mae_away,
    }
    with open(MODELS / "goals_model.pkl", "wb") as f:
        pickle.dump(model, f)
    ok("goals_model.pkl", "saved")

    section("Predicting lambdas for group fixtures")
    # The training data treats every match as if 'home' got the home advantage,
    # but in the WC, only host-nation matches at home actually do. We apply
    # gamma fully when home_is_host=1, and zero it out for neutral-venue matches.
    rows = []
    for _, fx in fixtures.iterrows():
        h, a = fx["home_team"], fx["away_team"]
        if h not in team_to_idx or a not in team_to_idx:
            warn("missing_team", f"match {fx['match_id']}: {h} or {a} not in trained set")
            rows.append({
                "match_id": fx["match_id"], "home_team": h, "away_team": a,
                "lambda_home": np.nan, "lambda_away": np.nan,
                "mode_home_score": np.nan, "mode_away_score": np.nan,
                "exp_home_score": np.nan, "exp_away_score": np.nan,
                "p_home_win": np.nan, "p_draw": np.nan, "p_away_win": np.nan,
            })
            continue
        hi, ai = team_to_idx[h], team_to_idx[a]
        home_is_host = int(fx.get("home_is_host", 0) or 0)
        away_is_host = int(fx.get("away_is_host", 0) or 0)
        # Apply gamma only to the team genuinely playing at home
        gamma_h = gamma if home_is_host else 0.0
        gamma_a = gamma if away_is_host else 0.0
        lh = math.exp(alpha + attack[hi] - defense[ai] + gamma_h)
        la = math.exp(alpha + attack[ai] - defense[hi] + gamma_a)
        P = score_distribution(lh, la, rho)
        mh, ma = best_scoreline(P)
        eh, ea = expected_goals_from_P(P)
        ph, pd_, pa = win_draw_loss_probs(P)
        rows.append({
            "match_id": fx["match_id"], "home_team": h, "away_team": a,
            "home_is_host": home_is_host, "away_is_host": away_is_host,
            "lambda_home": lh, "lambda_away": la,
            "mode_home_score": int(mh), "mode_away_score": int(ma),
            "exp_home_score": eh, "exp_away_score": ea,
            "p_home_win": ph, "p_draw": pd_, "p_away_win": pa,
        })
    preds = pd.DataFrame(rows)
    preds.to_csv(PROCESSED / "group_lambdas.csv", index=False)
    ok("group_lambdas.csv", f"{len(preds)} fixtures")

    section("Sample predictions (first 5 group matches)")
    show = preds.head(5)[[
        "match_id", "home_team", "away_team",
        "lambda_home", "lambda_away",
        "mode_home_score", "mode_away_score",
        "p_home_win", "p_draw", "p_away_win"
    ]].round(3)
    print(show.to_string(index=False))

    print("\n" + "=" * 60)
    print("Goals model training complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
