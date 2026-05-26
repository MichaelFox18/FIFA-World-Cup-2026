"""
09_format_output.py
Splits the combined predictions into the two submission CSVs DataLab expects,
runs the README's final sanity checks, and prints a tournament summary.

Group-stage submission (match_id 1-72):
  match_id, home_score, away_score, corners, yellow_cards, red_cards, winning_team

Knockout-stage submission (match_id 73-104):
  match_id, home_team, away_team, home_score, away_score, corners,
  yellow_cards, red_cards, match_winner, penalties

Run AFTER script 08:  py scripts/09_format_output.py

Outputs:
  output/predictions_group.csv      (72 rows x 7 cols)
  output/predictions_knockout.csv   (32 rows x 10 cols)
  output/predictions_final.csv      (combined view, kept in sync)
"""

import pandas as pd
from pathlib import Path

PROCESSED = Path("data/processed")
OUTPUT    = Path("output")
OUTPUT.mkdir(parents=True, exist_ok=True)


def section(t): print(f"\n-- {t} " + "-" * (60 - len(t)))
def ok(name, detail=""): print(f"  [OK]   {name:<26} {detail}")
def warn(name, detail=""): print(f"  [WARN] {name:<26} {detail}")
def fail(name, detail=""): print(f"  [FAIL] {name:<26} {detail}")


def main():
    print("=" * 60)
    print("09_format_output.py  |  DataLab submission format")
    print("=" * 60)

    section("Loading combined predictions")
    src = PROCESSED / "predictions_combined.csv"
    if not src.exists():
        fail("missing", "predictions_combined.csv -- run script 08 first")
        return
    df = pd.read_csv(src)
    ok("combined", f"{len(df)} rows")

    section("Building group-stage submission (matches 1-72)")
    group = df[df["match_id"] <= 72].copy()
    group_cols = ["match_id", "home_score", "away_score", "corners",
                  "yellow_cards", "red_cards", "winning_team"]
    group_out = group[group_cols].copy()
    for c in ["match_id", "home_score", "away_score", "corners",
              "yellow_cards", "red_cards"]:
        group_out[c] = group_out[c].astype(int)
    group_out["winning_team"] = group_out["winning_team"].astype(str).str.lower()
    ok("rows", f"{len(group_out)}")
    ok("cols", ", ".join(group_out.columns))

    section("Building knockout-stage submission (matches 73-104)")
    knock = df[df["match_id"] >= 73].copy()
    knock_cols = ["match_id", "home_team", "away_team", "home_score",
                  "away_score", "corners", "yellow_cards", "red_cards",
                  "match_winner", "penalties"]
    knock_out = knock[knock_cols].copy()
    for c in ["match_id", "home_score", "away_score", "corners",
              "yellow_cards", "red_cards"]:
        knock_out[c] = knock_out[c].astype(int)
    knock_out["match_winner"] = knock_out["match_winner"].astype(str).str.lower()
    knock_out["penalties"] = knock_out["penalties"].astype(bool)
    ok("rows", f"{len(knock_out)}")
    ok("cols", ", ".join(knock_out.columns))

    section("Final README sanity checks")
    all_preds = df.copy()
    all_preds["total_goals"] = all_preds["home_score"] + all_preds["away_score"]

    valid_winning = {"home", "away", "draw"}
    valid_winner  = {"home", "away"}

    checks = [
        ("avg_goals",
         2.2 <= all_preds["total_goals"].mean() <= 2.8,
         f"{all_preds['total_goals'].mean():.2f}  (target 2.2-2.8)"),
        ("avg_corners",
         9.0 <= all_preds["corners"].mean() <= 11.5,
         f"{all_preds['corners'].mean():.2f}  (target 9.0-11.5)"),
        ("avg_yellow",
         2.8 <= all_preds["yellow_cards"].mean() <= 3.8,
         f"{all_preds['yellow_cards'].mean():.2f}  (target 2.8-3.8)"),
        ("avg_red",
         all_preds["red_cards"].mean() < 0.25,
         f"{all_preds['red_cards'].mean():.2f}  (target <0.25)"),
        ("sum_red_total",
         int(all_preds["red_cards"].sum()) <= 20,
         f"{int(all_preds['red_cards'].sum())}  (target <=20)"),
        ("match_count_total",
         len(all_preds) == 104,
         f"{len(all_preds)}  (target 104)"),
        ("match_count_group",
         len(group_out) == 72,
         f"{len(group_out)}  (target 72)"),
        ("match_count_knockout",
         len(knock_out) == 32,
         f"{len(knock_out)}  (target 32)"),
        ("winning_team_valid",
         group_out["winning_team"].isin(valid_winning).all(),
         f"{int((~group_out['winning_team'].isin(valid_winning)).sum())} invalid"),
        ("match_winner_valid",
         knock_out["match_winner"].isin(valid_winner).all(),
         f"{int((~knock_out['match_winner'].isin(valid_winner)).sum())} invalid"),
        ("penalties_bool",
         knock_out["penalties"].dtype == bool,
         f"dtype = {knock_out['penalties'].dtype}"),
        ("group_nulls",
         int(group_out.isnull().sum().sum()) == 0,
         f"{int(group_out.isnull().sum().sum())}"),
        ("knockout_nulls",
         int(knock_out.isnull().sum().sum()) == 0,
         f"{int(knock_out.isnull().sum().sum())}"),
    ]

    all_passed = True
    for name, passed, val in checks:
        (ok if passed else warn)(name, val)
        if not passed:
            all_passed = False

    section("Writing submission files")
    p_group = OUTPUT / "predictions_group.csv"
    p_knock = OUTPUT / "predictions_knockout.csv"
    p_final = OUTPUT / "predictions_final.csv"

    group_out.to_csv(p_group, index=False)
    knock_out.to_csv(p_knock, index=False)

    df_to_save = df.drop(columns=[c for c in ["total_goals"] if c in df.columns])
    df_to_save.to_csv(p_final, index=False)

    ok("predictions_group.csv",    f"{p_group}  ({len(group_out)} rows)")
    ok("predictions_knockout.csv", f"{p_knock}  ({len(knock_out)} rows)")
    ok("predictions_final.csv",    f"{p_final}  ({len(df)} rows, combined)")

    section("Predicted tournament summary")
    final_row = knock_out[knock_out["match_id"] == 104].iloc[0]
    third_row = knock_out[knock_out["match_id"] == 103].iloc[0]
    champion  = final_row["home_team"] if final_row["match_winner"] == "home" else final_row["away_team"]
    runner_up = final_row["away_team"] if final_row["match_winner"] == "home" else final_row["home_team"]
    third     = third_row["home_team"] if third_row["match_winner"] == "home" else third_row["away_team"]
    fourth    = third_row["away_team"] if third_row["match_winner"] == "home" else third_row["home_team"]

    print(f"  Champion       : {champion}")
    print(f"  Runner-up      : {runner_up}")
    print(f"  3rd place      : {third}")
    print(f"  4th place      : {fourth}")
    print(f"  Penalties in   : {int(knock_out['penalties'].sum())} of 32 knockout matches")
    print(f"  Total goals    : {all_preds['total_goals'].sum()} (avg {all_preds['total_goals'].mean():.2f}/match)")
    print(f"  Total corners  : {int(all_preds['corners'].sum())} (avg {all_preds['corners'].mean():.2f}/match)")
    print(f"  Total yellow   : {int(all_preds['yellow_cards'].sum())} (avg {all_preds['yellow_cards'].mean():.2f}/match)")
    print(f"  Total red      : {int(all_preds['red_cards'].sum())}")

    print()
    if all_passed:
        print("=" * 60)
        print("  ALL CHECKS PASSED.  Submission files ready:")
        print(f"    {p_group}")
        print(f"    {p_knock}")
        print("=" * 60)
    else:
        print("=" * 60)
        print("  Some checks failed -- review WARN lines above before submitting.")
        print("=" * 60)


if __name__ == "__main__":
    main()
