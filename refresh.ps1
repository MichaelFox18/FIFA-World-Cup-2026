# refresh.ps1
# Re-runs the full FIFA 2026 prediction pipeline end-to-end. Use this on each
# scheduled refresh day to fold in the latest Polymarket / Odds API / FIFA
# ranking data and produce a fresh submission.
#
# Run from worldcup2026/:  .\refresh.ps1
#                          .\refresh.ps1 -SkipCollect   (skip 01/01c/01d/01e)
#                          .\refresh.ps1 -SkipMC        (skip 07 -- ~30s saved)
#
# Order matches the dependency chain in NextStep.md.

param(
    [switch]$SkipCollect,
    [switch]$SkipMC
)

$ErrorActionPreference = "Stop"
$started = Get-Date
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host " FIFA 2026 prediction pipeline -- full refresh"             -ForegroundColor Cyan
Write-Host " Started: $started"                                          -ForegroundColor Cyan
Write-Host "===========================================================" -ForegroundColor Cyan

function Step([string]$label, [string]$script) {
    Write-Host ""
    Write-Host ">>> $label  ($script)" -ForegroundColor Yellow
    $t0 = Get-Date
    py $script
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ABORT: $script exited $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    $dt = (Get-Date) - $t0
    Write-Host "    done in $([math]::Round($dt.TotalSeconds, 1))s" -ForegroundColor Green
}

if (-not $SkipCollect) {
    Step "Collect external sources"        "scripts/01_collect_data.py"
    Step "Load WC22 match stats"           "scripts/01c_load_match_stats.py"
    Step "Load external tournament stats"  "scripts/01d_load_external_match_stats.py"
    Step "Process Odds API h2h"            "scripts/01e_process_odds.py"
    Step "Compute player form (watchlist)" "scripts/01f_compute_player_form.py"
}

Step "Clean + merge"                       "scripts/02_clean_merge.py"
Step "Feature engineering"                 "scripts/03_feature_engineering.py"
Step "Train Dixon-Coles goals model"       "scripts/04_train_goals_model.py"
Step "Polymarket Bayesian calibration"     "scripts/04b_calibrate_goals_model.py"
Step "Train corners model"                 "scripts/05_train_corners_model.py"
Step "Train cards model"                   "scripts/06_train_cards_model.py"

if (-not $SkipMC) {
    Step "Monte Carlo tournament sim"      "scripts/07_monte_carlo_simulator.py"
}

Step "Generate combined predictions"       "scripts/08_generate_predictions.py"
Step "Format DataLab submission"           "scripts/09_format_output.py"

$elapsed = (Get-Date) - $started
Write-Host ""
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host " Pipeline complete in $([math]::Round($elapsed.TotalSeconds, 1))s" -ForegroundColor Cyan
Write-Host " Outputs:"                                                   -ForegroundColor Cyan
Write-Host "   output\predictions_group.csv      (72 group matches)"
Write-Host "   output\predictions_knockout.csv   (32 knockout matches)"
Write-Host "   output\predictions_final.csv      (104 combined, audit copy)"
Write-Host "===========================================================" -ForegroundColor Cyan
