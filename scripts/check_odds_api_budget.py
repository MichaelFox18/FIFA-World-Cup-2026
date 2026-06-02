"""
check_odds_api_budget.py
Hits The Odds API's `/sports` endpoint (1-credit cost) just to read the
x-requests-remaining + x-requests-used headers from the response, then
exits. Use this when you want to know your monthly-budget status without
re-running the full data pull.

Run from worldcup2026/:  py scripts/check_odds_api_budget.py
"""

import os
import sys
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
key = os.getenv("ODDS_API_KEY")
if not key:
    print("ERROR: ODDS_API_KEY not set in .env"); sys.exit(1)

r = requests.get("https://api.the-odds-api.com/v4/sports",
                 params={"apiKey": key}, timeout=15)
remaining = r.headers.get("x-requests-remaining", "?")
used = r.headers.get("x-requests-used", "?")
last_cost = r.headers.get("x-requests-last", "?")
print(f"Used this month:   {used}")
print(f"Remaining:         {remaining}")
print(f"Cost of this call: {last_cost}")
print()
print("Note: each /odds pull with markets=h2h,spreads,totals costs 3 credits.")
print("      Scheduled refreshes through 2026-06-09: budget ~12 credits.")
