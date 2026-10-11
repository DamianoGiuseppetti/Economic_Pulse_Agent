"""
Centralized config. Load from environment (or a .env file) with defaults.

Required:
- FRED_API_KEY: Free from https://fred.stlouisfed.org/docs/api/
- DB_PATH: Path to SQLite file (default: "./data/economic_pulse.db")

Optional:
- CACHE_TTL_FRED: seconds (default: 86400 = 24 hours)
- CACHE_TTL_STOCK: seconds (default: 3600 = 1 hour)

Nothing here raises at import time: a missing FRED key is reported by the
FRED fetcher at the moment it is actually needed, so yfinance-only code still
works without one.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent

# Look for .env next to this package first, then in the repo root.
load_dotenv(PROJECT_DIR / ".env")
load_dotenv(PROJECT_DIR.parent / ".env")

FRED_API_KEY: str | None = os.getenv("FRED_API_KEY") or None
FRED_BASE_URL = "https://api.stlouisfed.org/fred"

DB_PATH: str = os.getenv("DB_PATH", str(PROJECT_DIR / "data" / "economic_pulse.db"))

CACHE_TTL_FRED: int = int(os.getenv("CACHE_TTL_FRED", "86400"))
CACHE_TTL_STOCK: int = int(os.getenv("CACHE_TTL_STOCK", "3600"))
# Daily/weekly/monthly bars don't change intraday, so they get the longer TTL.
CACHE_TTL_STOCK_DAILY: int = 86400
# Macro context mixes daily market data and Fed releases: refresh a few times a day.
CACHE_TTL_MACRO: int = int(os.getenv("CACHE_TTL_MACRO", "21600"))

HTTP_TIMEOUT_SECONDS: int = 30

# --- Approval gate (approval.py) -------------------------------------------------------------------
# The tools are read-only, so the gate protects cost, latency and rate limits, not safety. Defaults come
# from a small sample of logged usage (22 turns: largest turn 2 tool calls, largest session about $0.11);
# they are starting points, not tuned values. Override with the environment variables.
# A cold-cache fetch estimated above this many observations (about five years of daily data).
APPROVAL_MAX_OBSERVATIONS: int = int(os.getenv("APPROVAL_MAX_OBSERVATIONS", "1300"))
# More tool calls than this already made in the current turn (max_iterations is 5).
APPROVAL_MAX_TOOL_CALLS_PER_TURN: int = int(os.getenv("APPROVAL_MAX_TOOL_CALLS_PER_TURN", "4"))
# Cumulative model cost of the session, in dollars.
APPROVAL_SESSION_COST_USD: float = float(os.getenv("APPROVAL_SESSION_COST_USD", "0.15"))
