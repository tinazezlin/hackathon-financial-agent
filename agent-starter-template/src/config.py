"""Team-level settings: dates, constraints, benchmark, models.

Everything investment-policy related lives here so it can be reviewed in one
place (and quoted in the report / presentation).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Reporting (rebalancing) dates as in deliverables.pdf (2026-11-11). The original
# briefing listed 30 Dec / 31 Jan / 28 Feb; the deliverables list is the newer one
# and uses business days only. Check the real template's list on the day.
REPORTING_DATES = [
    "2025-10-31",
    "2025-11-28",
    "2025-12-29",
    "2026-01-30",
    "2026-02-27",
    "2026-03-31",
]

BASE_CURRENCY = "EUR"
FX_HEDGED = False          # USD assets are converted to EUR unhedged for risk numbers
LOOKBACK_DAYS = 756        # ~3 years of daily data for risk statistics
MAX_REVISIONS = 1          # times the risk review may send the PM back to revise
MAX_TOOL_ROUNDS = 6        # safety cap on analyst tool-calling loops

MODEL_ID = os.getenv("MODEL_ID", "openai/gpt-oss-120b")


@dataclass(frozen=True)
class Limit:
    field: str         # column of universe.assets_meta() ("asset_class", "jurisdiction", ...) or "ticker"
    value: str         # value of that column the limit applies to
    bound: float       # weight bound
    sense: str = "<="  # "<=" (cap) or ">=" (floor)
    label: str = ""
    mandatory: bool = False


# Mandatory constraints from the briefing (no short selling is enforced as w >= 0)
LIMITS: list[Limit] = [
    Limit("asset_class", "Equity", 0.40, "<=", "Equities <= 40%", mandatory=True),
    Limit("jurisdiction", "US", 0.20, "<=", "US jurisdiction <= 20%", mandatory=True),
    Limit("asset_class", "Corporate Bond", 0.20, "<=", "Corporate bonds <= 20%", mandatory=True),
    # --- Extra prudential constraints (team choice; explain them in the report) ---
    Limit("asset_class", "Cash", 0.05, ">=", "Liquidity tranche: cash >= 5%"),
]
MAX_SINGLE_ASSET = 0.30    # extra: no single index above 30%

# Neutral strategic allocation the agents tilt away from (team choice), defined by
# (asset class, region) buckets so it works for whatever 23 assets the real data has:
# each bucket's weight is split equally across its assets (universe.strategic_benchmark).
# Region: "US" for US jurisdiction, "EUR" for everything else. Totals: Gov 60%,
# Corp 11%, Equity 14%, Cash 15%; US 18%. Satisfies all limits above.
BENCHMARK_BUCKETS = {
    ("Government Bond", "EUR"): 0.52, ("Government Bond", "US"): 0.08,
    ("Corporate Bond", "EUR"): 0.09, ("Corporate Bond", "US"): 0.02,
    ("Equity", "EUR"): 0.08, ("Equity", "US"): 0.06,
    ("Cash", "EUR"): 0.13, ("Cash", "US"): 0.02,
}
