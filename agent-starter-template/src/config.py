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
    field: str         # column in assets-metadata ("asset_class", "jurisdiction", "currency") or "ticker"
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

# Neutral strategic allocation the agents tilt away from (team choice).
# Satisfies all limits above. Sums to 1.
STRATEGIC_BENCHMARK = {
    "EUR_CASH": 0.05, "EUR_TBILL_0_1Y": 0.05, "US_CASH": 0.02,
    "DE_GOV_1_3Y": 0.13, "DE_GOV_7_10Y": 0.08, "DE_GOV_10Y_PLUS": 0.02,
    "FR_GOV_1_3Y": 0.07, "FR_GOV_7_10Y": 0.05,
    "IT_GOV_1_3Y": 0.04, "IT_GOV_7_10Y": 0.03,
    "ES_GOV_1_3Y": 0.04, "ES_GOV_7_10Y": 0.03, "EA_GOV_ALL": 0.05,
    "US_GOV_1_3Y": 0.05, "US_GOV_3_5Y": 0.03, "US_GOV_7_10Y": 0.01,
    "EUR_CORP_IG": 0.05, "EUR_CORP_IG_1_5Y": 0.04, "US_CORP_IG": 0.02,
    "EUROSTOXX50": 0.04, "DAX": 0.02, "CAC40": 0.02, "SP500": 0.06,
}
