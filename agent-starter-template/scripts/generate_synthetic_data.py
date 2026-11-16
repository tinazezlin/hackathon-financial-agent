"""Generate SYNTHETIC practice data that mimics the hackathon datasets.

Everything here is invented. Paths are loosely shaped like 2018-2026 history
(COVID crash, 2022 inflation/hiking cycle, 2024-25 cuts, spread episodes) so
that agents have something "realistic" to reason about, but numbers, dates and
headlines are NOT real and the Oct 2025 - Mar 2026 period is pure fiction.

Files written to data/, in the formats of the organisers' data-detailed.pdf:
    assets.csv              long: date (dd/mm/yyyy), index, price, price_description
    assets_metadata.csv     jurisdiction, category, indicator, index, field, description (utf-8-sig)
    financial.csv           long: date (dd/mm/yyyy), index, price, price_description
    financial_metadata.csv  jurisdiction, category, indicator, index, field, description (utf-8-sig)
    macro.csv               long: date (dd/mm/yyyy, FIRST day of the reference month; PMI on the
                            LAST day; quarterly on the first day of the quarter), identifier, value
    macro_metadata.csv      identifier, jurisdiction, category, indicator, database,
                            frequencyDescription, dataType, aggregationTypeDescription (utf-8-sig)
    news.csv                stand-in for the parquet files: published_at (UTC), date,
                            version_created, story_id, version, version_role, title,
                            topic_name ("|"-separated list), has_body, body
    news_topics.csv         topic, count

Internally the script uses readable names (DE_GOV_1_3Y, ...); the files use codes, as
the real data does. The 23-asset universe (16 government bonds, no Spain, jurisdictions
EUR/US/DE/IT/FR) is our guess at the real one.

Run:  .venv/bin/python scripts/generate_synthetic_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
START, END = "2018-01-01", "2026-03-31"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
rng = np.random.default_rng(SEED)

DATES = pd.bdate_range(START, END)
N = len(DATES)


# --------------------------------------------------------------------------- helpers
def anchors(points: dict[str, float], dates=DATES) -> np.ndarray:
    """Piecewise-linear path through (date -> value) anchor points."""
    s = pd.Series(points, dtype=float)
    s.index = pd.to_datetime(s.index)
    s = s.reindex(s.index.union(dates)).interpolate("time").ffill().bfill()
    return s.reindex(dates).to_numpy()


def steps(changes: dict[str, float], start_value: float, dates=DATES) -> np.ndarray:
    """Step function (policy rates): value changes on the given dates."""
    out = np.full(len(dates), start_value, dtype=float)
    for d, v in sorted(changes.items()):
        out[dates >= pd.Timestamp(d)] = v
    return out


def ou(n: int, sd: float, kappa: float, shocks: np.ndarray | None = None) -> np.ndarray:
    """Mean-reverting (Ornstein-Uhlenbeck) noise around 0."""
    eps = rng.standard_normal(n) if shocks is None else shocks
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = (1 - kappa) * x[t - 1] + sd * eps[t]
    return x


def shift_forward(x: np.ndarray, days: int) -> np.ndarray:
    """Value `days` business days ahead (markets price expected policy)."""
    return np.concatenate([x[days:], np.repeat(x[-1], days)])


# --------------------------------------------------------------------------- policy & rates
ECB_DEPO = steps({
    "2019-09-18": -0.50, "2022-07-27": 0.00, "2022-09-14": 0.75, "2022-11-02": 1.50,
    "2022-12-21": 2.00, "2023-02-08": 2.50, "2023-03-22": 3.00, "2023-05-10": 3.25,
    "2023-06-21": 3.50, "2023-08-02": 3.75, "2023-09-20": 4.00, "2024-06-12": 3.75,
    "2024-09-18": 3.50, "2024-10-23": 3.25, "2024-12-18": 3.00, "2025-02-05": 2.75,
    "2025-03-12": 2.50, "2025-04-23": 2.25, "2025-06-11": 2.00,
    # fictional from here on
    "2026-03-18": 1.75,
}, -0.40)
FED_UPPER = steps({
    "2018-03-22": 1.75, "2018-06-14": 2.00, "2018-09-27": 2.25, "2018-12-20": 2.50,
    "2019-08-01": 2.25, "2019-09-19": 2.00, "2019-10-31": 1.75, "2020-03-04": 1.25,
    "2020-03-16": 0.25, "2022-03-17": 0.50, "2022-05-05": 1.00, "2022-06-16": 1.75,
    "2022-07-28": 2.50, "2022-09-22": 3.25, "2022-11-03": 4.00, "2022-12-15": 4.50,
    "2023-02-02": 4.75, "2023-03-23": 5.00, "2023-05-04": 5.25, "2023-07-27": 5.50,
    "2024-09-19": 5.00, "2024-11-08": 4.75, "2024-12-19": 4.50, "2025-09-18": 4.25,
    "2025-10-30": 4.00, "2025-12-11": 3.75,
    # fictional from here on
    "2026-01-29": 3.75, "2026-03-19": 3.50,
}, 1.50)

eps_rates = rng.multivariate_normal([0, 0], [[1, 0.55], [0.55, 1]], size=N)
eps_eq_global = rng.standard_normal(N)

# 2Y yields: expected policy over ~6 months + noise
DE_2Y = 0.5 * ECB_DEPO + 0.5 * shift_forward(ECB_DEPO, 125) + 0.15 + ou(N, 0.035, 0.03, eps_rates[:, 0])
US_2Y = 0.5 * FED_UPPER + 0.5 * shift_forward(FED_UPPER, 125) - 0.15 + ou(N, 0.05, 0.03, eps_rates[:, 1])

DE_10Y = anchors({
    "2018-01-01": 0.45, "2018-10-01": 0.50, "2019-08-30": -0.65, "2020-03-09": -0.80,
    "2020-12-31": -0.55, "2021-12-31": -0.15, "2022-06-15": 1.70, "2022-10-20": 2.35,
    "2023-03-01": 2.70, "2023-10-04": 2.95, "2023-12-29": 2.00, "2024-12-03": 2.05,
    "2025-03-10": 2.85, "2025-06-30": 2.55, "2025-10-31": 2.62, "2025-12-31": 2.80,
    "2026-02-27": 2.70, "2026-03-31": 2.85,
}) + ou(N, 0.035, 0.02, 0.6 * eps_rates[:, 0] + 0.8 * rng.standard_normal(N))
US_10Y = anchors({
    "2018-01-01": 2.45, "2018-10-08": 3.20, "2019-08-30": 1.50, "2020-03-09": 0.55,
    "2020-08-04": 0.52, "2021-03-31": 1.70, "2021-12-31": 1.50, "2022-06-14": 3.45,
    "2022-10-21": 4.20, "2023-04-05": 3.30, "2023-10-19": 4.95, "2023-12-29": 3.88,
    "2024-04-25": 4.65, "2024-09-16": 3.62, "2025-01-14": 4.78, "2025-04-04": 4.00,
    "2025-05-21": 4.58, "2025-09-30": 4.10, "2025-12-31": 4.15, "2026-03-31": 4.25,
}) + ou(N, 0.05, 0.02, 0.6 * eps_rates[:, 1] + 0.8 * rng.standard_normal(N))

# Sovereign 10Y spreads vs Bund (percentage points)
SPREAD_10Y = {
    "FR": anchors({"2018-01-01": 0.35, "2020-03-18": 0.65, "2021-06-30": 0.35, "2022-12-31": 0.55,
                   "2024-06-10": 0.55, "2024-06-17": 0.80, "2024-12-03": 0.88, "2025-09-08": 0.85,
                   "2025-10-31": 0.80, "2026-03-31": 0.72}) + ou(N, 0.012, 0.05),
    "IT": anchors({"2018-01-01": 1.45, "2018-05-29": 2.60, "2018-10-19": 3.25, "2019-08-20": 2.30,
                   "2019-12-31": 1.60, "2020-03-18": 2.70, "2021-02-15": 0.95, "2022-06-14": 2.40,
                   "2022-10-14": 2.40, "2023-12-29": 1.65, "2024-12-31": 1.15, "2025-06-30": 0.90,
                   "2026-03-31": 0.80}) + ou(N, 0.03, 0.05),
    "ES": anchors({"2018-01-01": 1.05, "2018-12-31": 1.15, "2020-03-18": 1.30, "2021-06-30": 0.65,
                   "2022-10-14": 1.10, "2023-12-29": 0.95, "2025-06-30": 0.65, "2026-03-31": 0.55})
          + ou(N, 0.018, 0.05),
}
SPREAD_10Y["DE"] = np.zeros(N)

# Corporate IG spreads (OAS, percentage points)
EUR_IG_OAS = anchors({"2018-01-01": 0.85, "2018-12-31": 1.50, "2019-12-31": 1.05, "2020-03-24": 2.35,
                      "2020-12-31": 0.95, "2021-12-31": 0.95, "2022-10-12": 2.40, "2023-03-20": 1.90,
                      "2023-12-29": 1.40, "2024-12-31": 1.05, "2025-04-08": 1.35, "2025-12-31": 0.88,
                      "2026-03-31": 0.95}) + ou(N, 0.012, 0.05)
US_IG_OAS = anchors({"2018-01-01": 0.95, "2018-12-31": 1.55, "2019-12-31": 1.00, "2020-03-23": 3.70,
                     "2020-12-31": 0.98, "2021-12-31": 0.92, "2022-10-14": 1.60, "2023-03-17": 1.65,
                     "2023-12-29": 1.00, "2024-12-31": 0.80, "2025-04-08": 1.20, "2025-12-31": 0.78,
                     "2026-03-31": 0.85}) + ou(N, 0.012, 0.05)

ESTR = ECB_DEPO - 0.09
SOFR = FED_UPPER - 0.18 + ou(N, 0.01, 0.3)
EURIBOR_3M = 0.6 * ESTR + 0.4 * shift_forward(ESTR, 63) + 0.12 + ou(N, 0.01, 0.1)


def tenor_yield(y2: np.ndarray, y10: np.ndarray, maturity: float) -> np.ndarray:
    """Crude curve interpolation from 2Y and 10Y."""
    w = np.clip((maturity - 2) / 8, -0.2, 1.1)
    return y2 + w * (y10 - y2)


def bond_returns(yld: np.ndarray, duration: float) -> np.ndarray:
    """Daily total return of a constant-duration bond index: carry - D * dy."""
    dy = np.diff(yld, prepend=yld[0]) / 100
    carry = np.concatenate([[0], yld[:-1]]) / 100 / 252
    return carry - duration * dy + 0.5 * duration**2 * dy**2


# --------------------------------------------------------------------------- asset universe (23)
ASSETS = []  # (ticker, name, asset_class, jurisdiction, currency, duration, returns)


def add(ticker, name, cls, jur, ccy, dur, ret):
    ASSETS.append((ticker, name, cls, jur, ccy, dur, ret))


GOV = [  # ticker, jurisdiction, bucket label, maturity, duration, spread factor source
    ("DE_GOV_1_3Y", "DE", "1-3Y", 2.0, 1.9), ("DE_GOV_3_5Y", "DE", "3-5Y", 4.0, 3.8),
    ("DE_GOV_7_10Y", "DE", "7-10Y", 8.5, 7.6), ("DE_GOV_10Y_PLUS", "DE", "10Y+", 20.0, 16.5),
    ("FR_GOV_1_3Y", "FR", "1-3Y", 2.0, 1.9), ("FR_GOV_7_10Y", "FR", "7-10Y", 8.5, 7.6),
    ("IT_GOV_1_3Y", "IT", "1-3Y", 2.0, 1.9), ("IT_GOV_7_10Y", "IT", "7-10Y", 8.5, 7.4),
]
for tk, jur, bucket, mat, dur in GOV:
    spread = SPREAD_10Y[jur] * (0.45 if mat < 5 else 1.0) + (0.15 if mat > 10 else 0.0)
    add(tk, f"{jur} Government Bond {bucket}", "Government Bond", jur, "EUR", dur,
        bond_returns(tenor_yield(DE_2Y, DE_10Y, mat) + spread, dur))
# Euro area aggregates: Bund curve + a GDP-weighted share of the FR/IT/ES spreads
ea_spread = 0.25 * SPREAD_10Y["FR"] + 0.2 * SPREAD_10Y["IT"] + 0.12 * SPREAD_10Y["ES"]
for tk, bucket, mat, dur in [("EA_GOV_ALL", "All Maturities", 7.0, 7.2), ("EA_GOV_1_3Y", "1-3Y", 2.0, 1.9),
                             ("EA_GOV_7_10Y", "7-10Y", 8.5, 7.5)]:
    add(tk, f"Euro Area Government Bond {bucket}", "Government Bond", "EA", "EUR", dur,
        bond_returns(tenor_yield(DE_2Y, DE_10Y, mat) + ea_spread * (0.5 if mat < 5 else 1.0), dur))
for tk, bucket, mat, dur in [("US_GOV_1_3Y", "1-3Y", 2.0, 1.9), ("US_GOV_3_5Y", "3-5Y", 4.0, 3.8),
                             ("US_GOV_7_10Y", "7-10Y", 8.5, 7.4), ("US_GOV_10Y_PLUS", "10Y+", 20.0, 15.8),
                             ("US_GOV_ALL", "All Maturities", 7.0, 6.0)]:
    add(tk, f"US Treasury {bucket}", "Government Bond", "US", "USD", dur,
        bond_returns(tenor_yield(US_2Y, US_10Y, mat), dur))

eur_corp_y = tenor_yield(DE_2Y, DE_10Y, 5) + 0.25 + EUR_IG_OAS
add("EUR_CORP_IG", "EUR Corporate Investment Grade", "Corporate Bond", "EA", "EUR", 4.5,
    bond_returns(eur_corp_y, 4.5))
add("US_CORP_IG", "US Corporate Investment Grade", "Corporate Bond", "US", "USD", 6.9,
    bond_returns(tenor_yield(US_2Y, US_10Y, 9) + US_IG_OAS, 6.9))

# Equities: trend shaped by episode drifts + factor noise with stochastic vol
EPISODES = [  # (start, end, total log-return [unused for equities now], vol multiplier)
    ("2018-01-29", "2018-02-08", -0.10, 2.5), ("2018-10-03", "2018-12-24", -0.18, 1.8),
    ("2019-01-02", "2019-04-30", 0.15, 1.0), ("2020-02-20", "2020-03-23", -0.42, 4.5),
    ("2020-03-24", "2020-08-31", 0.38, 2.0), ("2020-11-09", "2020-11-30", 0.08, 1.2),
    ("2022-01-04", "2022-06-16", -0.24, 1.8), ("2022-08-17", "2022-10-12", -0.15, 1.8),
    ("2022-10-13", "2023-07-31", 0.22, 1.1), ("2023-08-01", "2023-10-27", -0.10, 1.2),
    ("2023-10-30", "2024-03-28", 0.22, 0.8), ("2024-08-01", "2024-08-05", -0.07, 2.5),
    ("2025-04-02", "2025-04-08", -0.15, 3.5), ("2025-04-09", "2025-06-30", 0.17, 1.6),
    ("2025-11-03", "2025-11-20", -0.06, 1.6),  # fictional
    ("2026-01-12", "2026-02-27", 0.06, 0.9),   # fictional
    ("2026-03-02", "2026-03-20", -0.07, 1.8),  # fictional
]
episode_drift = np.zeros(N)
vol_mult = np.ones(N)
for s, e, tot, vm in EPISODES:
    mask = (DATES >= s) & (DATES <= e)
    episode_drift[mask] += tot / mask.sum()
    vol_mult[mask] = vm
vol_mult = pd.Series(vol_mult).ewm(span=10).mean().to_numpy()  # vol decays smoothly

# Equity total-return paths pinned to anchor points (shape of 2018-26, NOT real levels);
# realistic daily noise comes from a mean-reverting deviation around the anchor path.
ES_PATH = anchors({"2018-01-01": 100, "2018-01-23": 104, "2018-02-09": 96, "2018-12-27": 85,
                   "2019-12-31": 105, "2020-02-19": 112, "2020-03-18": 70, "2020-08-31": 92,
                   "2021-12-31": 125, "2022-03-07": 108, "2022-09-29": 100, "2023-07-31": 135,
                   "2023-10-27": 125, "2024-03-28": 150, "2024-12-31": 152, "2025-03-03": 172,
                   "2025-04-07": 150, "2025-06-30": 168, "2025-10-31": 182, "2025-11-20": 175,
                   "2026-01-30": 190, "2026-03-20": 177, "2026-03-31": 180})
SP_PATH = anchors({"2018-01-01": 100, "2018-01-26": 107, "2018-02-08": 97, "2018-09-20": 110,
                   "2018-12-24": 88, "2019-12-31": 128, "2020-02-19": 138, "2020-03-23": 92,
                   "2020-08-31": 138, "2021-12-31": 185, "2022-06-16": 145, "2022-10-12": 138,
                   "2023-07-31": 175, "2023-10-27": 160, "2024-07-16": 212, "2024-08-05": 200,
                   "2024-12-31": 228, "2025-02-19": 238, "2025-04-08": 192, "2025-06-30": 236,
                   "2025-10-31": 258, "2025-11-20": 248, "2026-01-30": 265, "2026-03-20": 249,
                   "2026-03-31": 253})
DAX_REL = anchors({"2018-01-01": 1.0, "2019-12-31": 0.97, "2020-12-31": 0.99, "2022-09-29": 0.95,
                   "2024-12-31": 1.02, "2025-06-30": 1.08, "2026-03-31": 1.09})
EQUITIES = [  # ticker, name, jur, ccy, anchor path, base vol, beta to global factor
    ("EUROSTOXX50", "EURO STOXX 50", "EA", "EUR", ES_PATH, 0.16, 0.90),
    ("DAX", "DAX", "DE", "EUR", ES_PATH * DAX_REL, 0.17, 0.85),
    ("SP500", "S&P 500", "US", "USD", SP_PATH, 0.16, 0.80),
]
for tk, name, jur, ccy, path, vol, beta in EQUITIES:
    shock = beta * eps_eq_global + np.sqrt(1 - beta**2) * rng.standard_normal(N)
    dev = np.zeros(N)
    daily_vol = vol / np.sqrt(252) * vol_mult**0.6  # crashes are already in the anchor path
    for t in range(1, N):
        dev[t] = 0.90 * dev[t - 1] + daily_vol[t] * shock[t]
    log_level = np.log(path) + dev
    r = np.expm1(np.diff(log_level, prepend=log_level[0]))
    add(tk, name, "Equity", jur, ccy, None, r)

add("EUR_CASH", "EUR Cash (ESTR compounded)", "Cash", "EA", "EUR", 0.0, ESTR / 100 / 252)
add("US_CASH", "USD Cash (SOFR compounded)", "Cash", "US", "USD", 0.0, SOFR / 100 / 252)

assert len(ASSETS) == 23, len(ASSETS)

returns = pd.DataFrame({a[0]: a[6] for a in ASSETS}, index=DATES)
returns.iloc[0] = 0.0
levels = 100 * (1 + returns).cumprod()
levels.index.name = "date"

assets_meta = pd.DataFrame([
    {"ticker": t, "name": n, "asset_class": c, "jurisdiction": j, "currency": ccy,
     "duration": d, "data_type": "total return index level (local currency)",
     "description": f"SYNTHETIC {n} total return index, base 100 on {START}"}
    for t, n, c, j, ccy, d, _ in ASSETS
])

# --------------------------------------------------------------------------- financial.csv
sp = levels["SP500"].to_numpy()
sp_ret = np.diff(np.log(sp), prepend=np.log(sp[0]))
realised = pd.Series(sp_ret).rolling(21, min_periods=5).std().bfill().to_numpy() * np.sqrt(252) * 100
VIX = np.clip(0.75 * realised + 6 + ou(N, 0.8, 0.1), 9, 85)
sx = levels["EUROSTOXX50"].to_numpy()
sx_ret = np.diff(np.log(sx), prepend=np.log(sx[0]))
VSTOXX = np.clip(0.75 * pd.Series(sx_ret).rolling(21, min_periods=5).std().bfill().to_numpy()
                 * np.sqrt(252) * 100 + 7 + ou(N, 0.8, 0.1), 10, 90)
EURUSD = anchors({"2018-01-01": 1.20, "2018-02-15": 1.25, "2018-11-12": 1.13, "2019-09-30": 1.09,
                  "2020-03-20": 1.07, "2021-01-06": 1.23, "2022-09-27": 0.96, "2023-07-17": 1.12,
                  "2023-10-03": 1.05, "2024-09-30": 1.12, "2025-01-13": 1.02, "2025-07-01": 1.18,
                  "2025-11-05": 1.15, "2026-01-27": 1.19, "2026-03-31": 1.16}) * np.exp(ou(N, 0.004, 0.02))
BRENT = anchors({"2018-01-01": 66, "2018-10-03": 86, "2018-12-24": 50, "2019-12-31": 66, "2020-04-21": 19,
                 "2021-10-26": 86, "2022-03-08": 128, "2022-06-14": 122, "2022-12-09": 76, "2023-09-27": 96,
                 "2024-12-31": 74, "2025-04-09": 59, "2025-06-20": 77, "2025-12-31": 62,
                 "2026-03-31": 66}) * np.exp(ou(N, 0.018, 0.03))
GOLD = anchors({"2018-01-01": 1300, "2018-08-16": 1175, "2019-12-31": 1520, "2020-08-06": 2060,
                "2021-03-08": 1680, "2022-03-08": 2050, "2022-09-28": 1620, "2023-12-28": 2070,
                "2024-10-30": 2790, "2025-04-22": 3450, "2025-10-20": 4300, "2025-12-31": 4250,
                "2026-03-31": 4600}) * np.exp(ou(N, 0.008, 0.03))
CDS = {
    "CDS_DE_5Y": 12 + 4 * (EUR_IG_OAS - 1) + ou(N, 0.4, 0.05),
    "CDS_FR_5Y": 20 + 40 * SPREAD_10Y["FR"] + ou(N, 0.8, 0.05),
    "CDS_IT_5Y": 30 + 55 * SPREAD_10Y["IT"] + ou(N, 2.0, 0.05),
    "CDS_US_5Y": 20 + 8 * (US_IG_OAS - 1) + ou(N, 0.6, 0.05),
}

fin_series = {
    # id: (values, jurisdiction, category, indicator, field, description)
    "EURUSD": (EURUSD, "EA", "FX", "EUR/USD spot", "PX_LAST", "USD per 1 EUR"),
    "ESTR": (ESTR, "EA", "Money market", "Euro short-term rate", "RATE", "Overnight rate, % p.a."),
    "SOFR": (SOFR, "US", "Money market", "Secured overnight financing rate", "RATE", "Overnight rate, % p.a."),
    "EURIBOR_3M": (EURIBOR_3M, "EA", "Money market", "Euribor 3M", "RATE", "% p.a."),
    "EUROSTOXX50_PX": (sx, "EA", "Equity index", "EURO STOXX 50", "PX_LAST", "Index level (rebased)"),
    "SP500_PX": (sp, "US", "Equity index", "S&P 500", "PX_LAST", "Index level (rebased)"),
    "VIX": (VIX, "US", "Volatility", "CBOE VIX", "PX_LAST", "Implied vol, index points"),
    "VSTOXX": (VSTOXX, "EA", "Volatility", "VSTOXX", "PX_LAST", "Implied vol, index points"),
    "DE_10Y_YLD": (DE_10Y, "DE", "Government bond", "Bund 10Y", "YIELD", "Yield to maturity, %"),
    "DE_2Y_YLD": (DE_2Y, "DE", "Government bond", "Schatz 2Y", "YIELD", "Yield to maturity, %"),
    "IT_10Y_YLD": (DE_10Y + SPREAD_10Y["IT"], "IT", "Government bond", "BTP 10Y", "YIELD", "Yield, %"),
    "FR_10Y_YLD": (DE_10Y + SPREAD_10Y["FR"], "FR", "Government bond", "OAT 10Y", "YIELD", "Yield, %"),
    "US_10Y_YLD": (US_10Y, "US", "Government bond", "UST 10Y", "YIELD", "Yield to maturity, %"),
    "US_2Y_YLD": (US_2Y, "US", "Government bond", "UST 2Y", "YIELD", "Yield to maturity, %"),
    "BUND_FUT": (130 - 8.5 * (DE_10Y - 0.5) + ou(N, 0.05, 0.2), "DE", "Bond future", "Euro-Bund future",
                 "PX_LAST", "Front-month continuous price"),
    "EUR_SWAP_2Y": (DE_2Y + 0.25 + ou(N, 0.01, 0.1), "EA", "Swap", "EUR IRS 2Y", "RATE", "% p.a."),
    "EUR_SWAP_10Y": (DE_10Y + 0.15 + ou(N, 0.01, 0.1), "EA", "Swap", "EUR IRS 10Y", "RATE", "% p.a."),
    "USD_SWAP_2Y": (US_2Y - 0.1 + ou(N, 0.01, 0.1), "US", "Swap", "USD SOFR swap 2Y", "RATE", "% p.a."),
    "USD_SWAP_10Y": (US_10Y - 0.4 + ou(N, 0.01, 0.1), "US", "Swap", "USD SOFR swap 10Y", "RATE", "% p.a."),
    **{k: (v, k.split("_")[1], "CDS", f"{k.split('_')[1]} sovereign CDS 5Y", "SPREAD_BP", "bp")
       for k, v in CDS.items()},
    "ITRAXX_MAIN": (55 + 45 * (EUR_IG_OAS - 1) + ou(N, 0.8, 0.05), "EA", "CDS", "iTraxx Europe Main 5Y",
                    "SPREAD_BP", "bp"),
    "CDX_IG": (55 + 30 * (US_IG_OAS - 1) + ou(N, 0.8, 0.05), "US", "CDS", "CDX North America IG 5Y",
               "SPREAD_BP", "bp"),
    "EUR_IG_OAS": (EUR_IG_OAS * 100, "EA", "Corporate bond", "EUR IG corporate index", "OAS_BP", "bp"),
    "US_IG_OAS": (US_IG_OAS * 100, "US", "Corporate bond", "US IG corporate index", "OAS_BP", "bp"),
    "BRENT": (BRENT, "GLOBAL", "Commodity", "Brent crude", "PX_LAST", "USD/bbl"),
    "GOLD": (GOLD, "GLOBAL", "Commodity", "Gold spot", "PX_LAST", "USD/oz"),
}
fin_frames = []
for sid, (vals, *_rest) in fin_series.items():
    s = pd.Series(np.round(vals, 4), index=DATES)
    if sid == "SOFR":
        s[s.index < "2018-04-03"] = np.nan   # coverage starts later, like the real series
    if sid == "ESTR":
        s[s.index < "2019-10-02"] = np.nan
    drop = rng.random(N) < 0.01             # random missing days (holidays, gaps)
    s[drop] = np.nan
    fin_frames.append(pd.DataFrame({"date": DATES, "series_id": sid, "value": s.to_numpy()}).dropna())
financial = pd.concat(fin_frames, ignore_index=True)
financial_meta = pd.DataFrame([
    {"series_id": sid, "jurisdiction": j, "category": c, "indicator": ind, "field": f,
     "description": f"SYNTHETIC. {d}"}
    for sid, (_v, j, c, ind, f, d) in fin_series.items()
])

# --------------------------------------------------------------------------- macro.csv
M_DATES = pd.date_range("2010-01-31", "2026-03-31", freq="ME")
Q_DATES = pd.date_range("2010-03-31", "2026-03-31", freq="QE")
NM, NQ = len(M_DATES), len(Q_DATES)


def m_anchor(points, sd=0.0, kappa=0.3, dates=M_DATES):
    return anchors(points, dates) + (ou(len(dates), sd, kappa) if sd else 0)


EA_HICP = m_anchor({"2010-01-31": 1.0, "2011-09-30": 3.0, "2013-12-31": 0.8, "2015-01-31": -0.6,
                    "2017-01-31": 1.8, "2018-10-31": 2.2, "2019-12-31": 1.3, "2020-12-31": -0.3,
                    "2021-12-31": 5.0, "2022-10-31": 10.6, "2023-10-31": 2.9, "2024-09-30": 1.7,
                    "2025-01-31": 2.5, "2025-06-30": 2.0, "2025-10-31": 2.1, "2026-01-31": 1.8,
                    "2026-03-31": 2.2}, 0.12)
US_CPI = m_anchor({"2010-01-31": 2.6, "2011-09-30": 3.9, "2015-04-30": -0.2, "2018-07-31": 2.9,
                   "2019-12-31": 2.3, "2020-05-31": 0.1, "2022-06-30": 9.1, "2023-06-30": 3.0,
                   "2024-09-30": 2.4, "2025-04-30": 2.3, "2025-09-30": 3.0, "2026-01-31": 2.7,
                   "2026-03-31": 2.8}, 0.12)
macro_series = {
    # id: (dates, values, jurisdiction, indicator, frequency, unit, data_type)
    "EA_HICP_YOY": (M_DATES, EA_HICP, "EA", "HICP headline inflation", "M", "% y/y", "Inflation"),
    "EA_CORE_HICP_YOY": (M_DATES, 0.55 * EA_HICP + 0.6 + ou(NM, 0.08, 0.3), "EA", "HICP core inflation",
                         "M", "% y/y", "Inflation"),
    **{f"{c}_HICP_YOY": (M_DATES, EA_HICP + off + ou(NM, 0.25, 0.3), c, "HICP headline inflation", "M",
                         "% y/y", "Inflation")
       for c, off in [("DE", 0.2), ("FR", -0.4), ("IT", 0.1)]},
    "US_CPI_YOY": (M_DATES, US_CPI, "US", "CPI headline inflation", "M", "% y/y", "Inflation"),
    "US_CORE_CPI_YOY": (M_DATES, 0.6 * US_CPI + 1.0 + ou(NM, 0.08, 0.3), "US", "CPI core inflation", "M",
                        "% y/y", "Inflation"),
    "EA_UNEMP": (M_DATES, m_anchor({"2010-01-31": 10.2, "2013-04-30": 12.1, "2019-12-31": 7.4,
                                    "2020-08-31": 8.6, "2022-12-31": 6.7, "2025-06-30": 6.2,
                                    "2026-03-31": 6.3}, 0.05), "EA", "Unemployment rate", "M", "%",
                 "Labour market"),
    "US_UNEMP": (M_DATES, m_anchor({"2010-01-31": 9.8, "2015-12-31": 5.0, "2019-12-31": 3.6,
                                    "2020-03-31": 4.4, "2020-04-30": 14.7, "2021-06-30": 5.9,
                                    "2023-04-30": 3.4, "2024-07-31": 4.3, "2025-09-30": 4.4,
                                    "2026-03-31": 4.5}, 0.05), "US", "Unemployment rate", "M", "%",
                 "Labour market"),
    "EA_PMI_MANUF": (M_DATES, m_anchor({"2010-01-31": 52, "2012-07-31": 44, "2017-12-31": 60.6,
                                        "2019-09-30": 45.7, "2020-04-30": 33.4, "2021-06-30": 63.4,
                                        "2023-07-31": 42.7, "2024-12-31": 45.1, "2025-08-31": 50.7,
                                        "2025-12-31": 49.6, "2026-03-31": 50.4}, 0.6), "EA",
                     "Manufacturing PMI", "M", "index", "Manufacturing"),
    "US_ISM_MANUF": (M_DATES, m_anchor({"2010-01-31": 57, "2015-12-31": 48, "2018-08-31": 60.8,
                                        "2019-12-31": 47.8, "2020-04-30": 41.5, "2021-03-31": 64.7,
                                        "2023-06-30": 46.0, "2025-01-31": 50.9, "2025-10-31": 48.7,
                                        "2026-03-31": 49.5}, 0.8), "US", "ISM manufacturing", "M",
                     "index", "Manufacturing"),
    "EA_CONS_CONF": (M_DATES, m_anchor({"2010-01-31": -16, "2012-11-30": -26, "2018-01-31": 0.5,
                                        "2020-04-30": -22, "2021-06-30": -3, "2022-09-30": -28.7,
                                        "2024-12-31": -14.5, "2025-04-30": -16.7, "2026-03-31": -14.0},
                                       0.6), "EA", "Consumer confidence", "M", "balance", "Sentiment"),
    "US_CONS_SENT": (M_DATES, m_anchor({"2010-01-31": 74, "2018-03-31": 101, "2020-04-30": 71.8,
                                        "2022-06-30": 50.0, "2024-03-31": 79.4, "2025-04-30": 52.2,
                                        "2025-11-30": 51.0, "2026-03-31": 56.0}, 1.5), "US",
                     "Consumer sentiment (UMich)", "M", "index", "Sentiment"),
    "EA_M3_YOY": (M_DATES, m_anchor({"2010-01-31": 0.2, "2015-12-31": 5.0, "2019-12-31": 5.0,
                                     "2021-01-31": 12.5, "2023-08-31": -1.2, "2024-12-31": 3.5,
                                     "2026-03-31": 2.8}, 0.2), "EA", "Monetary aggregate M3", "M",
                  "% y/y", "Money"),
    "ECB_DEPO_RATE": (M_DATES, steps({"2012-07-11": 0.0, "2014-06-11": -0.10, "2014-09-10": -0.20,
                                      "2015-12-09": -0.30, "2016-03-16": -0.40}, 0.25, M_DATES), "EA",
                      "ECB deposit facility rate", "M", "%", "Policy rate"),
    "FED_FUNDS_UPPER": (M_DATES, steps({"2008-12-16": 0.25, "2015-12-17": 0.50, "2016-12-15": 0.75,
                                        "2017-03-16": 1.00, "2017-06-15": 1.25, "2017-12-14": 1.50},
                                       0.25, M_DATES), "US", "Fed funds target upper bound", "M", "%",
                        "Policy rate"),
    "EA_GDP_QOQ": (Q_DATES, m_anchor({"2010-03-31": 0.4, "2012-06-30": -0.3, "2019-12-31": 0.1,
                                      "2020-03-31": -3.5, "2020-06-30": -11.4, "2020-09-30": 12.4,
                                      "2021-06-30": 2.1, "2022-12-31": -0.1, "2024-06-30": 0.2,
                                      "2025-03-31": 0.6, "2025-06-30": 0.1, "2025-12-31": 0.2,
                                      "2026-03-31": 0.3}, 0.1, 0.5, Q_DATES), "EA", "Real GDP growth", "Q",
                   "% q/q", "GDP"),
    "DE_GDP_QOQ": (Q_DATES, m_anchor({"2010-03-31": 0.8, "2019-12-31": 0.0, "2020-06-30": -9.7,
                                      "2020-09-30": 8.9, "2023-12-31": -0.3, "2025-06-30": -0.2,
                                      "2026-03-31": 0.2}, 0.15, 0.5, Q_DATES), "DE", "Real GDP growth", "Q",
                   "% q/q", "GDP"),
    "US_GDP_SAAR": (Q_DATES, m_anchor({"2010-03-31": 2.0, "2019-12-31": 2.4, "2020-03-31": -5.5,
                                       "2020-06-30": -28.0, "2020-09-30": 34.8, "2022-06-30": -0.6,
                                       "2023-09-30": 4.4, "2025-03-31": -0.5, "2025-06-30": 3.8,
                                       "2025-12-31": 1.4, "2026-03-31": 2.0}, 0.4, 0.5, Q_DATES), "US",
                    "Real GDP growth", "Q", "% q/q saar", "GDP"),
}
# Policy rates after 2018 follow the daily paths above
for sid, daily in [("ECB_DEPO_RATE", ECB_DEPO), ("FED_FUNDS_UPPER", FED_UPPER)]:
    d, v, *rest = macro_series[sid]
    v = v.copy()
    after = d >= DATES[0]
    v[after] = pd.Series(daily, index=DATES).reindex(d[after], method="ffill").to_numpy()
    macro_series[sid] = (d, v, *rest)

macro_frames = []
for sid, (d, v, *_r) in macro_series.items():
    s = pd.Series(np.round(v, 2), index=d)
    if sid in ("IT_HICP_YOY", "EA_M3_YOY"):
        s[s.index < "2012-01-01"] = np.nan   # coverage varies by indicator
    macro_frames.append(pd.DataFrame({"date": d, "series_id": sid, "value": s.to_numpy()}).dropna())
macro = pd.concat(macro_frames, ignore_index=True)
macro_meta = pd.DataFrame([
    {"series_id": sid, "jurisdiction": j, "indicator": ind, "frequency": f, "unit": u, "data_type": dt,
     "description": f"SYNTHETIC. {ind} ({j}). Date = end of reference period, NOT release date."}
    for sid, (_d, _v, j, ind, f, u, dt) in macro_series.items()
])

# --------------------------------------------------------------------------- news.csv
SOURCE = "SYNTHETIC newswire"
news = []


def emit(ts, headline, body, topic):
    news.append({"published_at": pd.Timestamp(ts), "headline": headline, "body": body, "topic": topic,
                 "source": SOURCE})


def direction(new, old):
    return "raises" if new > old else "cuts"


for name, path, who, where in [("ECB", ECB_DEPO, "European Central Bank", "Frankfurt"),
                               ("Fed", FED_UPPER, "Federal Reserve", "Washington")]:
    chg = np.where(np.diff(path) != 0)[0] + 1
    for i in chg:
        d, new, old = DATES[i], path[i], path[i - 1]
        bp = abs(new - old) * 100
        emit(d + pd.Timedelta(hours=13 if name == "ECB" else 19),
             f"{name} {direction(new, old)} rates by {bp:.0f} bp to {new:.2f}%",
             f"{where} - The {who} {direction(new, old)} its key policy rate by {bp:.0f} basis points to "
             f"{new:.2f}% on {d:%d %B %Y}, citing the outlook for inflation and growth. Policymakers said "
             f"future decisions would remain data-dependent.", "central_banks")
    # pre-meeting previews ~1 week before each change
    for i in chg:
        d = DATES[max(i - 5, 0)]
        emit(d + pd.Timedelta(hours=8), f"Markets price {name} move next week as officials signal shift",
             f"Money markets fully price a policy move at the {who}'s next meeting, after several "
             f"policymakers signalled their stance in recent speeches.", "central_banks")

# data releases: published ~2-6 weeks AFTER the reference period end (release lag!)
for sid, label, lag_days, topic in [("EA_HICP_YOY", "Euro zone inflation", 1, "macro"),
                                    ("US_CPI_YOY", "U.S. consumer prices", 13, "macro"),
                                    ("EA_PMI_MANUF", "Euro zone factory PMI", 1, "macro"),
                                    ("US_UNEMP", "U.S. unemployment rate", 6, "macro")]:
    s = macro[macro.series_id == sid].set_index("date")["value"]
    s = s[s.index >= "2017-11-01"]
    prev = s.shift(1)
    for ref, val in s.items():
        pub = ref + pd.Timedelta(days=lag_days) + pd.Timedelta(hours=11)
        if pub > pd.Timestamp(END) + pd.Timedelta(hours=23):
            continue
        p = prev.get(ref)
        if p is None or np.isnan(p):
            continue
        verb = "steady at" if abs(val - p) < 0.05 else (
            "rises to" if val > p else "falls to")
        emit(pub, f"{label} {verb} {val:.1f} in {ref:%B}",
             f"{label} {verb} {val:.1f} in {ref:%B %Y} from {p:.1f} the previous "
             f"month, according to official data published on {pub:%d %B %Y}.", topic)

# market-move stories driven by the simulated data (same-day, after the close)
for tk, label, thr in [("SP500", "Wall Street", 0.025), ("EUROSTOXX50", "European shares", 0.025)]:
    r = returns[tk]
    for d, v in r[np.abs(r) > thr].items():
        verb = "tumble" if v < 0 else "surge"
        emit(d + pd.Timedelta(hours=22), f"{label} {verb} {abs(v) * 100:.1f}% as volatility jumps",
             f"{label} {verb}d {abs(v) * 100:.1f}% on {d:%d %B %Y}, with investors reacting to the latest "
             f"headlines on growth and policy.", "markets")
it_spread = pd.Series(SPREAD_10Y["IT"] * 100, index=DATES)
for d, v in it_spread.diff()[it_spread.diff().abs() > 12].items():
    emit(d + pd.Timedelta(hours=18),
         f"Italian bond spread {'widens' if v > 0 else 'tightens'} {abs(v):.0f} bp",
         f"The gap between Italian and German 10-year yields moved {abs(v):.0f} bp to "
         f"{it_spread[d]:.0f} bp on {d:%d %B %Y} amid budget and political news.", "rates")

# filler: generic daily background noise
FILLER = [
    ("Oil prices edge {d} on supply outlook", "commodities"),
    ("Tech shares {d} ahead of earnings season", "markets"),
    ("Dollar {d} against major currencies", "fx"),
    ("Gold {d} as investors weigh rate path", "commodities"),
    ("EU leaders meet to discuss fiscal rules", "politics"),
    ("Trade talks resume between Washington and Beijing", "politics"),
    ("Corporate bond issuance {d} in busy week", "credit"),
    ("Bank lending survey shows tighter credit standards", "macro"),
    ("Chipmaker shares {d} on AI demand", "technology"),
    ("Euro {d} after comments from policymakers", "fx"),
]
for d in DATES:
    for _ in range(rng.integers(1, 4)):
        tpl, topic = FILLER[rng.integers(len(FILLER))]
        word = ["rise", "fall", "steady", "slip", "climb"][rng.integers(5)]
        h = tpl.format(d=word)
        emit(d + pd.Timedelta(hours=int(rng.integers(6, 22)), minutes=int(rng.integers(60))), h,
             f"{h}. Analysts said the move reflected positioning ahead of key data.", topic)

news_df = pd.DataFrame(news)
news_df = news_df[news_df.published_at <= pd.Timestamp(END) + pd.Timedelta(hours=23, minutes=59)]
news_df = news_df.sort_values("published_at").reset_index(drop=True)

# --------------------------------------------------------------------------- export (official formats)
DAYFIRST = "%d/%m/%Y"
JUR = {"EA": "EUR", "GLOBAL": "US"}            # official jurisdictions: EUR, US, DE, IT, FR

# internal ticker -> (official index code, indicator name, category)
CATEGORY = {"Government Bond": "Gov Bonds", "Corporate Bond": "Corp Bonds", "Equity": "Equities", "Cash": "Cash"}
ASSET_CODES = {
    "DE_GOV_1_3Y": "IBGDE13E", "DE_GOV_3_5Y": "IBGDE35E", "DE_GOV_7_10Y": "IBGDE710E",
    "DE_GOV_10Y_PLUS": "IBGDE10PE", "FR_GOV_1_3Y": "IBGFR13E", "FR_GOV_7_10Y": "IBGFR710E",
    "IT_GOV_1_3Y": "IBGIT13E", "IT_GOV_7_10Y": "IBGIT710E", "EA_GOV_ALL": "IBBEU008E",
    "EA_GOV_1_3Y": "IBGEU13E", "EA_GOV_7_10Y": "IBGEU710E", "US_GOV_1_3Y": "IBGUS13U",
    "US_GOV_3_5Y": "IBGUS35U", "US_GOV_7_10Y": "IBGUS710U", "US_GOV_10Y_PLUS": "IBGUS10PU",
    "US_GOV_ALL": "IBGUS00U", "EUR_CORP_IG": "ICREUIGE", "US_CORP_IG": "ICRUSIGU",
    "EUROSTOXX50": "SX5T", "DAX": "DAXK", "SP500": "SPXT", "EUR_CASH": "ESTRCMPE", "US_CASH": "SOFRCMPU",
}
assets_long, assets_meta_rows = [], []
for tk, name, cls, jur, ccy, dur, _ in ASSETS:
    code = ASSET_CODES[tk]
    if cls == "Government Bond":   # the real metadata mislabels these levels as a yield
        field, desc = "TOT_RETURN", "Total Return Yield"
    else:
        field, desc = "tri_today", "Total return index level on the pricing date"
    assets_meta_rows.append({"jurisdiction": JUR.get(jur, jur), "category": CATEGORY[cls],
                             "indicator": f"{name} Total Return Index ({ccy}) - SYNTHETIC",
                             "index": code, "field": field, "description": desc})
    assets_long.append(pd.DataFrame({"date": DATES.strftime(DAYFIRST), "index": code,
                                     "price": levels[tk].round(4).to_numpy(), "price_description": field}))
assets_out = pd.concat(assets_long, ignore_index=True)
assets_meta_out = pd.DataFrame(assets_meta_rows)

FIN_CODES = {
    "EURUSD": "EURUSD Curncy", "ESTR": "ESTRON Index", "SOFR": "SOFRRATE Index", "EURIBOR_3M": "EUR003M Index",
    "EUROSTOXX50_PX": "SX5E Index", "SP500_PX": "SPX Index", "VIX": "VIX Index", "VSTOXX": "V2X Index",
    "DE_10Y_YLD": "GDBR10 Index", "DE_2Y_YLD": "GDBR2 Index", "IT_10Y_YLD": "GBTPGR10 Index",
    "FR_10Y_YLD": "GFRN10 Index", "US_10Y_YLD": "USGG10YR Index", "US_2Y_YLD": "USGG2YR Index",
    "BUND_FUT": "RX1 Comdty", "EUR_SWAP_2Y": "EUSA2 Curncy", "EUR_SWAP_10Y": "EUSA10 Curncy",
    "USD_SWAP_2Y": "USOSFR2 Curncy", "USD_SWAP_10Y": "USOSFR10 Curncy", "CDS_DE_5Y": "GRMN CDS USD SR 5Y",
    "CDS_FR_5Y": "FRTR CDS USD SR 5Y", "CDS_IT_5Y": "ITAL CDS USD SR 5Y", "CDS_US_5Y": "USGB CDS EUR SR 5Y",
    "ITRAXX_MAIN": "ITRXEBE Index", "CDX_IG": "CDXIG5 Index", "EUR_IG_OAS": "LECPOAS Index",
    "US_IG_OAS": "LUACOAS Index", "BRENT": "CO1 Comdty", "GOLD": "XAU Curncy",
}
FIN_CATEGORY = {"FX": "Currency", "Money market": "Money Market", "Equity index": "Equity Index",
                "Volatility": "Volatility", "Government bond": "Government Bond Yield", "Bond future": "Bond Futures",
                "Swap": "Interest Rate Swaps", "CDS": "Credit Default Swaps", "Corporate bond": "Corporate Bond Index",
                "Commodity": "Commodities"}
financial_out = financial.assign(date=financial["date"].dt.strftime(DAYFIRST),
                                 index=financial["series_id"].map(FIN_CODES),
                                 price_description="close_price")[["date", "index", "value", "price_description"]]
financial_out = financial_out.rename(columns={"value": "price"})
financial_meta_out = pd.DataFrame([
    {"jurisdiction": JUR.get(j, j), "category": FIN_CATEGORY[c], "indicator": f"{ind} - SYNTHETIC",
     "index": FIN_CODES[sid], "field": "close_price", "description": "Close Price"}
    for sid, (_v, j, c, ind, _f, _d) in fin_series.items()
])

# Macro: HICP/CPI become price INDICES (the real file has P023ZH-style HICP indices),
# dates move to the FIRST day of the reference period (PMI stays at the month end).
MACRO_CODES = {  # internal id -> (identifier, category, indicator, database, dataType, aggregation)
    "EA_HICP_YOY": ("P023ZH", "Prices", "HICP - Overall index", "ECB Data Portal", "INDEX", "Average"),
    "EA_CORE_HICP_YOY": ("P023ZX", "Prices", "HICP - All-items excluding energy and food", "ECB Data Portal",
                         "INDEX", "Average"),
    "DE_HICP_YOY": ("P023DE", "Prices", "HICP - Overall index", "ECB Data Portal", "INDEX", "Average"),
    "FR_HICP_YOY": ("P023FR", "Prices", "HICP - Overall index", "ECB Data Portal", "INDEX", "Average"),
    "IT_HICP_YOY": ("P023IT", "Prices", "HICP - Overall index", "ECB Data Portal", "INDEX", "Average"),
    "US_CPI_YOY": ("CPIAUC", "Prices", "CPI - All items", "FRED", "INDEX", "Average"),
    "US_CORE_CPI_YOY": ("CPILFE", "Prices", "CPI - All items less food and energy", "FRED", "INDEX", "Average"),
    "EA_UNEMP": ("LRHUTT", "Labour market", "Unemployment rate", "Eurostat", "PERCENT", "Average"),
    "US_UNEMP": ("UNRATE", "Labour market", "Unemployment rate", "FRED", "PERCENT", "Average"),
    "EA_PMI_MANUF": ("PMIMEZ", "Business surveys", "Manufacturing PMI", "S&P Global", "INDEX", "Average"),
    "US_ISM_MANUF": ("ISMMAN", "Business surveys", "ISM manufacturing PMI", "ISM", "INDEX", "Average"),
    "EA_CONS_CONF": ("CONFEZ", "Consumer surveys", "Consumer confidence indicator", "European Commission",
                     "BALANCE", "Average"),
    "US_CONS_SENT": ("UMCSNT", "Consumer surveys", "University of Michigan consumer sentiment", "FRED", "INDEX",
                     "Average"),
    "EA_M3_YOY": ("M3EZYY", "Money", "Monetary aggregate M3, annual growth rate", "ECB Data Portal",
                  "PERCENT_CHANGE", "End of period"),
    "ECB_DEPO_RATE": ("ECBDFR", "Policy rates", "ECB deposit facility rate", "ECB Data Portal", "PERCENT",
                      "End of period"),
    "FED_FUNDS_UPPER": ("FDTRUP", "Policy rates", "Fed funds target range, upper bound", "FRED", "PERCENT",
                        "End of period"),
    "EA_GDP_QOQ": ("GDPEZQ", "National accounts", "Real GDP, % change q/q", "Eurostat", "PERCENT_CHANGE", "Sum"),
    "DE_GDP_QOQ": ("GDPDEQ", "National accounts", "Real GDP, % change q/q", "Eurostat", "PERCENT_CHANGE", "Sum"),
    "US_GDP_SAAR": ("GDPUSQ", "National accounts", "Real GDP, % change q/q annualised", "FRED", "PERCENT_CHANGE",
                    "Sum"),
}
PMI_IDS = {"EA_PMI_MANUF", "US_ISM_MANUF"}


def yoy_to_index(yoy: pd.Series) -> pd.Series:
    """Price index whose 12-month change reproduces the y/y path (base ~100)."""
    idx = np.empty(len(yoy))
    for t, y in enumerate(yoy.to_numpy()):
        idx[t] = (100 * (1 + y / 100) ** (t / 12)) if t < 12 else idx[t - 12] * (1 + y / 100)
    return pd.Series(idx, index=yoy.index)


macro_out_frames, macro_meta_rows = [], []
for sid, (code, cat, ind, db, dtype, agg) in MACRO_CODES.items():
    g = macro[macro.series_id == sid].set_index("date")["value"]
    _d, _v, jur, _ind, freq, _u, _dt = macro_series[sid]
    if cat == "Prices":
        g = yoy_to_index(g).round(2)
    if sid in PMI_IDS:
        dates = g.index                                            # last day of the month
    else:
        dates = g.index.to_period("Q" if freq == "Q" else "M").start_time  # first day
    macro_out_frames.append(pd.DataFrame({"date": dates.strftime(DAYFIRST), "identifier": code,
                                          "value": g.to_numpy()}))
    macro_meta_rows.append({"identifier": code, "jurisdiction": JUR.get(jur, jur), "category": cat,
                            "indicator": f"{ind} - SYNTHETIC", "database": db,
                            "frequencyDescription": {"M": "Monthly", "Q": "Quarterly"}[freq],
                            "dataType": dtype, "aggregationTypeDescription": agg})
macro_out = pd.concat(macro_out_frames, ignore_index=True)
macro_meta_out = pd.DataFrame(macro_meta_rows)

# News: stories with versions and Reuters-style topic lists (exact names from the annex).
TOPIC_MAP = {
    "central_banks": ["Central Banks / Central Bank Events", "Central Bank Interest Rates / Policy",
                      "Monetary / Fiscal Policy / Policy Makers", "Economic News"],
    "macro": ["Economic News"],
    "markets": ["Equities", "Equities Markets"],
    "rates": ["Fixed Income", "Debt / Fixed Income Markets", "Italy", "Euro Zone"],
    "commodities": ["Commodities", "Commodities Markets"],
    "fx": ["Foreign Exchange"],
    "politics": ["Government / Politics"],
    "credit": ["Fixed Income", "Credit Markets"],
    "technology": ["Equities", "Technology"],
}
KEYWORD_TOPICS = [  # (word in the title, extra topics)
    ("ECB", ["European Central Bank", "Euro Zone"]), ("Fed", ["United States"]),
    ("inflation", ["Inflation Data"]), ("consumer prices", ["Inflation Data", "United States"]),
    ("PMI", ["Purchasing Managers Survey Data"]), ("unemployment", ["Employment / Unemployment Data / Policy"]),
    ("Euro zone", ["Euro Zone"]), ("U.S.", ["United States"]), ("Wall Street", ["United States"]),
    ("European shares", ["Europe"]), ("Oil", ["Crude Oil"]), ("Gold", ["Gold"]),
    ("Trade talks", ["International Trade"]), ("Dollar", ["United States"]), ("Euro ", ["Euro Zone"]),
]


def topics_for(title: str, topic: str) -> list[str]:
    out = list(TOPIC_MAP.get(topic, ["Major News"]))
    for word, extra in KEYWORD_TOPICS:
        if word.lower() in title.lower():
            out += [t for t in extra if t not in out]
    return out


rows = []
for i, r in enumerate(news_df.itertuples()):
    pub = r.published_at.tz_localize("UTC")
    sid, topics = f"SYN{i:07d}", topics_for(r.headline, r.topic)
    updated = r.topic == "central_banks" or rng.random() < 0.2
    base = {"story_id": sid, "topic_name": "|".join(topics), "has_body": True}
    rows.append({**base, "published_at": pub, "version_created": pub - pd.Timedelta(seconds=30), "version": 1,
                 "version_role": "first", "title": r.headline, "body": r.body})
    if updated:   # a later version of the same story (an update), sometimes the next day
        later = pub + (pd.Timedelta(hours=18) if rng.random() < 0.1 else
                       pd.Timedelta(minutes=int(rng.integers(5, 120))))
        rows.append({**base, "published_at": later, "version_created": later - pd.Timedelta(seconds=30),
                     "version": int(rng.integers(2, 4)), "version_role": "last",
                     "title": f"UPDATE 1-{r.headline}", "body": r.body + " Adds details and market reaction."})
news_out = pd.DataFrame(rows)
news_out = news_out[news_out.published_at <= pd.Timestamp(END, tz="UTC") + pd.Timedelta(hours=23, minutes=59)]
news_out = news_out.sort_values(["published_at", "story_id"]).reset_index(drop=True)
news_out.insert(1, "date", news_out["published_at"].dt.strftime("%Y-%m-%d"))
topic_counts = (news_out.drop_duplicates("story_id")["topic_name"].str.split("|").explode()
                .value_counts().rename_axis("topic").reset_index(name="count"))

# --------------------------------------------------------------------------- write
DATA_DIR.mkdir(parents=True, exist_ok=True)
for old in ["assets-metadata.csv", "financial-metadata.csv", "macro-metadata.csv"]:
    (DATA_DIR / old).unlink(missing_ok=True)            # pre-16-Nov file names
assets_out.to_csv(DATA_DIR / "assets.csv", index=False)
assets_meta_out.to_csv(DATA_DIR / "assets_metadata.csv", index=False, encoding="utf-8-sig")
financial_out.to_csv(DATA_DIR / "financial.csv", index=False)
financial_meta_out.to_csv(DATA_DIR / "financial_metadata.csv", index=False, encoding="utf-8-sig")
macro_out.to_csv(DATA_DIR / "macro.csv", index=False)
macro_meta_out.to_csv(DATA_DIR / "macro_metadata.csv", index=False, encoding="utf-8-sig")
news_out.to_csv(DATA_DIR / "news.csv", index=False)
topic_counts.to_csv(DATA_DIR / "news_topics.csv", index=False)

if __name__ == "__main__":
    print(f"Wrote synthetic data to {DATA_DIR}")
    print(f"  assets:    {levels.shape[0]} days x {levels.shape[1]} indices")
    print(f"  financial: {financial.series_id.nunique()} series, {len(financial)} rows")
    print(f"  macro:     {macro.series_id.nunique()} series, {len(macro)} rows")
    print(f"  news:      {news_out.story_id.nunique()} stories, {len(news_out)} versions, "
          f"{len(topic_counts)} topics")
    ann = returns.mean() * 252
    vol = returns.std() * np.sqrt(252)
    print("\nAnnualised return / vol (sanity check):")
    print(pd.DataFrame({"ret": ann, "vol": vol}).round(3).to_string())
