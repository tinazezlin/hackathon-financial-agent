"""Adapter: official data formats -> the fields our code works with.

This is the ONE module that knows the official files' column names, codes and
categories. Everything else (tools, analytics, risk review, report) works on the
normalised frames below. On the day, adapt this file (and nothing else) to the real data.

Asset metadata, normalised (assets_meta):
    ticker        official index code (IBBEU008E, ...): also the weights.csv column
    name          readable name for the LLM and the report
    asset_class   Government Bond | Corporate Bond | Equity | Cash   (official: Gov Bonds, ...)
    jurisdiction  EUR | US | DE | IT | FR                             (official, unchanged)
    region        US | EUR    (everything not US is a euro-area exposure)
    currency      USD for US, EUR otherwise (not in the official metadata)
    duration      ESTIMATED from the maturity in the name (not in the official metadata)
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import pandas as pd

from src.utils import data

ASSET_CLASS = {"Gov Bonds": "Government Bond", "Corp Bonds": "Corporate Bond", "Equities": "Equity",
               "Cash": "Cash"}

# Market indicators shown to the analyst: official code -> (label, unit). Codes are the
# synthetic ones (Bloomberg-like); replace with the real codes from financial_metadata.csv.
MARKET_SERIES = {
    "GDBR2 Index": ("DE 2Y yield", "%"), "GDBR10 Index": ("DE 10Y yield", "%"),
    "GFRN10 Index": ("FR 10Y yield", "%"), "GBTPGR10 Index": ("IT 10Y yield", "%"),
    "USGG2YR Index": ("US 2Y yield", "%"), "USGG10YR Index": ("US 10Y yield", "%"),
    "ESTRON Index": ("EUR overnight rate (ESTR)", "%"), "SOFRRATE Index": ("USD overnight rate (SOFR)", "%"),
    "EURUSD Curncy": ("EUR/USD (USD per EUR)", "level"), "VIX Index": ("VIX (US equity implied vol)", "pts"),
    "V2X Index": ("VSTOXX (EA equity implied vol)", "pts"), "LECPOAS Index": ("EUR IG corporate spread", "bp"),
    "LUACOAS Index": ("US IG corporate spread", "bp"), "ITRXEBE Index": ("iTraxx Europe Main 5Y", "bp"),
    "ITAL CDS USD SR 5Y": ("Italy 5Y CDS", "bp"), "CO1 Comdty": ("Brent crude", "USD/bbl"),
    "XAU Curncy": ("Gold", "USD/oz"),
}
EURUSD = "EURUSD Curncy"            # USD per 1 EUR, for the FX conversion
YIELDS_10Y = ["GDBR10 Index", "USGG10YR Index", "GBTPGR10 Index"]   # report chart
IMPLIED_VOL = ["VIX Index", "V2X Index"]                            # report chart


# --------------------------------------------------------------------------- assets
def estimate_duration(name: str, asset_class: str) -> float:
    """Modified duration from the maturity bucket in the name (approximate, documented).

    "1-3Y" -> about the mid-point maturity x 0.95; "10Y+" -> 16; "All Maturities" -> 7.
    Cash 0.1; equities 0. A regression of returns on yield changes would be a check.
    """
    if asset_class == "Equity":
        return 0.0
    if asset_class == "Cash":
        return 0.1
    n = name.replace("Yr", "Y").replace("Year", "Y")
    if m := re.search(r"(\d+)\s*-\s*(\d+)\s*Y", n):
        return round(0.95 * (int(m.group(1)) + int(m.group(2))) / 2, 1)
    if re.search(r"(\d+)\s*Y?\s*\+", n):
        return 16.0
    if re.search(r"all\s*mat", n, re.I):
        return 7.0
    return 6.0 if asset_class == "Corporate Bond" else 7.0


def _readable(indicator: str) -> str:
    s = re.sub(r"\s*-\s*SYNTHETIC$", "", indicator)
    return re.sub(r"\s*Total Return Index.*$", "", s)


@lru_cache(maxsize=1)
def _assets_meta() -> pd.DataFrame:
    raw = data.read_csv_object("assets_metadata.csv", encoding="utf-8-sig")
    m = pd.DataFrame({
        "ticker": raw["index"],
        "name": raw["indicator"].map(_readable),
        "asset_class": raw["category"].map(ASSET_CLASS),
        "jurisdiction": raw["jurisdiction"],
        "field": raw["field"],
    })
    if m["asset_class"].isna().any():
        raise ValueError(f"Unknown asset categories: {set(raw.loc[m['asset_class'].isna(), 'category'])}")
    m["region"] = np.where(m["jurisdiction"] == "US", "US", "EUR")
    m["currency"] = np.where(m["jurisdiction"] == "US", "USD", "EUR")
    m["duration"] = [estimate_duration(n, c) for n, c in zip(m["name"], m["asset_class"])]
    return m


def assets_meta() -> pd.DataFrame:
    return _assets_meta().copy()


def tickers() -> list[str]:
    return _assets_meta()["ticker"].tolist()


def names() -> dict[str, str]:
    return dict(zip(_assets_meta()["ticker"], _assets_meta()["name"]))


def asset_levels(end_date) -> pd.DataFrame:
    """Wide frame: date x ticker of total-return index levels (local currency)."""
    df = data.read_assets(end_date=end_date)
    wide = df.pivot_table(index="date", columns="index", values="price", aggfunc="last").sort_index()
    return wide.reindex(columns=tickers())


def cash_sink(meta: pd.DataFrame | None = None) -> str:
    """The safest EUR asset (EUR cash, lowest duration): receives rounding residuals."""
    m = _assets_meta() if meta is None else meta
    c = m[(m["asset_class"] == "Cash") & (m["currency"] == "EUR")].sort_values("duration")
    if c.empty:
        raise ValueError("No EUR cash asset in the universe")
    return c["ticker"].iloc[0]


def strategic_benchmark(meta: pd.DataFrame | None = None) -> pd.Series:
    """Neutral portfolio from BENCHMARK_BUCKETS (config): each (asset class, region)
    bucket's weight is split equally across the assets in it. Works for any universe."""
    from src.config import BENCHMARK_BUCKETS

    m = _assets_meta() if meta is None else meta
    w = pd.Series(0.0, index=m["ticker"])
    for (cls, region), bw in BENCHMARK_BUCKETS.items():
        members = m.loc[(m["asset_class"] == cls) & (m["region"] == region), "ticker"]
        if len(members):
            w[members] = bw / len(members)
    return w / w.sum()


# --------------------------------------------------------------------------- financial
def financial_wide(end_date) -> pd.DataFrame:
    """Wide frame: date x official code, forward-filled (holidays, gaps)."""
    df = data.read_financial(end_date=end_date)
    return df.pivot_table(index="date", columns="index", values="price", aggfunc="last").sort_index().ffill()


# --------------------------------------------------------------------------- macro
FREQ = {"Monthly": "M", "Quarterly": "Q", "Annual": "A"}
UNIT = {"PERCENT": "%", "PERCENT_CHANGE": "%", "BALANCE": "balance", "INDEX": "index"}


@lru_cache(maxsize=1)
def _macro_meta() -> pd.DataFrame:
    raw = data.read_csv_object("macro_metadata.csv", encoding="utf-8-sig")
    m = raw.rename(columns={"identifier": "series"}).copy()
    m["frequency"] = m["frequencyDescription"].map(FREQ)
    # Price indices are turned into y/y inflation (the number everyone talks about)
    m["transform"] = np.where((m["category"] == "Prices") & (m["dataType"] == "INDEX"), "yoy", "level")
    m["unit"] = np.where(m["transform"] == "yoy", "% y/y", m["dataType"].map(UNIT).fillna(""))
    ind = m["indicator"].str.replace(r"\s*-\s*SYNTHETIC$", "", regex=True)
    m["label"] = m["jurisdiction"] + " " + np.where(m["transform"] == "yoy", ind + " (y/y)", ind)
    return m.set_index("series")


def macro_meta() -> pd.DataFrame:
    return _macro_meta().copy()


def macro_long(end_date=None) -> pd.DataFrame:
    """Long frame: series, date (as in the file), period_end, value (transformed).

    period_end = last day of the reference period. Most series are dated on the FIRST day
    of the month/quarter, PMIs on the LAST day: period_end handles both. Release lags are
    added in pit.py, counted from period_end.
    """
    df = data.read_macro(end_date=None).rename(columns={"identifier": "series"})  # transforms need history
    meta = _macro_meta()
    freq = df["series"].map(meta["frequency"]).fillna("M")
    df["period_end"] = [d.to_period("Q" if f == "Q" else "Y" if f == "A" else "M").end_time.normalize()
                        for d, f in zip(df["date"], freq)]
    out = []
    for sid, g in df.sort_values("period_end").groupby("series"):
        g = g.copy()
        if meta.loc[sid, "transform"] == "yoy":
            lag = {"M": 12, "Q": 4, "A": 1}[meta.loc[sid, "frequency"]]
            g["value"] = (g["value"] / g["value"].shift(lag) - 1) * 100
        out.append(g.dropna(subset=["value"]))
    df = pd.concat(out, ignore_index=True)
    if end_date is not None:
        df = df[df["period_end"] <= pd.Timestamp(end_date)]
    return df
