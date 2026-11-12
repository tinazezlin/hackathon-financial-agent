"""Point-in-time (PIT) data views: the ONLY way decision code should see data.

Rule: a decision "as of D" may only use information available at the close of D.
  * market data: observations dated <= D (weekends/holidays -> last obs before D)
  * macro data:  dated by reference period, so we add a conservative release lag
  * news:        published_at <= D 23:59:59
"""

from __future__ import annotations

import pandas as pd

from src.utils import data

# Conservative publication lags (reference period end -> data available).
# Real releases vary (EA HICP flash ~0 days, GDP ~30-45 days, US CPI ~2 weeks).
MACRO_RELEASE_LAG_DAYS = {"M": 35, "Q": 60}

# Per-series overrides (days). Policy rates are public the moment the central bank
# announces them, so there is no publication lag: the value dated at a month end is
# usable from that date on.
SERIES_RELEASE_LAG_DAYS = {"ECB_DEPO_RATE": 0, "FED_FUNDS_UPPER": 0}


def _cutoff(as_of) -> pd.Timestamp:
    return pd.Timestamp(as_of).normalize()


def assets_asof(as_of) -> pd.DataFrame:
    df = data.read_assets()
    return df[df.index <= _cutoff(as_of)]


def financial_asof(as_of) -> pd.DataFrame:
    """Wide frame of financial series up to and including as_of (forward-filled)."""
    df = data.read_financial(wide=True)
    return df[df.index <= _cutoff(as_of)].ffill()


def macro_asof(as_of) -> pd.DataFrame:
    """Long macro frame containing only values that would have been published by as_of."""
    df = data.read_macro()
    meta = data.read_macro_metadata().set_index("series_id")["frequency"]
    lag = df["series_id"].map(meta).map(MACRO_RELEASE_LAG_DAYS).fillna(60)
    override = df["series_id"].map(SERIES_RELEASE_LAG_DAYS)
    lag = override.fillna(lag)
    available_from = df["date"] + pd.to_timedelta(lag, unit="D")
    return df[available_from <= _cutoff(as_of)]


def returns_base_ccy_asof(as_of, lookback_days: int) -> pd.DataFrame:
    """Daily asset returns in the base currency (EUR), last `lookback_days` up to as_of."""
    from src.config import FX_HEDGED
    from src.utils.analytics import daily_returns, to_base_currency

    lv = assets_asof(as_of)
    r = daily_returns(lv).iloc[-lookback_days:]
    fx = financial_asof(as_of)["EURUSD"]
    return to_base_currency(r, data.read_assets_metadata(), fx, hedged=FX_HEDGED)


def news_asof(as_of, lookback_days: int = 31) -> pd.DataFrame:
    df = data.read_news()
    end = _cutoff(as_of) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    start = _cutoff(as_of) - pd.Timedelta(days=lookback_days)
    return df[(df["published_at"] <= end) & (df["published_at"] > start)]
