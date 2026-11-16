"""Point-in-time (PIT) data views: the ONLY way decision code should see data.

Rule: a decision "as of D" may only use information available at the close of D.
  * market data: the official readers with end_date=D (inclusive) PLUS our own <= D
    check (weekends/holidays -> last observation before D)
  * macro data:  dated by reference period, so we add a release lag counted from the
                 END of the reference period (most series are dated on its first day)
  * news:        published_at <= D 23:59:59 UTC, latest version of each story
"""

from __future__ import annotations

import pandas as pd

from src.utils import universe
from src.utils.news import read_news

# Release lag in days after the END of the reference period, by macro category.
# Conservative round numbers: EA HICP flash ~1 day, US CPI ~13 days, PMIs ~1 day
# (flash even before the month end), unemployment ~30 days, GDP flash ~30-45 days.
# Policy rates are known when announced (0) - see PITFALLS #3.
RELEASE_LAG_DAYS = {"Policy rates": 0, "Prices": 14, "Business surveys": 1, "Consumer surveys": 1,
                    "Labour market": 35, "Money": 30, "National accounts": 45}
DEFAULT_LAG_DAYS = {"M": 35, "Q": 60, "A": 90}


def _cutoff(as_of) -> pd.Timestamp:
    return pd.Timestamp(as_of).normalize()


def assets_asof(as_of) -> pd.DataFrame:
    """Wide: date x ticker (official codes), total-return levels up to and including as_of."""
    df = universe.asset_levels(end_date=_cutoff(as_of))
    return df[df.index <= _cutoff(as_of)]


def financial_asof(as_of) -> pd.DataFrame:
    """Wide: date x official code, forward-filled, up to and including as_of."""
    df = universe.financial_wide(end_date=_cutoff(as_of))
    return df[df.index <= _cutoff(as_of)]


def release_lag(series: pd.Series) -> pd.Series:
    meta = universe.macro_meta()
    by_cat = series.map(meta["category"]).map(RELEASE_LAG_DAYS)
    by_freq = series.map(meta["frequency"]).map(DEFAULT_LAG_DAYS)
    return by_cat.fillna(by_freq).fillna(60)


def macro_asof(as_of) -> pd.DataFrame:
    """Long macro frame (series, date, period_end, value, available_from) containing only
    values that would have been published by as_of."""
    df = universe.macro_long(end_date=_cutoff(as_of))
    df["available_from"] = df["period_end"] + pd.to_timedelta(release_lag(df["series"]), unit="D")
    return df[df["available_from"] <= _cutoff(as_of)]


def returns_base_ccy_asof(as_of, lookback_days: int) -> pd.DataFrame:
    """Daily asset returns in the base currency (EUR), last `lookback_days` up to as_of."""
    from src.config import FX_HEDGED
    from src.utils.analytics import daily_returns, to_base_currency

    lv = assets_asof(as_of)
    r = daily_returns(lv).iloc[-lookback_days:]
    fx = financial_asof(as_of)[universe.EURUSD]
    return to_base_currency(r, universe.assets_meta(), fx, hedged=FX_HEDGED)


def news_asof(as_of, lookback_days: int = 31, topics: list[str] | None = None,
              title_contains: str | None = None, latest_version: bool = True) -> pd.DataFrame:
    """News published in the lookback window up to as_of 23:59:59 UTC.

    latest_version=True keeps one row per story: its last version published by as_of
    (an update published after as_of is invisible, as it should be).
    """
    end = _cutoff(as_of)
    start = end - pd.Timedelta(days=lookback_days)
    df = read_news(start.date(), end.date(), topics=topics, title_contains=title_contains).to_pandas()
    df = df[df["published_at"] <= pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23, minutes=59, seconds=59)]
    if latest_version:
        df = df.sort_values(["story_id", "published_at"]).drop_duplicates("story_id", keep="last")
    return df.sort_values("published_at").reset_index(drop=True)
