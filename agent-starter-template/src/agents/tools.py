"""Tools the analyst agent can call.

Key design point: the reporting date is NOT a tool argument. It is injected
from the graph state (InjectedState), so the model cannot ask for data after
the as-of date, even by mistake. All data goes through src/utils/pit.py.

Outputs are compact text tables to save tokens.
"""

from __future__ import annotations

from typing import Annotated

import pandas as pd
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from src.utils import data, pit
from src.utils.analytics import asset_statistics

PRIORITY_TOPICS = ["central_banks", "macro", "rates", "markets", "credit", "fx"]

MARKET_SERIES = ["DE_2Y_YLD", "DE_10Y_YLD", "US_2Y_YLD", "US_10Y_YLD", "IT_10Y_YLD", "FR_10Y_YLD",
                 "ESTR", "SOFR", "EURUSD", "VIX", "VSTOXX", "EUR_IG_OAS", "US_IG_OAS", "ITRAXX_MAIN",
                 "CDS_IT_5Y", "BRENT", "GOLD"]


# --------------------------------------------------------------------------- plain functions
def asset_statistics_table(as_of: str, lookback_days: int = 252) -> str:
    lv = pit.assets_asof(as_of)
    st = asset_statistics(lv, lookback_days)
    meta = data.read_assets_metadata().set_index("ticker")
    st.insert(0, "class", meta["asset_class"].reindex(st.index).str.replace(" Bond", "", regex=False))
    st.insert(1, "jur", meta["jurisdiction"].reindex(st.index))
    pct = ["ret_1m", "ret_3m", "ret_12m", "ann_vol", "max_dd", "ret_ann"]
    st[pct] = (st[pct] * 100).round(1)
    st["sharpe_like"] = st["sharpe_like"].round(2)
    return (f"Asset statistics as of {lv.index[-1]:%Y-%m-%d} (local currency, %; lookback {lookback_days}d)\n"
            + st.to_string())


def market_indicators_table(as_of: str) -> str:
    fin = pit.financial_asof(as_of)
    cols = [c for c in MARKET_SERIES if c in fin.columns]
    f = fin[cols]
    last = f.iloc[-1]

    def back(n):
        return f.iloc[-(n + 1)] if len(f) > n else f.iloc[0]

    tbl = pd.DataFrame({"last": last, "chg_1m": last - back(21), "chg_3m": last - back(63),
                        "chg_12m": last - back(252)}).round(3)
    return (f"Market indicators as of {f.index[-1]:%Y-%m-%d} (yields/rates in %, spreads in bp, "
            f"changes are absolute)\n" + tbl.to_string())


def macro_snapshot_table(as_of: str) -> str:
    m = pit.macro_asof(as_of).sort_values("date")
    meta = data.read_macro_metadata().set_index("series_id")
    rows = []
    for sid, g in m.groupby("series_id"):
        last, prev = g.iloc[-1], (g.iloc[-2] if len(g) > 1 else g.iloc[-1])
        # Compare the values as displayed (2 dp), so the LLM never sees "+0.00 up".
        value, prev_value = round(last["value"], 2), round(prev["value"], 2)
        change = round(value - prev_value, 2)
        direction = "up" if change > 0 else "down" if change < 0 else "unchanged"
        rows.append({"series": sid, "indicator": meta.loc[sid, "indicator"], "unit": meta.loc[sid, "unit"],
                     "ref_period": f"{last['date']:%Y-%m}", "value": value, "prev": prev_value,
                     "change": change, "direction": direction})
    df = pd.DataFrame(rows).set_index("series")
    df["change"] = df["change"].map(lambda x: f"{x:+.2f}")
    return (f"Macro data published by {as_of} (only releases available at that date; "
            f"assumed release lags {pit.MACRO_RELEASE_LAG_DAYS} days, "
            f"policy rates {pit.SERIES_RELEASE_LAG_DAYS}). "
            f"change = value - prev (previous period), in the series' unit.\n" + df.to_string())


def recent_news_text(as_of: str, lookback_days: int = 14, max_items: int = 30) -> str:
    n = pit.news_asof(as_of, lookback_days)
    n = n.assign(prio=n["topic"].map({t: i for i, t in enumerate(PRIORITY_TOPICS)}).fillna(99))
    n = n.sort_values(["prio", "published_at"], ascending=[True, False]).head(max_items)
    n = n.sort_values("published_at", ascending=False)
    lines = [f"{r.published_at:%Y-%m-%d %H:%M} | {r.topic} | {r.headline}" for r in n.itertuples()]
    return f"News headlines, last {lookback_days} days up to {as_of} ({len(lines)} items):\n" + "\n".join(lines)


# --------------------------------------------------------------------------- LangChain tools
@tool
def get_asset_statistics(state: Annotated[dict, InjectedState], lookback_days: int = 252) -> str:
    """Trailing return, volatility and drawdown statistics for all 23 investable indices,
    as of the reporting date. lookback_days: window for vol/drawdown (e.g. 63, 252, 756)."""
    return asset_statistics_table(state["as_of_date"], int(lookback_days))


@tool
def get_market_indicators(state: Annotated[dict, InjectedState]) -> str:
    """Latest levels and 1m/3m/12m changes of key market indicators (yields, policy-related
    rates, EURUSD, VIX/VSTOXX, credit spreads, sovereign CDS, oil, gold) as of the reporting date."""
    return market_indicators_table(state["as_of_date"])


@tool
def get_macro_snapshot(state: Annotated[dict, InjectedState]) -> str:
    """Latest published macro data (inflation, unemployment, PMI, confidence, GDP, policy rates)
    for the euro area, selected countries and the US, as available at the reporting date."""
    return macro_snapshot_table(state["as_of_date"])


@tool
def get_recent_news(state: Annotated[dict, InjectedState], lookback_days: int = 14) -> str:
    """Most relevant news headlines (central banks, macro, rates, markets first) published in the
    last `lookback_days` days up to the reporting date."""
    return recent_news_text(state["as_of_date"], int(lookback_days))


ANALYST_TOOLS = [get_asset_statistics, get_market_indicators, get_macro_snapshot, get_recent_news]
