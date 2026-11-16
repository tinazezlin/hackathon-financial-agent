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

from src.utils import pit, universe
from src.utils.analytics import asset_statistics
from src.utils.news import read_topic_index

# Exact Reuters topic names (data-detailed.pdf), in priority order for the analyst.
# There is no "Federal Reserve" topic: Fed news comes via the central-bank topics.
NEWS_TOPICS = ["Central Bank Interest Rates / Policy", "European Central Bank", "Central Banks / Central Bank Events",
               "Inflation Data", "Employment / Unemployment Data / Policy", "Purchasing Managers Survey Data",
               "Consumer Sentiment Survey Data", "Debt / Fixed Income Markets", "Fixed Income", "Credit Markets",
               "Equities Markets", "Equities", "Foreign Exchange", "Crude Oil", "Gold"]


# --------------------------------------------------------------------------- plain functions
def asset_statistics_table(as_of: str, lookback_days: int = 252) -> str:
    lv = pit.assets_asof(as_of)
    st = asset_statistics(lv, lookback_days)
    meta = universe.assets_meta().set_index("ticker")
    st.insert(0, "name", meta["name"].reindex(st.index))
    st.insert(1, "class", meta["asset_class"].reindex(st.index).str.replace(" Bond", "", regex=False))
    st.insert(2, "jur", meta["jurisdiction"].reindex(st.index))
    pct = ["ret_1m", "ret_3m", "ret_12m", "ann_vol", "max_dd", "ret_ann"]
    st[pct] = (st[pct] * 100).round(1)
    st["sharpe_like"] = st["sharpe_like"].where(st["ann_vol"] > 0.5).round(2)   # meaningless for cash
    st = st.rename(columns={"ret_1m": "ret_1m_%", "ret_3m": "ret_3m_%", "ret_12m": "ret_12m_trailing_%",
                            "ann_vol": "ann_vol_%", "max_dd": "max_dd_%", "ret_ann": "ret_ann_%"})
    return (f"Asset statistics as of {lv.index[-1]:%Y-%m-%d} (local currency; returns are TRAILING over "
            f"1, 3 and 12 months, not year-to-date; vol/drawdown over the last {lookback_days} days)\n"
            + st.to_string())


def market_indicators_table(as_of: str) -> str:
    fin = pit.financial_asof(as_of)
    cols = [c for c in universe.MARKET_SERIES if c in fin.columns]
    f = fin[cols]
    last = f.iloc[-1]

    def back(n):
        return f.iloc[-(n + 1)] if len(f) > n else f.iloc[0]

    tbl = pd.DataFrame({"indicator": [universe.MARKET_SERIES[c][0] for c in cols],
                        "unit": [universe.MARKET_SERIES[c][1] for c in cols],
                        "last": last, "chg_1m": last - back(21), "chg_3m": last - back(63),
                        "chg_12m": last - back(252)}).round(3)
    tbl = tbl.set_index("indicator")
    return (f"Market indicators as of {f.index[-1]:%Y-%m-%d}. Changes are ABSOLUTE, in the unit of the "
            f"row (a % row changes in percentage points, not percent).\n" + tbl.to_string())


def macro_snapshot_table(as_of: str) -> str:
    m = pit.macro_asof(as_of).sort_values("period_end")
    meta = universe.macro_meta()
    rows = []
    for sid, g in m.groupby("series"):
        last, prev = g.iloc[-1], (g.iloc[-2] if len(g) > 1 else g.iloc[-1])
        # Compare the values as displayed (2 dp), so the LLM never sees "+0.00 up".
        value, prev_value = round(last["value"], 2), round(prev["value"], 2)
        change = round(value - prev_value, 2)
        direction = "up" if change > 0 else "down" if change < 0 else "unchanged"
        freq = meta.loc[sid, "frequency"]
        period = f"{last['period_end']:%Y-%m}" if freq == "M" else f"{pd.Period(last['period_end'], 'Q')}"
        rows.append({"indicator": meta.loc[sid, "label"], "unit": meta.loc[sid, "unit"], "ref_period": period,
                     "value": value, "prev": prev_value, "change": change, "direction": direction})
    df = pd.DataFrame(rows).sort_values("indicator").set_index("indicator")
    df["change"] = df["change"].map(lambda x: f"{x:+.2f}")
    return (f"Macro data published by {as_of} (only releases available at that date; release lags in days "
            f"after the end of the reference period: {pit.RELEASE_LAG_DAYS}). change = value - prev "
            f"(previous period) in the series' unit; for % series that is percentage points.\n" + df.to_string())


def recent_news_text(as_of: str, lookback_days: int = 14, max_items: int = 30) -> str:
    known = set(read_topic_index()["topic"])
    topics = [t for t in NEWS_TOPICS if t in known]                  # read_news rejects unknown names
    n = pit.news_asof(as_of, lookback_days, topics=topics)           # latest version per story
    prio = {t: i for i, t in enumerate(NEWS_TOPICS)}
    n = n.assign(prio=n["topic_name"].map(lambda ts: min((prio[t] for t in ts if t in prio), default=99)),
                 main=n["topic_name"].map(lambda ts: next((t for t in NEWS_TOPICS if t in ts), ts[0])),
                 title=n["title"].str.replace(r"^UPDATE \d+-", "", regex=True))
    # Same headline repeated = one item with a count (repetition is not extra evidence)
    g = (n.sort_values("published_at").groupby("title")
         .agg(published_at=("published_at", "last"), prio=("prio", "min"), main=("main", "first"),
              times=("story_id", "nunique")).reset_index())
    g = g.sort_values(["prio", "published_at"], ascending=[True, False]).head(max_items)
    g = g.sort_values("published_at", ascending=False)
    lines = [f"{r.published_at:%Y-%m-%d %H:%M} UTC | {r.main} | {r.title}"
             + (f"  (x{r.times} similar stories)" if r.times > 1 else "") for r in g.itertuples()]
    return (f"News headlines, last {lookback_days} days up to {as_of} ({len(lines)} items, latest version of "
            f"each story, repeated headlines merged):\n" + "\n".join(lines))


# --------------------------------------------------------------------------- LangChain tools
@tool
def get_asset_statistics(state: Annotated[dict, InjectedState], lookback_days: int = 252) -> str:
    """Trailing (not year-to-date) 1m/3m/12m returns, volatility and drawdown for all 23
    investable indices, as of the reporting date. lookback_days: window for vol/drawdown (e.g. 63, 252, 756)."""
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
