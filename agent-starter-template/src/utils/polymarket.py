"""Policy expectations from Polymarket prediction markets, point-in-time.

    policy_expectations("2025-10-31", "ECB")  ->  next ECB meeting after T, probability of
                                                  each decision as priced at the end of T

Source (POLYMARKET_MODE env):
    auto  (default) the official ecb_polymarket_api if installed (hackathon), else the mock
    real  always the library (fails if it is missing)
    mock  synthetic odds in the same shape, consistent with our synthetic policy path
    off   no data (the tool says so)

Look-ahead rules from polymarket_detailed.pdf, applied here:
  * the cutoff is T 23:59:59 UTC (a plain date would mean midnight = the START of T)
  * events are selected by schedule (opened by T, scheduled to close after T), NEVER by
    `closed`, `volume` or `liquidity`: those describe TODAY, not T
  * order books / midpoints / spreads are live values: not used
  * YES prices of exclusive answers only roughly sum to 1: we report the raw sum and
    normalise only when the event is flagged neg_risk (mutually exclusive answers)

UNVERIFIED (no access to the library before the event): the column names returned by
get_prices_by_event and some attribute names. _real_expectations() is written from the
PDF; check it on day one with the notebook cell in multi_agents_market.ipynb.
"""

from __future__ import annotations

import os
import zlib
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
import pandas as pd

TAGS = {"ECB": "european-central-bank", "Fed": "fed-rates"}
TITLE_WORDS = {"ECB": "ECB", "Fed": "Fed"}


def mode() -> str:
    m = os.getenv("POLYMARKET_MODE", "auto").lower()
    if m != "auto":
        return m
    try:
        import ecb_polymarket_api  # noqa: F401
        return "real"
    except ImportError:
        return "mock"


def cutoff(as_of) -> datetime:
    """End of the reporting day in UTC: the only correct `end=` for the price helpers."""
    d = pd.Timestamp(as_of)
    return datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=timezone.utc)


def _summarise(daily: pd.DataFrame, meeting: pd.Timestamp, title: str, exclusive: bool, source: str) -> dict:
    """daily: UTC days x answers (YES prices), already cut at T."""
    daily = daily.dropna(how="all").ffill()
    last = daily.iloc[-1]
    week = daily.iloc[-8] if len(daily) > 7 else daily.iloc[0]
    raw_sum = float(last.sum())
    prob = last / raw_sum if exclusive and raw_sum > 0 else last
    prob_week = week / week.sum() if exclusive and week.sum() > 0 else week
    table = pd.DataFrame({"prob": prob, "chg_1w": prob - prob_week}).sort_values("prob", ascending=False)
    return {"event": title, "meeting": meeting.normalize(), "as_of_price_date": daily.index[-1],
            "raw_yes_sum": raw_sum, "normalised": exclusive, "table": table, "source": source}


# --------------------------------------------------------------------------- real library
def _real_expectations(as_of, bank: str) -> dict | None:
    from ecb_polymarket_api import discover_events, get_prices_by_event

    T = cutoff(as_of)
    events = discover_events(tags=[TAGS[bank]], start_date_max=T.date(), end_date_min=T.date())
    events = [e for e in events if TITLE_WORDS[bank].lower() in e.title.lower()
              and "interest rate" in e.title.lower()]
    if not events:
        return None
    ev = min(events, key=lambda e: pd.Timestamp(e.schedule.end_date))     # the NEXT meeting
    try:
        df = get_prices_by_event(ev, end=T)
    except Exception:
        # a market that closed before T raises with an end after its last price:
        # fall back to each market's own history up to T
        from ecb_polymarket_api import get_prices_by_market
        parts = []
        for m in ev.markets:
            try:
                parts.append(get_prices_by_market(m, end=T).iloc[:, 0].rename(m.group_item_title))
            except Exception:
                continue
        if not parts:
            return None
        df = pd.concat(parts, axis=1)
    df = df[df.index <= pd.Timestamp(T)]                                   # belt and braces
    labels = {getattr(m, "id", None): m.group_item_title for m in ev.markets}
    df = df.rename(columns=lambda c: labels.get(c, c))
    daily = df.resample("1D").last().ffill()
    return _summarise(daily, pd.Timestamp(ev.schedule.end_date).tz_localize(None), ev.title,
                      bool(getattr(ev.trading, "neg_risk", True)), "Polymarket (ecb_polymarket_api)")


# --------------------------------------------------------------------------- mock
# Synthetic meeting calendar consistent with scripts/generate_synthetic_data.py:
# meeting date -> decision. The 2026 decisions are fictional, like the rest of the data.
MOCK_MEETINGS = {
    "ECB": {"2025-09-11": "No change", "2025-10-30": "No change", "2025-12-18": "No change",
            "2026-02-05": "No change", "2026-03-18": "25 bps decrease", "2026-04-30": "No change"},
    "Fed": {"2025-09-18": "25 bps decrease", "2025-10-30": "25 bps decrease", "2025-12-11": "25 bps decrease",
            "2026-01-29": "No change", "2026-03-19": "25 bps decrease", "2026-04-30": "No change"},
}
ANSWERS = ["No change", "25 bps decrease", "50+ bps decrease", "25+ bps increase"]


@lru_cache(maxsize=32)
def _mock_history(bank: str, meeting: str) -> pd.DataFrame:
    """Daily YES prices from ~2 months before the meeting until the meeting. A real market
    partly anticipates the outcome, so the eventual answer drifts up, with noise; the
    quotes deliberately do not sum exactly to 1 (spreads, stale quotes)."""
    m = pd.Timestamp(meeting)
    days = pd.date_range(m - pd.Timedelta(days=60), m, freq="D", tz="UTC")
    # zlib.crc32, not hash(): Python randomises string hashes per process (PYTHONHASHSEED)
    rng = np.random.default_rng(zlib.crc32(f"{bank}|{meeting}".encode()))
    outcome = MOCK_MEETINGS[bank][meeting]
    progress = np.linspace(0, 1, len(days))
    p_true = np.clip(0.45 + 0.5 * progress**1.5 + np.cumsum(rng.normal(0, 0.02, len(days))) * 0.5, 0.05, 0.98)
    others = [a for a in ANSWERS if a != outcome]
    split = np.array([0.7, 0.2, 0.1])
    out = {outcome: p_true}
    for a, s in zip(others, split):
        out[a] = (1 - p_true) * s
    df = pd.DataFrame(out, index=days)[ANSWERS]
    return (df * rng.uniform(0.97, 1.02, size=df.shape)).clip(0.001, 0.999).round(4)


def _mock_expectations(as_of, bank: str) -> dict | None:
    T = pd.Timestamp(cutoff(as_of))
    upcoming = [d for d in MOCK_MEETINGS[bank] if pd.Timestamp(d, tz="UTC") > T]
    if not upcoming:
        return None
    meeting = upcoming[0]
    hist = _mock_history(bank, meeting)
    hist = hist[hist.index <= T]                                          # point-in-time cut
    if hist.empty:
        return None
    title = f"{bank} Interest Rates: {pd.Timestamp(meeting):%B %Y} (SYNTHETIC)"
    return _summarise(hist, pd.Timestamp(meeting), title, True, "mock (synthetic odds)")


# --------------------------------------------------------------------------- public
@lru_cache(maxsize=64)
def policy_expectations(as_of: str, bank: str) -> dict | None:
    """Market-implied probabilities for the next `bank` ("ECB" or "Fed") meeting after as_of."""
    m = mode()
    if m == "off":
        return None
    if m == "real":
        return _real_expectations(as_of, bank)
    return _mock_expectations(as_of, bank)
