"""Unit tests for the quant core, the data adapter and point-in-time discipline."""

import numpy as np
import pandas as pd
import pytest

from src.config import LIMITS, REPORTING_DATES
from src.utils import pit, universe
from src.utils.analytics import (check_constraints, portfolio_metrics, project_to_constraints, round_weights,
                                 to_base_currency)

META = universe.assets_meta()
TICKERS = META["ticker"].tolist()
BENCH = universe.strategic_benchmark()


def by_name(name: str) -> str:
    return META.loc[META["name"] == name, "ticker"].item()


SP500, DAX, EUR_CASH = by_name("S&P 500"), by_name("DAX"), universe.cash_sink()


# ---------------------------------------------------------------- adapter
def test_universe_has_23_assets_with_known_classes():
    assert len(META) == 23 and META["ticker"].is_unique
    assert set(META["asset_class"]) == {"Government Bond", "Corporate Bond", "Equity", "Cash"}
    assert set(META["jurisdiction"]) <= {"EUR", "US", "DE", "IT", "FR"}
    assert (META.loc[META["jurisdiction"] == "US", "currency"] == "USD").all()


@pytest.mark.parametrize("name,cls,expected", [("DE Government Bond 1-3Y", "Government Bond", 1.9),
                                               ("US Treasury 10Y+", "Government Bond", 16.0),
                                               ("Euro Area Government Bond All Maturities", "Government Bond", 7.0),
                                               ("DAX", "Equity", 0.0)])
def test_duration_estimate(name, cls, expected):
    assert universe.estimate_duration(name, cls) == expected


def test_hicp_index_becomes_yoy():
    m = universe.macro_long()
    hicp = m[m["series"] == "P023ZH"].set_index("period_end")["value"]
    assert -2 < hicp.iloc[-1] < 12                      # a y/y rate, not an index level of ~130


def test_macro_period_end_handles_first_and_last_day_dating():
    m = universe.macro_long()
    first = m[(m["series"] == "P023ZH") & (m["date"] == "2025-03-01")]["period_end"].item()
    last = m[(m["series"] == "PMIMEZ") & (m["date"] == "2025-03-31")]["period_end"].item()
    assert first == last == pd.Timestamp("2025-03-31")


# ---------------------------------------------------------------- portfolio construction
def test_benchmark_is_compliant():
    assert BENCH.sum() == pytest.approx(1.0)
    assert check_constraints(BENCH, META) == []


@pytest.mark.parametrize("seed", range(20))
def test_projection_always_compliant(seed):
    rng = np.random.default_rng(seed)
    w0 = pd.Series(rng.dirichlet(np.ones(len(TICKERS)) * 0.3), index=TICKERS)
    w0[SP500] += 0.5  # force violations
    w = round_weights(project_to_constraints(w0 / w0.sum(), META))
    assert check_constraints(w, META) == []
    assert w.sum() == pytest.approx(1.0, abs=1e-9)


def test_projection_keeps_compliant_portfolio():
    assert (project_to_constraints(BENCH, META) - BENCH).abs().max() < 1e-4


def test_rounding_cannot_breach_cap():
    # Three US assets at exactly 20% in total; each one rounds UP at 10 decimals,
    # so naive rounding gives 20.00000001% > cap.
    us = META.loc[META["jurisdiction"] == "US", "ticker"].tolist()[:3]
    rest = META.loc[META["jurisdiction"] != "US", "ticker"].tolist()
    w = pd.Series(0.0, index=TICKERS)
    w[us] = 0.2 / 3
    w[rest] = 0.8 / len(rest)
    assert w.round(10)[us].sum() > 0.2                          # the bug we guard against
    assert check_constraints(round_weights(w), META, tol=0) == []


def test_round_weights_no_float_artefacts():
    w = pd.Series(0.0, index=TICKERS)
    w[TICKERS[0]], w[EUR_CASH] = 0.13, 0.87
    assert round_weights(w)[TICKERS[0]] == 0.13                # not 0.1299999999


def test_usd_returns_converted_to_eur():
    idx = pd.bdate_range("2024-01-01", periods=3)
    r = pd.DataFrame({SP500: [0.0, 0.0, 0.0], DAX: [0.0, 0.0, 0.0]}, index=idx)
    fx = pd.Series([1.10, 1.21, 1.21], index=idx)  # EUR +10% vs USD
    out = to_base_currency(r, META, fx)
    assert out.loc[idx[1], SP500] == pytest.approx(1 / 1.1 - 1)
    assert out[DAX].abs().sum() == 0


def test_metrics_sane():
    rets = pit.returns_base_ccy_asof("2025-10-31", 756)
    b = BENCH.reindex(rets.columns).fillna(0)
    m = portfolio_metrics(b, rets, META, benchmark=b)
    assert 0 < m["ann_vol"] < 0.2
    assert m["tracking_error"] == pytest.approx(0, abs=1e-12)
    assert m["es95_1d"] >= m["var95_1d"] > 0
    assert sum(m["risk_contrib_by_class"].values()) == pytest.approx(1, abs=1e-3)  # parts rounded to 4 dp


# ---------------------------------------------------------------- look-ahead / point-in-time
@pytest.mark.parametrize("as_of", REPORTING_DATES + ["2026-01-31"])
def test_pit_views_have_no_future_data(as_of):
    end = pd.Timestamp(as_of)
    assert pit.assets_asof(as_of).index.max() <= end
    assert pit.financial_asof(as_of).index.max() <= end
    assert pit.news_asof(as_of)["published_at"].max() <= pd.Timestamp(as_of, tz="UTC") + pd.Timedelta(days=1)
    m = pit.macro_asof(as_of)
    assert (m["available_from"] <= end).all()
    assert (m["period_end"] <= end).all()                       # no reference period after T


def test_macro_lag_counts_from_period_end():
    # March 2026 HICP is dated 01/03/2026; on 31 March it is NOT published yet (14-day lag).
    m = pit.macro_asof("2026-03-31")
    last_hicp = m[m["series"] == "P023ZH"]["period_end"].max()
    assert last_hicp == pd.Timestamp("2026-02-28")


def test_policy_rates_not_lagged():
    m = pit.macro_asof("2025-10-31")
    fed = m[m["series"] == "FDTRUP"].sort_values("period_end").iloc[-1]
    assert fed["period_end"] == pd.Timestamp("2025-10-31") and fed["value"] == 4.0  # cut on 2025-10-30


def test_news_latest_version_by_T_only():
    n = pit.news_asof("2025-10-31", 14, title_contains="Fed cuts")
    assert n["story_id"].is_unique
    assert (n["published_at"] <= pd.Timestamp("2025-10-31 23:59:59", tz="UTC")).all()


def test_unknown_news_topic_raises():
    from src.utils.news import read_news
    with pytest.raises(ValueError):
        read_news("2025-10-01", "2025-10-31", topics=["Federal Reserve"])


def test_reporting_dates_match_deliverables():
    assert REPORTING_DATES == ["2025-10-31", "2025-11-28", "2025-12-29", "2026-01-30", "2026-02-27", "2026-03-31"]


def test_weekend_date_uses_friday():
    assert pit.assets_asof("2026-01-31").index.max() == pd.Timestamp("2026-01-30")


def test_limits_cover_briefing_constraints():
    mandatory = {(lim.field, lim.value, lim.bound) for lim in LIMITS if lim.mandatory}
    assert mandatory == {("asset_class", "Equity", 0.40), ("jurisdiction", "US", 0.20),
                         ("asset_class", "Corporate Bond", 0.20)}
