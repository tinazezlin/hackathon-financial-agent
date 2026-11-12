"""Unit tests for the quant core and point-in-time discipline."""

import numpy as np
import pandas as pd
import pytest

from src.config import LIMITS, STRATEGIC_BENCHMARK
from src.utils import data, pit
from src.utils.analytics import (check_constraints, portfolio_metrics, project_to_constraints, round_weights,
                                 to_base_currency)

META = data.read_assets_metadata()
TICKERS = META["ticker"].tolist()


def test_benchmark_is_compliant():
    b = pd.Series(STRATEGIC_BENCHMARK).reindex(TICKERS).fillna(0)
    assert set(STRATEGIC_BENCHMARK) <= set(TICKERS)
    assert check_constraints(b, META) == []


@pytest.mark.parametrize("seed", range(20))
def test_projection_always_compliant(seed):
    rng = np.random.default_rng(seed)
    w0 = pd.Series(rng.dirichlet(np.ones(len(TICKERS)) * 0.3), index=TICKERS)
    w0["SP500"] += 0.5  # force violations
    w = round_weights(project_to_constraints(w0 / w0.sum(), META))
    assert check_constraints(w, META) == []
    assert w.sum() == pytest.approx(1.0, abs=1e-9)


def test_projection_keeps_compliant_portfolio():
    b = pd.Series(STRATEGIC_BENCHMARK).reindex(TICKERS).fillna(0)
    assert (project_to_constraints(b, META) - b).abs().max() < 1e-4


def test_rounding_cannot_breach_cap():
    # Three US assets at exactly 20% in total; each one rounds UP at 6 decimals,
    # so naive rounding gives 20.0001% > cap.
    us = ["SP500", "US_GOV_1_3Y", "US_CASH"]
    w = pd.Series(0.0, index=TICKERS)
    w[us] = 0.2 / 3
    non_us = set(META.loc[META["jurisdiction"] != "US", "ticker"])
    rest = [t for t in TICKERS if t in non_us]
    w[rest] = 0.8 / len(rest)
    assert w.round(6)[us].sum() > 0.2                          # the bug we guard against
    assert check_constraints(round_weights(w), META, tol=0) == []


def test_usd_returns_converted_to_eur():
    idx = pd.bdate_range("2024-01-01", periods=3)
    r = pd.DataFrame({"SP500": [0.0, 0.0, 0.0], "DAX": [0.0, 0.0, 0.0]}, index=idx)
    fx = pd.Series([1.10, 1.21, 1.21], index=idx)  # EUR +10% vs USD
    out = to_base_currency(r, META, fx)
    assert out.loc[idx[1], "SP500"] == pytest.approx(1 / 1.1 - 1)
    assert out["DAX"].abs().sum() == 0


def test_metrics_sane():
    rets = pit.returns_base_ccy_asof("2025-10-31", 756)
    b = pd.Series(STRATEGIC_BENCHMARK).reindex(rets.columns).fillna(0)
    m = portfolio_metrics(b, rets, META, benchmark=b)
    assert 0 < m["ann_vol"] < 0.2
    assert m["tracking_error"] == pytest.approx(0, abs=1e-12)
    assert m["es95_1d"] >= m["var95_1d"] > 0
    assert sum(m["risk_contrib_by_class"].values()) == pytest.approx(1, abs=1e-6)


# ---------------------------------------------------------------- look-ahead / point-in-time
@pytest.mark.parametrize("as_of", ["2025-10-31", "2026-01-31", "2026-02-28"])
def test_pit_views_have_no_future_data(as_of):
    end = pd.Timestamp(as_of)
    assert pit.assets_asof(as_of).index.max() <= end
    assert pit.financial_asof(as_of).index.max() <= end
    assert pit.news_asof(as_of)["published_at"].max() < end + pd.Timedelta(days=1)
    m = pit.macro_asof(as_of)
    assert m["date"].max() <= end
    lagged = m[~m["series_id"].isin(pit.SERIES_RELEASE_LAG_DAYS)]
    assert lagged["date"].max() <= end - pd.Timedelta(days=min(pit.MACRO_RELEASE_LAG_DAYS.values()))


def test_policy_rates_not_lagged():
    m = pit.macro_asof("2025-10-31")
    fed = m[m["series_id"] == "FED_FUNDS_UPPER"].iloc[-1]
    assert fed["date"] == pd.Timestamp("2025-10-31") and fed["value"] == 4.0  # cut on 2025-10-30


def test_weekend_reporting_date_uses_friday():
    assert pit.assets_asof("2026-01-31").index.max() == pd.Timestamp("2026-01-30")


def test_limits_cover_briefing_constraints():
    mandatory = {(lim.field, lim.value, lim.bound) for lim in LIMITS if lim.mandatory}
    assert mandatory == {("asset_class", "Equity", 0.40), ("jurisdiction", "US", 0.20),
                         ("asset_class", "Corporate Bond", 0.20)}
