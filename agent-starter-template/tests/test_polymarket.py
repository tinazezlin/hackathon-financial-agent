"""Point-in-time rules for the prediction-market signal (mock mode)."""

import subprocess
import sys

import pandas as pd
import pytest

from src.utils import polymarket as pm


@pytest.fixture(autouse=True)
def mock_mode(monkeypatch):
    monkeypatch.setenv("POLYMARKET_MODE", "mock")
    pm.policy_expectations.cache_clear()


def test_cutoff_is_end_of_day_utc():
    T = pm.cutoff("2026-03-31")
    assert (T.hour, T.minute, T.second, str(T.tzinfo)) == (23, 59, 59, "UTC")


@pytest.mark.parametrize("as_of", ["2025-10-31", "2025-12-29", "2026-03-16", "2026-03-31"])
@pytest.mark.parametrize("bank", ["ECB", "Fed"])
def test_next_meeting_and_no_prices_after_T(as_of, bank):
    e = pm.policy_expectations(as_of, bank)
    assert e["meeting"] > pd.Timestamp(as_of)                      # the NEXT meeting
    assert e["as_of_price_date"] <= pd.Timestamp(pm.cutoff(as_of))  # nothing after T
    assert e["table"]["prob"].sum() == pytest.approx(1.0)           # normalised (exclusive answers)


def test_odds_move_towards_the_decision():
    early = pm.policy_expectations("2026-02-27", "ECB")["table"]["prob"]
    late = pm.policy_expectations("2026-03-16", "ECB")["table"]["prob"]
    assert late["25 bps decrease"] > early["25 bps decrease"]


def test_mock_is_deterministic_across_processes():
    code = ("from src.utils.polymarket import policy_expectations as p;"
            "print(p('2025-10-31','ECB')['table'].round(6).to_json())")
    runs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
            for _ in range(2)}
    assert len(runs) == 1


def test_off_mode_gives_no_data(monkeypatch):
    monkeypatch.setenv("POLYMARKET_MODE", "off")
    from src.agents.tools import policy_expectations_text
    assert "no prediction-market data" in policy_expectations_text("2025-10-31")
