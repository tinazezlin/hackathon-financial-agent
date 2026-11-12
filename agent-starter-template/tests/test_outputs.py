"""Validate the outputs of the LATEST session (like the hackathon's validation tests:
they check files, they do not run the agents).

    .venv/bin/python -m pytest tests/test_outputs.py
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.config import REPORTING_DATES
from src.core.results import sha256_file
from src.utils import data
from src.utils.analytics import check_constraints

OUT = Path(__file__).resolve().parents[1] / "outputs"
sessions = sorted(p for p in OUT.glob("*") if (p / "weights.csv").exists()) if OUT.exists() else []
pytestmark = pytest.mark.skipif(not sessions, reason="no completed session in outputs/ yet")


@pytest.fixture(scope="module")
def session_dir():
    return sessions[-1]


@pytest.fixture(scope="module")
def weights(session_dir):
    return pd.read_csv(session_dir / "weights.csv", index_col=0)


def test_all_dates_present(weights):
    assert sorted(weights.index) == sorted(REPORTING_DATES)


def test_all_tickers_present(weights):
    assert set(weights.columns) == set(data.read_assets_metadata()["ticker"])


def test_constraints(weights):
    meta = data.read_assets_metadata()
    for d, row in weights.iterrows():
        assert check_constraints(row, meta) == [], d


def test_pdfs_exist_and_match_hashes(session_dir):
    hashes = json.loads((session_dir / "hashes.json").read_text())
    for d in REPORTING_DATES:
        pdf = session_dir / f"report_{d}.pdf"
        assert pdf.exists(), pdf
        assert sha256_file(pdf) == hashes[d]["pdf"], f"{pdf} was modified after the run"
