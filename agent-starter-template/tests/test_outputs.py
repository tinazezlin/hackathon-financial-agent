"""Validate the outputs of the LATEST session against deliverables.pdf (like the
hackathon's checks: they look at files, they do not run the agents).

    .venv/bin/python -m pytest tests/test_outputs.py
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.config import REPORTING_DATES
from src.core.results import sha256_file
from src.utils import universe
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


def test_folder_has_exactly_seven_files(session_dir):
    expected = {f"{d}.pdf" for d in REPORTING_DATES} | {"weights.csv"}
    assert {p.name for p in session_dir.iterdir()} == expected


def test_all_dates_present(weights):
    assert list(weights.index) == sorted(REPORTING_DATES)


def test_all_assets_present(weights):
    assert set(weights.columns) == set(universe.tickers())


def test_rows_sum_to_one_and_ten_decimals(session_dir, weights):
    assert (weights.sum(axis=1) - 1).abs().max() < 1e-9
    first_row = (session_dir / "weights.csv").read_text().splitlines()[1].split(",")[1:]
    assert all(len(v.split(".")[1]) == 10 for v in first_row)


def test_constraints(weights):
    meta = universe.assets_meta()
    for d, row in weights.iterrows():
        assert check_constraints(row, meta) == [], d


def test_pdfs_match_fingerprints(session_dir):
    hf = session_dir.parent / ".hashes" / f"{session_dir.name}.json"
    if not hf.exists():
        pytest.skip("no local fingerprints for this session")
    hashes = json.loads(hf.read_text())
    for d in REPORTING_DATES:
        pdf = session_dir / f"{d}.pdf"
        assert sha256_file(pdf) == hashes[d]["pdf"], f"{pdf} was modified after the run"
