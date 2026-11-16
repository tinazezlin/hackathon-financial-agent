"""Raw data readers, mimicking the official src/utils/data.py (data-detailed.pdf).

Hackathon: the files are in an S3 bucket. Here: ./data (synthetic), or DATA_DIR.

    prices = read_assets(end_date="2025-10-31")     # date, index, price, price_description
    market = read_financial(end_date="2025-10-31")
    macro  = read_macro(end_date="2025-10-31")       # date, identifier, value
    meta   = read_csv_object("assets_metadata.csv", encoding="utf-8-sig")

LOOK-AHEAD: without end_date these return EVERYTHING, including the future. Decision
code should not call them directly: use src/utils/pit.py, which always passes end_date
and adds its own checks. The dates in the files are day-first (02/01/2018 = 2 January).

What the official readers return exactly (strings or parsed dates?) is not documented;
here `date` is parsed to datetime. Check on the day and adapt src/utils/universe.py.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.core.llm import ROOT

DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))


@lru_cache(maxsize=32)
def _read(name: str, encoding: str = "utf-8") -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name, encoding=encoding)


def read_csv_object(name: str, encoding: str = "utf-8") -> pd.DataFrame:
    """Read any CSV object from the bucket (metadata files: encoding="utf-8-sig")."""
    return _read(name, encoding).copy()


@lru_cache(maxsize=8)
def _read_dated(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / name)
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
    return df


def _upto(name: str, end_date) -> pd.DataFrame:
    df = _read_dated(name)
    if end_date is not None:
        df = df[df["date"] <= pd.Timestamp(end_date)]      # inclusive, like the official readers
    return df.copy()


def read_assets(end_date=None) -> pd.DataFrame:
    """Long frame: date, index, price, price_description (23 total-return indices)."""
    return _upto("assets.csv", end_date)


def read_financial(end_date=None) -> pd.DataFrame:
    """Long frame: date, index, price, price_description."""
    return _upto("financial.csv", end_date)


def read_macro(end_date=None) -> pd.DataFrame:
    """Long frame: date (reference period, NOT release date), identifier, value.

    end_date filters on the reference period: a March value dated 01/03 is returned for
    end_date=15 March although it was published in April. pit.py handles release lags.
    """
    return _upto("macro.csv", end_date)
