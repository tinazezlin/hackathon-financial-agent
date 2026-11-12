"""Data access (same function names as the hackathon's src/utils/data.py).

Hackathon: files are read from S3-compatible object storage.
Here: read from ./data (synthetic). Set DATA_DIR to point elsewhere.

All readers return COPIES, so callers may mutate them safely.
For anything that feeds a decision, use src/utils/pit.py (point-in-time views)
instead of these raw readers.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.core.llm import ROOT

DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))


@lru_cache(maxsize=32)
def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name)


def read_csv_object(name: str) -> pd.DataFrame:
    return _read(name).copy()


def read_assets() -> pd.DataFrame:
    """Wide frame: DatetimeIndex x 23 index levels (local currency)."""
    df = read_csv_object("assets.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def read_assets_metadata() -> pd.DataFrame:
    return read_csv_object("assets-metadata.csv")


def read_financial(wide: bool = False) -> pd.DataFrame:
    df = read_csv_object("financial.csv")
    df["date"] = pd.to_datetime(df["date"])
    if wide:
        return df.pivot(index="date", columns="series_id", values="value").sort_index()
    return df


def read_financial_metadata() -> pd.DataFrame:
    return read_csv_object("financial-metadata.csv")


def read_macro() -> pd.DataFrame:
    df = read_csv_object("macro.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


def read_macro_metadata() -> pd.DataFrame:
    return read_csv_object("macro-metadata.csv")


def read_news() -> pd.DataFrame:
    df = read_csv_object("news.csv")
    df["published_at"] = pd.to_datetime(df["published_at"])
    return df
