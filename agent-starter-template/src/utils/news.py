"""News readers, mimicking the official src/utils/news.py (data-detailed.pdf).

    find_topics("central bank")                 # topics whose name contains the words
    read_news("2025-10-01", "2025-10-31", topics=["Inflation Data"])
    read_news(start, end, title_contains="ECB", columns=["published_at", "title", "body"])

The official reader works on parquet and returns an Arrow table (call .to_pandas()).
pyarrow is not in our venv, so here a small wrapper with the same .to_pandas() and
.num_rows stands in for it. Columns: published_at (UTC), date, version_created,
story_id, version, version_role, title, topic_name (list), has_body, body.

Each story can appear in several versions (an update later the same day or the next
day). For a point-in-time view keep, per story, the last version published by T.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from src.utils.data import DATA_DIR, read_csv_object

DEFAULT_COLUMNS = ["published_at", "date", "version_created", "story_id", "version", "version_role",
                   "title", "topic_name", "has_body"]


class NewsTable:
    """Stand-in for a pyarrow.Table: only what we use."""

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def to_pandas(self) -> pd.DataFrame:
        return self._df.copy()

    @property
    def num_rows(self) -> int:
        return len(self._df)


@lru_cache(maxsize=1)
def _all_news() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "news.csv")
    for c in ("published_at", "version_created"):
        df[c] = pd.to_datetime(df[c], utc=True)
    df["topic_name"] = df["topic_name"].str.split("|")
    return df


def read_topic_index() -> pd.DataFrame:
    """All topics with their number of stories (news_topics.csv)."""
    return read_csv_object("news_topics.csv")


def find_topics(*words: str) -> pd.DataFrame:
    """Topics whose name contains ANY of the words (case-insensitive); all if no words."""
    t = read_topic_index()
    if not words:
        return t
    mask = pd.Series(False, index=t.index)
    for w in words:
        mask |= t["topic"].str.contains(w, case=False, regex=False)
    return t[mask]


def read_news(start, end, topics: list[str] | None = None, title_contains: str | None = None,
              columns: list[str] | None = None) -> NewsTable:
    """News versions published between start and end (UTC calendar days, both inclusive).

    topics: exact names, ANY of them must match (unknown names raise ValueError).
    title_contains: case-insensitive; combined with topics, BOTH must match.
    body is only returned when asked for in `columns`.
    """
    df = _all_news()
    lo = pd.Timestamp(start, tz="UTC")
    hi = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)          # end day included
    df = df[(df["published_at"] >= lo) & (df["published_at"] < hi)]
    if topics:
        known = set(read_topic_index()["topic"])
        unknown = [t for t in topics if t not in known]
        if unknown:
            raise ValueError(f"Unknown topic(s) {unknown}. Look names up with find_topics().")
        wanted = set(topics)
        df = df[df["topic_name"].map(lambda ts: bool(wanted.intersection(ts)))]
    if title_contains:
        df = df[df["title"].str.contains(title_contains, case=False, regex=False)]
    return NewsTable(df[columns or DEFAULT_COLUMNS].reset_index(drop=True))
