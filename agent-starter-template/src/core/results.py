"""Saving results (MOCK of the hackathon's src/core/results.py - do not edit there).

After each date: calls build_report(result, path) and hashes the PDF + weights.
After all dates: write_weights() writes weights.csv.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.core.session import Session, current_session
from src.utils.report import build_report


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_weights(weights: dict) -> str:
    canon = json.dumps({k: round(float(v), 10) for k, v in sorted(weights.items())}, sort_keys=True)
    return hashlib.sha256(canon.encode()).hexdigest()


def save_date_result(result: dict, session: Session) -> dict:
    date = result["as_of_date"]
    if not result.get("weights"):
        raise ValueError(f"{date}: final state has no 'weights'")
    pdf = session.dir / f"report_{date}.pdf"
    build_report(result, pdf)
    if not pdf.exists():
        raise FileNotFoundError(f"build_report did not write {pdf}")
    hashes = {"pdf": sha256_file(pdf), "weights": sha256_weights(result["weights"])}
    session.results[date] = result
    session.hashes[date] = hashes
    return hashes


def write_weights(session: Session | None = None) -> Path:
    session = session or current_session()
    rows = {d: r["weights"] for d, r in sorted(session.results.items())}
    df = pd.DataFrame.from_dict(rows, orient="index").fillna(0.0)
    df.index.name = "date"
    path = session.dir / "weights.csv"
    df.to_csv(path, float_format="%.6f")
    (session.dir / "hashes.json").write_text(json.dumps(session.hashes, indent=2))
    print(f"Wrote {path}")
    return path
