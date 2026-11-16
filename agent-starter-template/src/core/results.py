"""Saving results (MOCK of the hackathon's src/core/results.py - do not edit there).

Mirrors deliverables.pdf:
  * after each date: check approved_weights holds all 23 assets (fail the date
    otherwise), write <date>.pdf with build_report(result, path), and fingerprint
    the PDF and the weights (the fingerprints go to the Langfuse trace);
  * write_weights(results, session): weights.csv, 10 decimals, one row per date,
    and print the Langfuse session URL.

The session folder holds exactly 7 files (6 PDFs + weights.csv). For our offline
tests we also keep the fingerprints OUTSIDE the folder, in outputs/.hashes/<id>.json.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.core.session import Session, current_session
from src.utils import data
from src.utils.report import build_report


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_weights(weights: dict) -> str:
    canon = json.dumps({k: round(float(v), 10) for k, v in sorted(weights.items())}, sort_keys=True)
    return hashlib.sha256(canon.encode()).hexdigest()


def _check_assets(date: str, weights: dict) -> None:
    assets = set(data.read_assets_metadata()["ticker"])
    missing, unknown = sorted(assets - set(weights)), sorted(set(weights) - assets)
    if missing or unknown:
        raise ValueError(f"{date}: approved_weights must hold all {len(assets)} assets. "
                         f"Missing: {missing or 'none'}. Unknown: {unknown or 'none'}.")


def _hash_file(session: Session) -> Path:
    return session.dir.parent / ".hashes" / f"{session.id}.json"


def save_date_result(result: dict, session: Session) -> dict:
    date = result["as_of_date"]
    weights = result.get("approved_weights")
    if not weights:
        raise ValueError(f"{date}: final state has no 'approved_weights'")
    _check_assets(date, weights)
    pdf = session.dir / f"{date}.pdf"
    build_report(result, pdf)
    if not pdf.exists():
        raise FileNotFoundError(f"build_report did not write {pdf}")
    hashes = {"pdf": sha256_file(pdf), "weights": sha256_weights(weights)}
    session.results[date] = result
    session.hashes[date] = hashes
    hf = _hash_file(session)
    hf.parent.mkdir(exist_ok=True)
    hf.write_text(json.dumps(session.hashes, indent=2))
    return hashes


def write_weights(results: dict, session: Session | None = None) -> Path:
    """results: {date: final graph state}. Writes weights.csv and prints the Langfuse URL."""
    session = session or current_session()
    rows = {d: r["approved_weights"] for d, r in sorted(results.items())}
    df = pd.DataFrame.from_dict(rows, orient="index").fillna(0.0)
    df.index.name = "date"
    path = session.dir / "weights.csv"
    df.to_csv(path, float_format="%.10f")
    print(f"Wrote {path}")
    print(f"Langfuse session: {session.url or '(tracing off - a submission without it is penalised)'}")
    return path
