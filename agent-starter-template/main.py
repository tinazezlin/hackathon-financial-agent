"""Run the workflow for the reporting dates from the command line.

    .venv/bin/python main.py                          # all 6 dates, Groq gpt-oss-120b
    .venv/bin/python main.py --dates 2025-10-31       # one date
    LLM_PROVIDER=mock .venv/bin/python main.py        # no LLM, free, seconds
    MODEL_ID=openai/gpt-oss-20b .venv/bin/python main.py

Like the official notebook: results = {date: run_date(...)}, then write_weights(results,
session). A failed date is retried in the SAME session (its last successful trace counts),
and weights.csv is written only when every date has a result.
"""

from __future__ import annotations

import argparse
import sys
import traceback

from src.agents.graph import build_workflow
from src.config import MODEL_ID, REPORTING_DATES
from src.core.results import write_weights
from src.core.run import run_date
from src.core.session import langfuse_enabled, start_session


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="*", default=REPORTING_DATES)
    ap.add_argument("--outputs", default=None, help="outputs folder (default ./outputs)")
    ap.add_argument("--attempts", type=int, default=2, help="tries per date, in the same session")
    args = ap.parse_args()

    session = start_session(args.outputs)
    workflow = build_workflow()
    results = {}
    for attempt in range(1, args.attempts + 1):
        for d in [d for d in args.dates if d not in results]:
            try:
                results[d] = run_date(workflow, d, MODEL_ID, session)
            except Exception:  # keep going so we see all failures at once
                traceback.print_exc()
                print(f"[{d}] FAILED (attempt {attempt}/{args.attempts})")

    if langfuse_enabled():
        from langfuse import get_client

        get_client().flush()

    failed = [d for d in args.dates if d not in results]
    if failed:
        print(f"FAILED dates: {failed}. weights.csv NOT written. Rerun them in this session "
              f"(notebook) or start a new run.")
        return 1
    write_weights(results, session)
    return 0


if __name__ == "__main__":
    sys.exit(main())
