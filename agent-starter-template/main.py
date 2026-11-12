"""Run the workflow for the reporting dates from the command line.

    .venv/bin/python main.py                          # all 6 dates, Groq gpt-oss-120b
    .venv/bin/python main.py --dates 2025-10-31       # one date
    LLM_PROVIDER=mock .venv/bin/python main.py        # no LLM, free, seconds
    MODEL_ID=openai/gpt-oss-20b .venv/bin/python main.py

weights.csv is written only if ALL dates succeed (like the hackathon template).
"""

from __future__ import annotations

import argparse
import sys
import traceback

from src.agents.graph import build_workflow
from src.config import REPORTING_DATES
from src.core.results import write_weights
from src.core.run import run_date
from src.core.session import langfuse_enabled, start_session


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="*", default=REPORTING_DATES)
    ap.add_argument("--outputs", default=None, help="outputs folder (default ./outputs)")
    args = ap.parse_args()

    session = start_session(args.outputs)
    workflow = build_workflow()
    failed = []
    for d in args.dates:
        try:
            run_date(workflow, d, session)
        except Exception:  # keep going so we see all failures at once
            traceback.print_exc()
            failed.append(d)

    if langfuse_enabled():
        from langfuse import get_client

        get_client().flush()

    if failed:
        print(f"FAILED dates: {failed}. weights.csv NOT written.")
        return 1
    write_weights(session)
    return 0


if __name__ == "__main__":
    sys.exit(main())
