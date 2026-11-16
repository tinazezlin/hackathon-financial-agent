"""Run one reporting date (MOCK of the hackathon's src/core/run.py - do not edit there).

    result = run_date(workflow, "2025-10-31", llm.model_name, session)

Each call opens one Langfuse trace named "reporting_date <date>" inside the session,
runs the graph, saves <date>.pdf and records the fingerprints of the report and the
weights on the trace. If a date fails, call run_date again for it in the SAME session:
its last successful trace is the one that counts.
"""

from __future__ import annotations

import time

import pandas as pd

from src.core.results import save_date_result
from src.core.session import Session, current_session, langfuse_enabled


def _initial_state(date: str) -> dict:
    return {"as_of_date": date, "messages": [], "revision_count": 0, "adjustments": []}


def run_date(workflow, date: str, model_name: str | None = None, session: Session | None = None) -> dict:
    session = session or current_session()
    date = pd.Timestamp(date).strftime("%Y-%m-%d")
    t0 = time.time()
    config = {"run_name": "markets_workflow", "recursion_limit": 50}

    if langfuse_enabled():
        from langfuse import get_client, propagate_attributes
        from langfuse.langchain import CallbackHandler

        lf = get_client()
        name = f"reporting_date {date}"
        with lf.start_as_current_observation(as_type="chain", name=name,
                                             input={"as_of_date": date, "model": model_name}) as root:
            with propagate_attributes(session_id=session.id, trace_name=name):
                result = workflow.invoke(_initial_state(date),
                                         config={**config, "callbacks": [CallbackHandler()]})
                hashes = save_date_result(result, session)
            root.update(output={"approved_weights": result["approved_weights"]},
                        metadata={"sha256": hashes, "model": model_name})
    else:
        result = workflow.invoke(_initial_state(date), config=config)
        hashes = save_date_result(result, session)

    print(f"[{date}] done in {time.time() - t0:.1f}s  pdf={hashes['pdf'][:12]}...")
    return result
