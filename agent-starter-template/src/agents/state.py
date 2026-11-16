"""Shared graph state. Every agent reads from / writes to this dict, which is
what makes the decision process inspectable (and it all ends up in the report
and the Langfuse trace)."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class PortfolioState(TypedDict, total=False):
    as_of_date: str                         # reporting date (YYYY-MM-DD)
    messages: Annotated[list, add_messages]  # analyst <-> tools conversation
    analyst_summary: str                     # market intelligence brief (text)
    signals: list[dict]                      # [{ticker, view(-2..2), confidence, reason}]
    proposed_weights: dict[str, float]       # PM proposal (raw, before checks)
    pm_rationale: str
    approved_weights: dict[str, float]       # final approved weights (all assets, zeros included)
    violations: list[str]                    # constraint breaches found in the last review
    revision_count: int
    risk_feedback: str                       # message from risk review back to the PM
    adjustments: list[str]                   # audit log of deterministic changes
    risk_metrics: dict
    exposures: dict
    risk_commentary: str
