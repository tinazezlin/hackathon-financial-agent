"""Graph nodes.

    analyst            LLM + tools      -> market intelligence brief
    analyst_signals    LLM (structured) -> per-asset views
    portfolio_manager  LLM (structured) -> proposed weights + rationale
    risk_review        plain Python     -> checks constraints, may send PM back once,
                                           otherwise projects onto the feasible set; risk metrics
    risk_manager       LLM              -> risk commentary

In LLM_PROVIDER=mock mode every LLM node uses a deterministic heuristic instead,
so the whole pipeline can be run and tested for free.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from src.agents import prompts
from src.agents.tools import ANALYST_TOOLS, asset_statistics_table, market_indicators_table
from src.config import (LIMITS, LOOKBACK_DAYS, MAX_REVISIONS, MAX_SINGLE_ASSET, MAX_TOOL_ROUNDS, MODEL_ID,
                        STRATEGIC_BENCHMARK)
from src.core.llm import get_llm, is_mock, provider
from src.utils import data, pit
from src.utils.analytics import (asset_statistics, check_constraints, exposures, portfolio_metrics,
                                 project_to_constraints, round_weights)


# --------------------------------------------------------------------------- schemas
class AssetView(BaseModel):
    ticker: str
    # float, not int: models happily answer 0.5 and a strict int schema then fails the whole call
    view: float = Field(ge=-2, le=2, description="-2 strong underweight ... +2 strong overweight")
    confidence: float = Field(ge=0, le=1)
    reason: str


class Signals(BaseModel):
    views: list[AssetView]


class TickerWeight(BaseModel):
    ticker: str
    weight: float = Field(ge=0, le=1)


class Proposal(BaseModel):
    weights: list[TickerWeight]
    rationale: str


# --------------------------------------------------------------------------- helpers
def _tickers() -> list[str]:
    return data.read_assets_metadata()["ticker"].tolist()


def _universe_table() -> str:
    m = data.read_assets_metadata()[["ticker", "asset_class", "jurisdiction", "currency", "duration"]]
    m = m.assign(benchmark=m["ticker"].map(STRATEGIC_BENCHMARK).fillna(0).round(3))
    return m.to_string(index=False)


def _constraints_text() -> str:
    lines = ["- No short selling (all weights >= 0); weights sum to 1"]
    lines += [f"- {lim.label}" for lim in LIMITS]
    if MAX_SINGLE_ASSET:
        lines.append(f"- No single ticker above {MAX_SINGLE_ASSET:.0%}")
    return "\n".join(lines)


def _views_text(signals: list[dict]) -> str:
    return "\n".join(f"{v['ticker']}: {v['view']:+.1f}, {v['confidence']:.2f} - {v['reason']}" for v in signals)


def _sys(template: str, state: dict, **kw) -> SystemMessage:
    return SystemMessage(template.format(as_of=state["as_of_date"], **kw))


def _structured(schema, messages: list, run_name: str, retries: int = 2):
    """Structured-output call with retries; the error is fed back to the model."""
    # Native JSON-schema mode is more reliable than tool calling for gpt-oss on Groq;
    # other providers may only support function calling.
    method = "json_schema" if provider() == "groq" else "function_calling"
    llm = get_llm(MODEL_ID).with_structured_output(schema, method=method)
    msgs = list(messages)
    for attempt in range(retries + 1):
        try:
            return llm.invoke(msgs, config={"run_name": f"{run_name}" + (f"_retry{attempt}" if attempt else "")})
        except Exception as e:  # schema/validation errors from the provider or pydantic
            if attempt == retries:
                raise
            msgs = list(messages) + [HumanMessage(f"Your previous answer was invalid: {str(e)[:500]}. "
                                                  "Answer again, strictly following the schema.")]


# --------------------------------------------------------------------------- analyst
def analyst(state: dict) -> dict:
    as_of = state["as_of_date"]
    if is_mock():
        brief = ("[MOCK BRIEF - no LLM]\n```\n" + market_indicators_table(as_of) + "\n\n"
                 + asset_statistics_table(as_of, 252) + "\n```")
        return {"messages": [AIMessage(brief)], "analyst_summary": brief}

    msgs = state.get("messages") or []
    if not msgs:
        msgs = [HumanMessage(f"Prepare the market intelligence brief for the rebalancing on {as_of}.")]
    tool_rounds = sum(isinstance(m, ToolMessage) for m in msgs)
    llm = get_llm(MODEL_ID)
    model = llm if tool_rounds >= MAX_TOOL_ROUNDS else llm.bind_tools(ANALYST_TOOLS)
    resp = model.invoke([_sys(prompts.ANALYST_SYSTEM, state)] + msgs, config={"run_name": "analyst_llm"})
    out = {"messages": ([msgs[0]] if not state.get("messages") else []) + [resp]}
    if not getattr(resp, "tool_calls", None):
        out["analyst_summary"] = resp.content
    return out


def route_after_analyst(state: dict) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "analyst_signals"


# --------------------------------------------------------------------------- signals
def analyst_signals(state: dict) -> dict:
    tickers = _tickers()
    if is_mock():
        st = asset_statistics(pit.assets_asof(state["as_of_date"]), 252)
        z = st["ret_3m"] / (st["ann_vol"] * np.sqrt(63 / 252)).replace(0, np.nan)
        views = [{"ticker": t, "view": float(np.clip(round(z.get(t, 0) or 0), -2, 2)), "confidence": 0.5,
                  "reason": f"3m risk-adjusted momentum z={z.get(t, 0):.2f}"} for t in tickers]
        return {"signals": views}

    res: Signals = _structured(Signals, [
        _sys(prompts.SIGNALS_SYSTEM, state),
        HumanMessage(f"Tickers:\n{_universe_table()}\n\nAnalyst brief:\n{state['analyst_summary']}"),
    ], "analyst_signals_llm")
    got = {v.ticker: v.model_dump() for v in res.views if v.ticker in tickers}
    views = [got.get(t, {"ticker": t, "view": 0, "confidence": 0.0, "reason": "no view returned"})
             for t in tickers]
    return {"signals": views}


# --------------------------------------------------------------------------- portfolio manager
def portfolio_manager(state: dict) -> dict:
    tickers = _tickers()
    bench = pd.Series(STRATEGIC_BENCHMARK).reindex(tickers).fillna(0.0)
    if is_mock():
        view = pd.Series({s["ticker"]: s["view"] * s["confidence"] for s in state["signals"]})
        w = (bench * (1 + 0.5 * view.reindex(tickers).fillna(0))).clip(lower=0)
        w = w / w.sum()
        return {"proposed_weights": w.round(6).to_dict(),
                "pm_rationale": "[MOCK] Benchmark tilted by +/-50% x view x confidence."}

    feedback = state.get("risk_feedback")
    user = (f"Universe and strategic benchmark:\n{_universe_table()}\n\n"
            f"Analyst views (ticker: view, confidence - reason):\n{_views_text(state['signals'])}\n\n"
            f"Market brief:\n{state['analyst_summary']}")
    if feedback:
        user += (f"\n\nRISK REVIEW REJECTED YOUR PREVIOUS PROPOSAL:\n{feedback}\n"
                 f"Previous proposal: {json.dumps(state.get('proposed_weights'))}\nFix it.")
    res: Proposal = _structured(Proposal, [_sys(prompts.PM_SYSTEM, state, constraints=_constraints_text()),
                                           HumanMessage(user)], "portfolio_manager_llm")
    w = {t: 0.0 for t in tickers}
    for tw in res.weights:
        if tw.ticker in w:
            w[tw.ticker] += float(tw.weight)
    return {"proposed_weights": w, "pm_rationale": res.rationale}


# --------------------------------------------------------------------------- risk review (no LLM)
def risk_review(state: dict) -> dict:
    meta = data.read_assets_metadata()
    tickers = meta["ticker"].tolist()
    raw = pd.Series(state["proposed_weights"]).reindex(tickers).fillna(0.0)
    adjustments = list(state.get("adjustments") or [])

    w = raw.clip(lower=0)
    if (raw < 0).any():
        adjustments.append("Negative weights set to 0.")
    if abs(w.sum() - 1) > 1e-6:
        adjustments.append(f"Weights summed to {w.sum():.4f}; renormalised to 1.")
        w = w / w.sum()

    violations = check_constraints(w, meta)
    rev = state.get("revision_count", 0)
    if violations and rev < MAX_REVISIONS:
        return {"violations": violations, "revision_count": rev + 1, "adjustments": adjustments,
                "risk_feedback": "Constraint violations:\n- " + "\n- ".join(violations)}
    if violations:
        before = w.copy()
        w = project_to_constraints(w, meta)
        moved = (w - before).abs().sum() / 2
        adjustments.append(f"Still violating after {rev} revision(s): {violations}. Projected onto the "
                           f"feasible set (least-squares), turnover {moved:.2%}.")

    w = round_weights(w)  # final rounding that cannot break limits
    assert not check_constraints(w, meta), f"final portfolio not compliant: {check_constraints(w, meta)}"

    rets = pit.returns_base_ccy_asof(state["as_of_date"], LOOKBACK_DAYS)
    bench = pd.Series(STRATEGIC_BENCHMARK).reindex(tickers).fillna(0.0)
    metrics = portfolio_metrics(w, rets, meta, benchmark=bench)
    expo = {f: exposures(w, meta, f).round(4).to_dict() for f in ("asset_class", "jurisdiction", "currency")}
    return {"weights": w.to_dict(), "violations": violations, "risk_feedback": "",
            "adjustments": adjustments, "risk_metrics": metrics, "exposures": expo}


def route_after_risk_review(state: dict) -> str:
    return "portfolio_manager" if state.get("risk_feedback") else "risk_manager"


# --------------------------------------------------------------------------- risk manager
def risk_manager(state: dict) -> dict:
    m = state["risk_metrics"]
    if is_mock():
        txt = (f"[MOCK] Ex-post vol {m['ann_vol']:.2%} (benchmark {m['benchmark_ann_vol']:.2%}), tracking "
               f"error {m['tracking_error']:.2%}, 1d VaR95 {m['var95_1d']:.2%}, duration {m['duration']:.2f}. "
               f"Adjustments: {state.get('adjustments') or 'none'}.")
        return {"risk_commentary": txt}
    payload = {"weights": {k: v for k, v in state["weights"].items() if v > 0}, "exposures": state["exposures"],
               "risk_metrics": m, "adjustments": state.get("adjustments"), "pm_rationale": state["pm_rationale"]}
    resp = get_llm(MODEL_ID).invoke([_sys(prompts.RISK_MANAGER_SYSTEM, state),
                                     HumanMessage(json.dumps(payload, default=str))],
                                    config={"run_name": "risk_manager_llm"})
    return {"risk_commentary": resp.content}
