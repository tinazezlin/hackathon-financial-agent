"""The LangGraph workflow (sequential pipeline with a tool loop and one feedback loop).

    START -> analyst <-> tools
               |
               v
         analyst_signals -> portfolio_manager -> risk_review --(violations, once)--> portfolio_manager
                                                     |
                                                     v
                                               risk_manager -> END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.agents import nodes
from src.agents.state import PortfolioState
from src.agents.tools import ANALYST_TOOLS


def build_workflow():
    g = StateGraph(PortfolioState)
    g.add_node("analyst", nodes.analyst)
    g.add_node("tools", ToolNode(ANALYST_TOOLS))
    g.add_node("analyst_signals", nodes.analyst_signals)
    g.add_node("portfolio_manager", nodes.portfolio_manager)
    g.add_node("risk_review", nodes.risk_review)
    g.add_node("risk_manager", nodes.risk_manager)

    g.add_edge(START, "analyst")
    g.add_conditional_edges("analyst", nodes.route_after_analyst, ["tools", "analyst_signals"])
    g.add_edge("tools", "analyst")
    g.add_edge("analyst_signals", "portfolio_manager")
    g.add_edge("portfolio_manager", "risk_review")
    g.add_conditional_edges("risk_review", nodes.route_after_risk_review, ["portfolio_manager", "risk_manager"])
    g.add_edge("risk_manager", END)
    return g.compile()
