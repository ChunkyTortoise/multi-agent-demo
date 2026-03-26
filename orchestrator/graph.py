"""LangGraph state machine for the content pipeline."""

from __future__ import annotations

import functools
import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from orchestrator.nodes import (
    LLMProvider,
    draft_node,
    publish_node,
    research_node,
    review_node,
)
from orchestrator.planner import planner_node, should_plan
from orchestrator.state import PipelineState

logger = logging.getLogger(__name__)

AGENT_SEQUENCE = ["researcher", "drafter", "reviewer", "publisher"]


class ContentPipeline:
    """LangGraph-based content pipeline with optional planning and tool use.

    Modeled after EnterpriseHub's LeadQualificationOrchestrator but fully
    decoupled from GHL/Jorge domain logic. Uses the same patterns:
    - StateGraph with TypedDict state
    - Sequential nodes with compile()
    - ainvoke() for execution

    Args:
        llm: LLM provider (MockLLM or ClaudeLLM).
        tool_provider: Optional tool backend for grounded research. When
            supplied, the researcher calls web_search + retrieve_docs before
            generating its response.
        use_planner: When True, a planner node runs before the researcher for
            topics that benefit from decomposition (see ``should_plan()``).
    """

    def __init__(
        self,
        llm: LLMProvider,
        tool_provider: Any = None,
        use_planner: bool = False,
    ) -> None:
        self.llm = llm
        self.tool_provider = tool_provider
        self.use_planner = use_planner
        self._graph = self._build_graph()

    async def run(self, topic: str) -> PipelineState:
        """Run the full pipeline for a given topic."""
        initial_state: PipelineState = {
            "topic": topic,
            "research_output": {},
            "draft_output": {},
            "review_output": {},
            "publish_output": {},
            "current_agent": "researcher",
            "completed_agents": [],
            "total_tokens": 0,
            "error": "",
            "revision_count": 0,
            "review_score": 0.0,
            "tool_calls": [],
            "retrieved_context": [],
            "plan": [],
            "use_planner": self.use_planner,
        }
        try:
            return await self._graph.ainvoke(initial_state)
        except Exception as exc:
            logger.error("Pipeline failed: %s", exc, exc_info=True)
            return {**initial_state, "error": str(exc)}

    def _build_graph(self) -> Any:
        """Build and compile the LangGraph state machine."""
        workflow = StateGraph(PipelineState)

        # Always register the planner node (it's a no-op when not routed to)
        workflow.add_node(
            "planner", functools.partial(planner_node, llm=self.llm)
        )
        workflow.add_node(
            "researcher",
            functools.partial(
                research_node, llm=self.llm, tool_provider=self.tool_provider
            ),
        )
        workflow.add_node(
            "drafter", functools.partial(draft_node, llm=self.llm)
        )
        workflow.add_node(
            "reviewer", functools.partial(review_node, llm=self.llm)
        )
        workflow.add_node(
            "publisher", functools.partial(publish_node, llm=self.llm)
        )

        # Conditional entry: route through planner for complex topics when enabled
        use_planner = self.use_planner

        def route_entry(state: PipelineState) -> str:
            if use_planner and should_plan(state["topic"]):
                return "planner"
            return "researcher"

        workflow.add_conditional_edges(
            START,
            route_entry,
            {"planner": "planner", "researcher": "researcher"},
        )
        workflow.add_edge("planner", "researcher")
        workflow.add_edge("researcher", "drafter")
        workflow.add_edge("drafter", "reviewer")

        def route_after_review(state: PipelineState) -> str:
            """Route to publisher if quality passes, else back to drafter for revision."""
            score = state.get("review_score", 0.0)
            revision_count = state.get("revision_count", 0)
            if score >= 0.7 or revision_count >= 2:
                return "publisher"
            return "drafter"

        workflow.add_conditional_edges(
            "reviewer",
            route_after_review,
            {"publisher": "publisher", "drafter": "drafter"},
        )
        workflow.add_edge("publisher", END)

        return workflow.compile()
