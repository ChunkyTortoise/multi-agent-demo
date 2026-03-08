"""LangGraph state machine for the content pipeline."""

from __future__ import annotations

import functools
import logging
from typing import Any

from langgraph.graph import END, StateGraph

from orchestrator.nodes import (
    LLMProvider,
    draft_node,
    publish_node,
    research_node,
    review_node,
)
from orchestrator.state import PipelineState

logger = logging.getLogger(__name__)

AGENT_SEQUENCE = ["researcher", "drafter", "reviewer", "publisher"]


class ContentPipeline:
    """LangGraph-based sequential content pipeline.

    Modeled after EnterpriseHub's LeadQualificationOrchestrator but fully
    decoupled from GHL/Jorge domain logic. Uses the same patterns:
    - StateGraph with TypedDict state
    - Sequential nodes with compile()
    - ainvoke() for execution
    """

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm
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
        }
        try:
            return await self._graph.ainvoke(initial_state)
        except Exception as exc:
            logger.error("Pipeline failed: %s", exc, exc_info=True)
            return {**initial_state, "error": str(exc)}

    def _build_graph(self) -> Any:
        """Build and compile the LangGraph state machine."""
        workflow = StateGraph(PipelineState)

        # Bind llm to each node via functools.partial
        workflow.add_node(
            "researcher", functools.partial(research_node, llm=self.llm)
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

        # Pipeline: research -> draft -> review --(conditional)--> publish or back to draft
        workflow.set_entry_point("researcher")
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
