"""LangGraph state machine for the content pipeline."""

from __future__ import annotations

import functools
import logging
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from orchestrator.nodes import (
    LLMProvider,
    aggregator_node,
    draft_node,
    publish_node,
    research_node,
    review_node,
    sub_researcher_node,
)
from orchestrator.planner import planner_node, should_plan
from orchestrator.state import PipelineState

logger = logging.getLogger(__name__)

AGENT_SEQUENCE = ["researcher", "drafter", "reviewer", "publisher"]


class ContentPipeline:
    """LangGraph content pipeline with planning, tool use, and parallel execution.

    Modeled after EnterpriseHub's LeadQualificationOrchestrator but fully
    decoupled from GHL/Jorge domain logic.  Uses the same patterns:
    - StateGraph with TypedDict state
    - Sequential nodes with compile()
    - ainvoke() for execution

    Args:
        llm: LLM provider (MockLLM or ClaudeLLM).
        tool_provider: Optional tool backend (web_search, retrieve_docs, …).
        use_planner: When True, a planner node decomposes the topic before research.
        use_parallel: When True, planner sub-tasks are dispatched to parallel
            researchers via LangGraph Send().  Automatically enables use_planner.
    """

    def __init__(
        self,
        llm: LLMProvider,
        tool_provider: Any = None,
        use_planner: bool = False,
        use_parallel: bool = False,
    ) -> None:
        if use_parallel and not use_planner:
            use_planner = True  # parallel execution requires planning
        self.llm = llm
        self.tool_provider = tool_provider
        self.use_planner = use_planner
        self.use_parallel = use_parallel
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
            "parallel_results": [],
            "use_parallel": self.use_parallel,
        }
        try:
            return await self._graph.ainvoke(initial_state)
        except Exception as exc:
            logger.error("Pipeline failed: %s", exc, exc_info=True)
            return {**initial_state, "error": str(exc)}

    def _build_graph(self) -> Any:
        """Build and compile the LangGraph state machine.

        Graph topology (all branches):

            START
              |
              +--[use_planner + complex topic]--> planner
              |                                      |
              |         +---[use_parallel + N>1 tasks]---> sub_researcher ×N
              |         |                                         |
              |         |                                    aggregator
              |         |                                         |
              +--[else]-+------------------------------------> researcher
                                                                  |
                                                               drafter
                                                                  |
                                                              reviewer
                                                         (conditional loop)
                                                                  |
                                                             publisher --> END
        """
        workflow = StateGraph(PipelineState)

        # Register all nodes
        workflow.add_node("planner", functools.partial(planner_node, llm=self.llm))
        workflow.add_node(
            "researcher",
            functools.partial(
                research_node, llm=self.llm, tool_provider=self.tool_provider
            ),
        )
        workflow.add_node(
            "sub_researcher",
            functools.partial(
                sub_researcher_node, llm=self.llm, tool_provider=self.tool_provider
            ),
        )
        workflow.add_node("aggregator", aggregator_node)
        workflow.add_node("drafter", functools.partial(draft_node, llm=self.llm))
        workflow.add_node("reviewer", functools.partial(review_node, llm=self.llm))
        workflow.add_node("publisher", functools.partial(publish_node, llm=self.llm))

        # ── Entry routing ──────────────────────────────────────────────────────
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

        # ── After planner: parallel fan-out or sequential ──────────────────────
        use_parallel = self.use_parallel

        def dispatch_research(state: PipelineState) -> str | list[Send]:
            plan = state.get("plan", [])
            if use_parallel and len(plan) > 1:
                return [
                    Send(
                        "sub_researcher",
                        {
                            "topic": task,
                            "total_tokens": 0,
                            "completed_agents": [],
                            "tool_calls": [],
                            "retrieved_context": [],
                            "parallel_results": [],
                            "plan": [],
                        },
                    )
                    for task in plan
                ]
            return "researcher"

        workflow.add_conditional_edges(
            "planner",
            dispatch_research,
            {"researcher": "researcher"},
        )

        # ── Parallel path: fan-in to aggregator, then drafter ──────────────────
        workflow.add_edge("sub_researcher", "aggregator")
        workflow.add_edge("aggregator", "drafter")

        # ── Sequential path ────────────────────────────────────────────────────
        workflow.add_edge("researcher", "drafter")

        # ── Common path ────────────────────────────────────────────────────────
        workflow.add_edge("drafter", "reviewer")

        def route_after_review(state: PipelineState) -> str:
            """Route to publisher if quality passes, else back to drafter."""
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
