"""Planner agent node: decomposes a topic into a structured research plan.

Demonstrates the planning pattern: before research begins, a planner LLM call
produces an ordered list of sub-questions or research angles that guide the
researcher agent's work.
"""

from __future__ import annotations

import re
from typing import Any

from orchestrator.state import PipelineState


_PLANNING_PROMPT = (
    "Decompose the following research topic into 2-4 specific sub-questions or "
    "research angles. Each sub-question should be concrete and independently "
    "researchable.\n\n"
    "Topic: {topic}\n\n"
    "Output format (numbered list only, no preamble):\n"
    "1. [first sub-question]\n"
    "2. [second sub-question]\n"
    "3. [third sub-question]"
)


def should_plan(topic: str) -> bool:
    """Return True when the topic is complex enough to warrant decomposition.

    Heuristic: topics longer than 4 words usually involve multiple angles.
    """
    return len(topic.split()) > 4


def _parse_plan(content: str) -> list[str]:
    """Parse a numbered list from LLM output into a list of task strings."""
    tasks: list[str] = []
    for line in content.strip().splitlines():
        line = line.strip()
        match = re.match(r"^\d+[.)]\s+(.+)", line)
        if match:
            tasks.append(match.group(1).strip())
    # Fallback: treat entire content as a single task if no numbered items found
    return tasks if tasks else [content.strip()]


async def planner_node(state: PipelineState, *, llm: Any) -> dict[str, Any]:
    """Decompose the pipeline topic into an ordered research plan.

    Calls the LLM with a structured decomposition prompt, parses the numbered
    list response, and stores the result in ``state["plan"]``.
    """
    topic = state["topic"]
    prompt = _PLANNING_PROMPT.format(topic=topic)
    output = await llm.generate(prompt, "planner")
    plan = _parse_plan(output.get("content", ""))
    tokens_so_far = state.get("total_tokens", 0) + output.get("tokens_used", 0)
    return {
        "plan": plan,
        "current_agent": "researcher",
        "total_tokens": tokens_so_far,
    }
