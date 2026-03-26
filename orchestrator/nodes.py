"""Individual agent node functions for the content pipeline."""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from orchestrator.state import AgentOutput, PipelineState, ToolCall

logger = logging.getLogger(__name__)


class LLMProvider(Protocol):
    """Protocol for LLM providers (mock or real)."""

    async def generate(self, prompt: str, agent_name: str) -> AgentOutput: ...


def _build_prompt(agent: str, topic: str, prior: str) -> str:
    """Build a prompt for the given agent."""
    prompts = {
        "researcher": (
            f"Research the following topic thoroughly. Provide key facts, statistics, "
            f"and expert insights.\n\nTopic: {topic}"
        ),
        "drafter": (
            f"Using the research below, draft a compelling article. Use clear structure "
            f"with introduction, body sections, and conclusion.\n\nResearch:\n{prior}"
        ),
        "reviewer": (
            f"Review the following draft for accuracy, clarity, tone, and grammar. "
            f"Provide the improved version with corrections applied.\n\nDraft:\n{prior}"
        ),
        "publisher": (
            f"Format the following reviewed article for publication. Add a headline, "
            f"subheadings, and a brief summary. Output the final publishable version."
            f"\n\nReviewed article:\n{prior}"
        ),
    }
    return prompts[agent]


def _tools_for_topic(topic: str) -> list[tuple[str, dict[str, Any]]]:
    """Return (tool_name, inputs) pairs to execute for a given topic."""
    return [
        ("web_search", {"query": topic, "max_results": 3}),
        ("retrieve_docs", {"query": topic, "top_k": 3}),
    ]


def _format_tool_context(tool_calls: list[ToolCall]) -> str:
    """Format tool results into a readable context block for the LLM prompt."""
    sections: list[str] = []
    for tc in tool_calls:
        if tc.get("error"):
            continue
        name = tc["tool_name"]
        raw = tc.get("output", "")
        try:
            parsed = json.loads(raw)
            if name == "web_search" and isinstance(parsed, list):
                items = "\n".join(
                    f"  - [{r.get('title', '')}] {r.get('snippet', '')}"
                    for r in parsed
                )
                sections.append(f"Web Search Results:\n{items}")
            elif name == "retrieve_docs" and isinstance(parsed, list):
                items = "\n".join(
                    f"  - (score {r.get('score', 0):.2f}) {r.get('content', '')}"
                    for r in parsed
                )
                sections.append(f"Retrieved Documents:\n{items}")
            else:
                sections.append(f"{name}: {raw}")
        except (json.JSONDecodeError, TypeError):
            sections.append(f"{name}: {raw}")
    return "\n\n".join(sections)


async def research_node(
    state: PipelineState,
    *,
    llm: LLMProvider,
    tool_provider: Any = None,
) -> dict[str, Any]:
    """Researcher agent: gathers information on the topic.

    When tool_provider is supplied, executes web_search and retrieve_docs
    before prompting the LLM so the response is grounded in retrieved context.
    Falls back to prompt-only behaviour when tool_provider is None.
    """
    topic = state["topic"]
    prior_tool_calls: list[ToolCall] = list(state.get("tool_calls", []))
    retrieved_docs: list[dict[str, Any]] = []
    new_tool_calls: list[ToolCall] = []

    if tool_provider is not None:
        for tool_name, inputs in _tools_for_topic(topic):
            result: ToolCall = await tool_provider.execute(tool_name, inputs)
            new_tool_calls.append(result)
            if tool_name == "retrieve_docs" and not result.get("error"):
                try:
                    retrieved_docs = json.loads(result["output"])
                except (json.JSONDecodeError, TypeError):
                    pass

        tool_context = _format_tool_context(new_tool_calls)
        plan = state.get("plan", [])
        plan_section = (
            "\n\nResearch Plan:\n" + "\n".join(f"{i + 1}. {t}" for i, t in enumerate(plan))
            if plan
            else ""
        )
        prompt = (
            f"Research the following topic thoroughly. Provide key facts, statistics, "
            f"and expert insights. Use the tool results below as your primary sources.\n\n"
            f"Topic: {topic}{plan_section}\n\n"
            f"Tool Results:\n{tool_context}"
        )
    else:
        plan = state.get("plan", [])
        plan_section = (
            "\n\nResearch Plan:\n" + "\n".join(f"{i + 1}. {t}" for i, t in enumerate(plan))
            if plan
            else ""
        )
        prompt = _build_prompt("researcher", topic, "") + plan_section

    output = await llm.generate(prompt, "researcher")
    tokens_so_far = state.get("total_tokens", 0) + output.get("tokens_used", 0)
    completed = list(state.get("completed_agents", [])) + ["researcher"]
    return {
        "research_output": output,
        "current_agent": "drafter",
        "completed_agents": completed,
        "total_tokens": tokens_so_far,
        "tool_calls": prior_tool_calls + new_tool_calls,
        "retrieved_context": retrieved_docs,
    }


async def sub_researcher_node(
    state: PipelineState,
    *,
    llm: LLMProvider,
    tool_provider: Any = None,
) -> dict[str, Any]:
    """Research a single sub-task. Dispatched in parallel via LangGraph Send().

    Returns to ``parallel_results`` (Annotated reducer) so results from all
    concurrent invocations are automatically accumulated before the aggregator runs.
    """
    topic = state.get("topic", "")
    new_tool_calls: list[ToolCall] = []

    if tool_provider is not None:
        for tool_name, inputs in _tools_for_topic(topic):
            result: ToolCall = await tool_provider.execute(tool_name, inputs)
            new_tool_calls.append(result)
        tool_context = _format_tool_context(new_tool_calls)
        prompt = (
            f"Research this specific angle thoroughly. Use the tool results below "
            f"as primary sources.\n\nAngle: {topic}\n\nTool Results:\n{tool_context}"
        )
    else:
        prompt = f"Research this specific angle thoroughly:\n\n{topic}"

    output = await llm.generate(prompt, "researcher")
    return {
        "parallel_results": [
            {
                "task": topic,
                "content": output.get("content", ""),
                "tokens_used": output.get("tokens_used", 0),
            }
        ],
    }


async def aggregator_node(state: PipelineState) -> dict[str, Any]:
    """Merge all parallel research results into a single research_output.

    Runs once after the fan-in: all sub_researcher invocations have completed
    and their results accumulated in ``state["parallel_results"]`` via operator.add.
    """
    results: list[dict[str, Any]] = state.get("parallel_results", [])
    combined = "\n\n---\n\n".join(
        f"**Research Angle: {r['task']}**\n\n{r['content']}" for r in results
    )
    new_tokens = sum(r.get("tokens_used", 0) for r in results)
    tokens_so_far = state.get("total_tokens", 0) + new_tokens
    completed = list(state.get("completed_agents", [])) + ["researcher"]
    return {
        "research_output": {"content": combined, "tokens_used": new_tokens},
        "current_agent": "drafter",
        "completed_agents": completed,
        "total_tokens": tokens_so_far,
    }


async def draft_node(
    state: PipelineState, *, llm: LLMProvider
) -> dict[str, Any]:
    """Drafter agent: writes article from research."""
    prior = state.get("research_output", {}).get("content", "")
    prompt = _build_prompt("drafter", state["topic"], prior)
    output = await llm.generate(prompt, "drafter")
    tokens_so_far = state.get("total_tokens", 0) + output.get("tokens_used", 0)
    completed = list(state.get("completed_agents", [])) + ["drafter"]

    # Increment revision_count if this is a revision (review_output already set)
    revision_count = state.get("revision_count", 0)
    if state.get("review_output", {}).get("content"):
        revision_count += 1

    return {
        "draft_output": output,
        "current_agent": "reviewer",
        "completed_agents": completed,
        "total_tokens": tokens_so_far,
        "revision_count": revision_count,
    }


async def review_node(
    state: PipelineState, *, llm: LLMProvider
) -> dict[str, Any]:
    """Reviewer agent: checks and improves the draft."""
    prior = state.get("draft_output", {}).get("content", "")
    prompt = _build_prompt("reviewer", state["topic"], prior)
    output = await llm.generate(prompt, "reviewer")
    tokens_so_far = state.get("total_tokens", 0) + output.get("tokens_used", 0)
    completed = list(state.get("completed_agents", [])) + ["reviewer"]

    # Score based on content quality and revision history
    content = output.get("content", "")
    revision_count = state.get("revision_count", 0)

    if revision_count >= 2:
        score = 0.9  # forced pass after max revisions
    elif len(str(content)) > 100 and revision_count >= 1:
        score = 0.85
    elif len(str(content)) > 50:
        score = 0.75
    else:
        score = 0.4

    return {
        "review_output": output,
        "current_agent": "publisher",
        "completed_agents": completed,
        "total_tokens": tokens_so_far,
        "review_score": score,
    }


async def publish_node(
    state: PipelineState, *, llm: LLMProvider
) -> dict[str, Any]:
    """Publisher agent: formats for final publication."""
    prior = state.get("review_output", {}).get("content", "")
    prompt = _build_prompt("publisher", state["topic"], prior)
    output = await llm.generate(prompt, "publisher")
    tokens_so_far = state.get("total_tokens", 0) + output.get("tokens_used", 0)
    completed = list(state.get("completed_agents", [])) + ["publisher"]
    return {
        "publish_output": output,
        "current_agent": "done",
        "completed_agents": completed,
        "total_tokens": tokens_so_far,
    }
