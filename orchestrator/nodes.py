"""Individual agent node functions for the content pipeline."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Protocol

from orchestrator.state import AgentOutput, PipelineState

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


async def research_node(
    state: PipelineState, *, llm: LLMProvider
) -> dict[str, Any]:
    """Researcher agent: gathers information on the topic."""
    prompt = _build_prompt("researcher", state["topic"], "")
    output = await llm.generate(prompt, "researcher")
    tokens_so_far = state.get("total_tokens", 0) + output.get("tokens_used", 0)
    completed = list(state.get("completed_agents", [])) + ["researcher"]
    return {
        "research_output": output,
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
    return {
        "draft_output": output,
        "current_agent": "reviewer",
        "completed_agents": completed,
        "total_tokens": tokens_so_far,
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
    return {
        "review_output": output,
        "current_agent": "publisher",
        "completed_agents": completed,
        "total_tokens": tokens_so_far,
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
