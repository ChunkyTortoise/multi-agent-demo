"""Tests for the LangGraph content pipeline state machine."""

from __future__ import annotations

import pytest

from orchestrator.graph import AGENT_SEQUENCE, ContentPipeline
from orchestrator.state import AgentOutput, PipelineState


class FakeLLM:
    """Deterministic LLM for testing."""

    def __init__(self, token_count: int = 42) -> None:
        self.calls: list[tuple[str, str]] = []
        self.token_count = token_count

    async def generate(self, prompt: str, agent_name: str) -> AgentOutput:
        self.calls.append((agent_name, prompt))
        return AgentOutput(
            content=f"output from {agent_name}",
            tokens_used=self.token_count,
        )


@pytest.mark.asyncio
async def test_pipeline_runs_all_agents():
    """All four agents execute in sequence."""
    llm = FakeLLM()
    pipeline = ContentPipeline(llm=llm)
    result = await pipeline.run("test topic")

    agent_names = [call[0] for call in llm.calls]
    assert agent_names == AGENT_SEQUENCE


@pytest.mark.asyncio
async def test_pipeline_accumulates_tokens():
    """Total tokens should be sum of all agent token counts."""
    llm = FakeLLM(token_count=100)
    pipeline = ContentPipeline(llm=llm)
    result = await pipeline.run("test topic")

    assert result["total_tokens"] == 400  # 4 agents * 100 tokens


@pytest.mark.asyncio
async def test_pipeline_passes_output_downstream():
    """Each agent receives the prior agent's output in its prompt."""
    llm = FakeLLM()
    pipeline = ContentPipeline(llm=llm)
    result = await pipeline.run("test topic")

    # Drafter should receive researcher's output
    drafter_prompt = llm.calls[1][1]
    assert "output from researcher" in drafter_prompt

    # Reviewer should receive drafter's output
    reviewer_prompt = llm.calls[2][1]
    assert "output from drafter" in reviewer_prompt

    # Publisher should receive reviewer's output
    publisher_prompt = llm.calls[3][1]
    assert "output from reviewer" in publisher_prompt


@pytest.mark.asyncio
async def test_pipeline_state_completed_agents():
    """completed_agents list should contain all four agents after run."""
    llm = FakeLLM()
    pipeline = ContentPipeline(llm=llm)
    result = await pipeline.run("any topic")

    assert result["completed_agents"] == AGENT_SEQUENCE


@pytest.mark.asyncio
async def test_pipeline_outputs_populated():
    """All agent outputs should be populated in final state."""
    llm = FakeLLM()
    pipeline = ContentPipeline(llm=llm)
    result = await pipeline.run("any topic")

    assert result["research_output"]["content"] == "output from researcher"
    assert result["draft_output"]["content"] == "output from drafter"
    assert result["review_output"]["content"] == "output from reviewer"
    assert result["publish_output"]["content"] == "output from publisher"


@pytest.mark.asyncio
async def test_pipeline_no_error_on_success():
    """Error field should be empty on successful run."""
    llm = FakeLLM()
    pipeline = ContentPipeline(llm=llm)
    result = await pipeline.run("any topic")

    assert result.get("error", "") == ""


@pytest.mark.asyncio
async def test_pipeline_error_on_failure():
    """Pipeline captures errors without crashing."""

    class FailingLLM:
        async def generate(self, prompt: str, agent_name: str) -> AgentOutput:
            raise RuntimeError("LLM unavailable")

    pipeline = ContentPipeline(llm=FailingLLM())
    result = await pipeline.run("any topic")

    assert "LLM unavailable" in result.get("error", "")


@pytest.mark.asyncio
async def test_pipeline_current_agent_is_done():
    """After completion, current_agent should be 'done'."""
    llm = FakeLLM()
    pipeline = ContentPipeline(llm=llm)
    result = await pipeline.run("any topic")

    assert result["current_agent"] == "done"
