"""Tests for the LangGraph content pipeline state machine."""

from __future__ import annotations

import pytest

from orchestrator.graph import AGENT_SEQUENCE, ContentPipeline
from orchestrator.state import AgentOutput


class FakeLLM:
    """Deterministic LLM for testing."""

    def __init__(self, token_count: int = 42) -> None:
        self.calls: list[tuple[str, str]] = []
        self.token_count = token_count

    async def generate(self, prompt: str, agent_name: str) -> AgentOutput:
        self.calls.append((agent_name, prompt))
        # Content must be >50 chars to pass review scoring threshold (0.75)
        content = f"output from {agent_name}: " + "detailed content " * 5
        return AgentOutput(
            content=content.strip(),
            tokens_used=self.token_count,
        )


@pytest.mark.asyncio
async def test_pipeline_runs_all_agents():
    """All four agents execute in sequence."""
    llm = FakeLLM()
    pipeline = ContentPipeline(llm=llm)
    await pipeline.run("test topic")

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
    await pipeline.run("test topic")

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

    assert "output from researcher" in result["research_output"]["content"]
    assert "output from drafter" in result["draft_output"]["content"]
    assert "output from reviewer" in result["review_output"]["content"]
    assert "output from publisher" in result["publish_output"]["content"]


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


# --- Conditional routing / revision loop tests ---


class ShortOutputLLM:
    """LLM that returns short content, triggering revision loops."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def generate(self, prompt: str, agent_name: str) -> AgentOutput:
        self.calls.append((agent_name, prompt))
        # Short content (<50 chars) scores 0.4, below 0.7 threshold
        return AgentOutput(content=f"short {agent_name}", tokens_used=10)


@pytest.mark.asyncio
async def test_pipeline_review_score_populated():
    """review_score should be set after pipeline run."""
    llm = FakeLLM()
    pipeline = ContentPipeline(llm=llm)
    result = await pipeline.run("any topic")

    assert result["review_score"] >= 0.7


@pytest.mark.asyncio
async def test_pipeline_no_revision_when_quality_passes():
    """No revision loop when content passes quality threshold."""
    llm = FakeLLM()  # >50 char content, scores 0.75
    pipeline = ContentPipeline(llm=llm)
    result = await pipeline.run("any topic")

    assert result["revision_count"] == 0
    agent_names = [call[0] for call in llm.calls]
    assert agent_names == AGENT_SEQUENCE


@pytest.mark.asyncio
async def test_pipeline_revision_loop_on_low_quality():
    """Pipeline loops back to drafter when review score is low."""
    llm = ShortOutputLLM()  # <50 chars, scores 0.4
    pipeline = ContentPipeline(llm=llm)
    result = await pipeline.run("any topic")

    # Should have revision loops (max 2 before forced pass)
    assert result["revision_count"] >= 1
    agent_names = [call[0] for call in llm.calls]
    # Should have more than the standard 4 agent calls
    assert len(agent_names) > 4
    # Should still end with publisher
    assert agent_names[-1] == "publisher"


@pytest.mark.asyncio
async def test_pipeline_max_revisions_capped():
    """Pipeline forces pass after 2 revisions to prevent infinite loops."""
    llm = ShortOutputLLM()
    pipeline = ContentPipeline(llm=llm)
    result = await pipeline.run("any topic")

    # revision_count should not exceed 2
    assert result["revision_count"] <= 2
    # Score should be 0.9 (forced pass) after max revisions
    assert result["review_score"] >= 0.9


@pytest.mark.asyncio
async def test_pipeline_revision_count_defaults_zero():
    """Initial revision_count should be 0."""
    llm = FakeLLM()
    pipeline = ContentPipeline(llm=llm)
    result = await pipeline.run("any topic")

    assert result["revision_count"] == 0
