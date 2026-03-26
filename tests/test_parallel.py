"""Tests for parallel fan-out/fan-in execution via LangGraph Send()."""

from __future__ import annotations

import pytest

from orchestrator.graph import ContentPipeline
from orchestrator.nodes import aggregator_node
from orchestrator.state import AgentOutput, PipelineState


class _FakeLLM:
    """Deterministic LLM with enough outputs for all agent roles."""

    def __init__(self, token_count: int = 50) -> None:
        self.calls: list[tuple[str, str]] = []
        self.token_count = token_count

    async def generate(self, prompt: str, agent_name: str) -> AgentOutput:
        self.calls.append((agent_name, prompt))
        if agent_name == "planner":
            content = (
                "1. Research current market size and growth trends\n"
                "2. Analyze key technical developments and frameworks\n"
                "3. Identify major players and competitive dynamics"
            )
        else:
            content = f"output from {agent_name}: " + "detailed content " * 5
        return AgentOutput(content=content.strip(), tokens_used=self.token_count)


# ---------------------------------------------------------------------------
# use_parallel auto-enables use_planner
# ---------------------------------------------------------------------------


def test_use_parallel_implies_use_planner():
    llm = _FakeLLM()
    pipeline = ContentPipeline(llm=llm, use_parallel=True)
    assert pipeline.use_planner is True


# ---------------------------------------------------------------------------
# Parallel dispatch for complex topics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parallel_pipeline_calls_multiple_researchers():
    """Complex topic with use_parallel=True dispatches N researcher calls."""
    llm = _FakeLLM()
    pipeline = ContentPipeline(llm=llm, use_parallel=True)
    await pipeline.run("the impact of multi-agent AI on enterprise workflows")

    researcher_calls = [c for c in llm.calls if c[0] == "researcher"]
    # Planner returns 3 sub-tasks → 3 parallel researcher calls
    assert len(researcher_calls) >= 2


@pytest.mark.asyncio
async def test_parallel_pipeline_runs_planner_first():
    """Planner executes before any researcher calls."""
    llm = _FakeLLM()
    pipeline = ContentPipeline(llm=llm, use_parallel=True)
    await pipeline.run("the impact of multi-agent AI on enterprise workflows")

    names = [c[0] for c in llm.calls]
    assert "planner" in names
    assert names.index("planner") < names.index("researcher")


@pytest.mark.asyncio
async def test_parallel_results_accumulated_in_state():
    """Final state has parallel_results with one entry per sub-task."""
    llm = _FakeLLM()
    pipeline = ContentPipeline(llm=llm, use_parallel=True)
    result = await pipeline.run("the impact of multi-agent AI on enterprise workflows")

    assert isinstance(result.get("parallel_results"), list)
    assert len(result["parallel_results"]) >= 2


@pytest.mark.asyncio
async def test_parallel_research_output_contains_all_angles():
    """research_output from aggregator contains content from multiple sub-tasks."""
    llm = _FakeLLM()
    pipeline = ContentPipeline(llm=llm, use_parallel=True)
    result = await pipeline.run("the impact of multi-agent AI on enterprise workflows")

    content = result.get("research_output", {}).get("content", "")
    # Aggregator combines all parallel results; content should be substantial
    assert len(content) > 100


# ---------------------------------------------------------------------------
# Sequential fallback for simple topics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parallel_pipeline_falls_back_to_sequential_for_simple_topic():
    """Simple topic (should_plan=False) goes directly to researcher, no fan-out."""
    llm = _FakeLLM()
    pipeline = ContentPipeline(llm=llm, use_parallel=True)
    await pipeline.run("LangGraph")

    names = [c[0] for c in llm.calls]
    assert "planner" not in names
    assert names.count("researcher") == 1


# ---------------------------------------------------------------------------
# aggregator_node unit test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aggregator_merges_parallel_results():
    state: PipelineState = {
        "topic": "test",
        "total_tokens": 10,
        "completed_agents": [],
        "parallel_results": [
            {"task": "angle A", "content": "Content about A", "tokens_used": 30},
            {"task": "angle B", "content": "Content about B", "tokens_used": 20},
        ],
    }
    result = await aggregator_node(state)

    assert "angle A" in result["research_output"]["content"]
    assert "angle B" in result["research_output"]["content"]
    assert result["total_tokens"] == 60  # 10 prior + 30 + 20
    assert "researcher" in result["completed_agents"]
