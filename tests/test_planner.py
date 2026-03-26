"""Tests for the planner node and related helpers."""

from __future__ import annotations

import pytest

from orchestrator.graph import ContentPipeline
from orchestrator.planner import _parse_plan, planner_node, should_plan
from orchestrator.state import AgentOutput, PipelineState


class _FakeLLM:
    """Deterministic LLM that returns suitable content for planner tests."""

    def __init__(self, token_count: int = 42) -> None:
        self.calls: list[tuple[str, str]] = []
        self.token_count = token_count

    async def generate(self, prompt: str, agent_name: str) -> AgentOutput:
        self.calls.append((agent_name, prompt))
        if agent_name == "planner":
            content = (
                "1. Research current market size and growth trends\n"
                "2. Analyze key technical developments and frameworks\n"
                "3. Identify major industry players and competition\n"
                "4. Examine implementation challenges and ROI metrics"
            )
        else:
            content = f"output from {agent_name}: " + "detailed content " * 5
        return AgentOutput(content=content.strip(), tokens_used=self.token_count)


# ---------------------------------------------------------------------------
# should_plan helper
# ---------------------------------------------------------------------------


def test_should_plan_complex_topic():
    assert should_plan("the impact of multi-agent AI on enterprise workflows") is True


def test_should_plan_simple_topic():
    assert should_plan("LangGraph") is False


def test_should_plan_four_word_topic_is_simple():
    assert should_plan("agentic AI systems overview") is False


def test_should_plan_five_word_topic_is_complex():
    assert should_plan("agentic AI systems in production") is True


# ---------------------------------------------------------------------------
# _parse_plan helper
# ---------------------------------------------------------------------------


def test_parse_plan_numbered_list():
    content = "1. Research trends\n2. Analyze frameworks\n3. Study competitors"
    plan = _parse_plan(content)
    assert plan == ["Research trends", "Analyze frameworks", "Study competitors"]


def test_parse_plan_with_period_separator():
    content = "1. First task\n2. Second task"
    plan = _parse_plan(content)
    assert len(plan) == 2


def test_parse_plan_with_paren_separator():
    content = "1) First task\n2) Second task"
    plan = _parse_plan(content)
    assert len(plan) == 2


def test_parse_plan_fallback_on_non_numbered():
    content = "Just a plain sentence with no numbers"
    plan = _parse_plan(content)
    assert plan == [content]


def test_parse_plan_strips_whitespace():
    content = "  1. Task one   \n  2. Task two  "
    plan = _parse_plan(content)
    assert plan[0] == "Task one"
    assert plan[1] == "Task two"


# ---------------------------------------------------------------------------
# planner_node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planner_node_produces_plan():
    llm = _FakeLLM()
    state: PipelineState = {"topic": "enterprise AI adoption in 2025", "total_tokens": 0}
    result = await planner_node(state, llm=llm)
    assert isinstance(result["plan"], list)
    assert len(result["plan"]) >= 1


@pytest.mark.asyncio
async def test_planner_node_plan_has_multiple_items():
    llm = _FakeLLM()
    state: PipelineState = {"topic": "the future of large language models in production", "total_tokens": 0}
    result = await planner_node(state, llm=llm)
    assert len(result["plan"]) >= 2


@pytest.mark.asyncio
async def test_planner_node_stores_plan_in_state():
    llm = _FakeLLM()
    state: PipelineState = {"topic": "multi-agent AI orchestration with LangGraph", "total_tokens": 0}
    result = await planner_node(state, llm=llm)
    assert "plan" in result
    assert all(isinstance(t, str) for t in result["plan"])


@pytest.mark.asyncio
async def test_planner_node_accumulates_tokens():
    llm = _FakeLLM(token_count=50)
    state: PipelineState = {"topic": "agentic AI for enterprise use cases today", "total_tokens": 100}
    result = await planner_node(state, llm=llm)
    assert result["total_tokens"] == 150


# ---------------------------------------------------------------------------
# ContentPipeline with use_planner=True
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_with_planner_calls_planner_for_complex_topic():
    llm = _FakeLLM()
    pipeline = ContentPipeline(llm=llm, use_planner=True)
    await pipeline.run("the impact of multi-agent AI on enterprise workflows")

    agent_names = [call[0] for call in llm.calls]
    assert "planner" in agent_names
    planner_idx = agent_names.index("planner")
    researcher_idx = agent_names.index("researcher")
    assert planner_idx < researcher_idx


@pytest.mark.asyncio
async def test_pipeline_skips_planner_for_simple_topic():
    llm = _FakeLLM()
    pipeline = ContentPipeline(llm=llm, use_planner=True)
    await pipeline.run("LangGraph")

    agent_names = [call[0] for call in llm.calls]
    assert "planner" not in agent_names
    assert "researcher" in agent_names


@pytest.mark.asyncio
async def test_pipeline_without_use_planner_never_calls_planner():
    llm = _FakeLLM()
    pipeline = ContentPipeline(llm=llm, use_planner=False)
    await pipeline.run("the impact of multi-agent AI on enterprise workflows")

    agent_names = [call[0] for call in llm.calls]
    assert "planner" not in agent_names


@pytest.mark.asyncio
async def test_pipeline_plan_available_in_final_state():
    llm = _FakeLLM()
    pipeline = ContentPipeline(llm=llm, use_planner=True)
    result = await pipeline.run("the future of LLM deployment in production systems")
    assert isinstance(result.get("plan"), list)
    assert len(result["plan"]) >= 1
