"""Tests for tool definitions and MockToolProvider."""

from __future__ import annotations

import json

import pytest

from orchestrator.tools import TOOL_DEFINITIONS, MockToolProvider
from orchestrator.nodes import research_node, _tools_for_topic
from orchestrator.state import AgentOutput, PipelineState


# ---------------------------------------------------------------------------
# MockToolProvider: web_search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_web_search_returns_list():
    provider = MockToolProvider()
    result = await provider.execute("web_search", {"query": "LangGraph agents"})
    assert result["error"] == ""
    items = json.loads(result["output"])
    assert isinstance(items, list)
    assert len(items) > 0


@pytest.mark.asyncio
async def test_web_search_respects_max_results():
    provider = MockToolProvider()
    result = await provider.execute("web_search", {"query": "RAG", "max_results": 2})
    items = json.loads(result["output"])
    assert len(items) == 2


@pytest.mark.asyncio
async def test_web_search_result_has_expected_keys():
    provider = MockToolProvider()
    result = await provider.execute("web_search", {"query": "agents"})
    first = json.loads(result["output"])[0]
    assert "title" in first
    assert "snippet" in first
    assert "url" in first


# ---------------------------------------------------------------------------
# MockToolProvider: retrieve_docs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_docs_returns_list():
    provider = MockToolProvider()
    result = await provider.execute("retrieve_docs", {"query": "vector search"})
    assert result["error"] == ""
    docs = json.loads(result["output"])
    assert isinstance(docs, list)
    assert len(docs) > 0


@pytest.mark.asyncio
async def test_retrieve_docs_respects_top_k():
    provider = MockToolProvider()
    result = await provider.execute("retrieve_docs", {"query": "embeddings", "top_k": 2})
    docs = json.loads(result["output"])
    assert len(docs) == 2


@pytest.mark.asyncio
async def test_retrieve_docs_result_has_expected_keys():
    provider = MockToolProvider()
    result = await provider.execute("retrieve_docs", {"query": "RAG"})
    first = json.loads(result["output"])[0]
    assert "doc_id" in first
    assert "content" in first
    assert "score" in first


# ---------------------------------------------------------------------------
# MockToolProvider: calculate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_calculate_addition():
    provider = MockToolProvider()
    result = await provider.execute("calculate", {"expression": "2 + 2"})
    assert result["output"] == "4"
    assert result["error"] == ""


@pytest.mark.asyncio
async def test_calculate_complex_expression():
    provider = MockToolProvider()
    result = await provider.execute("calculate", {"expression": "(10 * 3) / 5"})
    assert result["output"] == "6.0"


@pytest.mark.asyncio
async def test_calculate_rejects_dangerous_input():
    provider = MockToolProvider()
    result = await provider.execute("calculate", {"expression": "__import__('os')"})
    assert "Error" in result["output"]


# ---------------------------------------------------------------------------
# MockToolProvider: summarize
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summarize_short_text_unchanged():
    provider = MockToolProvider()
    text = "Short text"
    result = await provider.execute("summarize", {"text": text, "max_words": 100})
    assert result["output"] == text


@pytest.mark.asyncio
async def test_summarize_long_text_truncated():
    provider = MockToolProvider()
    text = " ".join([f"word{i}" for i in range(200)])
    result = await provider.execute("summarize", {"text": text, "max_words": 10})
    assert result["output"].endswith("...")
    assert len(result["output"].split()) <= 11  # 10 words + "..."


# ---------------------------------------------------------------------------
# MockToolProvider: unknown tool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_tool_returns_error():
    provider = MockToolProvider()
    result = await provider.execute("does_not_exist", {})
    assert result["error"] != ""
    assert result["output"] == ""


# ---------------------------------------------------------------------------
# TOOL_DEFINITIONS schema validation
# ---------------------------------------------------------------------------


def test_tool_definitions_not_empty():
    assert len(TOOL_DEFINITIONS) >= 4


def test_tool_definitions_have_required_fields():
    for defn in TOOL_DEFINITIONS:
        assert "name" in defn
        assert "description" in defn
        assert "input_schema" in defn


def test_tool_definitions_input_schema_valid():
    for defn in TOOL_DEFINITIONS:
        schema = defn["input_schema"]
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema


def test_get_definitions_returns_tool_definitions():
    provider = MockToolProvider()
    defns = provider.get_definitions()
    names = {d["name"] for d in defns}
    assert "web_search" in names
    assert "retrieve_docs" in names
    assert "calculate" in names
    assert "summarize" in names


# ---------------------------------------------------------------------------
# research_node integration with tool_provider
# ---------------------------------------------------------------------------


class _SimpleLLM:
    """Minimal LLM for node-level tests."""

    def __init__(self) -> None:
        self.last_prompt = ""

    async def generate(self, prompt: str, agent_name: str) -> AgentOutput:
        self.last_prompt = prompt
        return AgentOutput(
            content=f"research result about {agent_name}",
            tokens_used=10,
        )


@pytest.mark.asyncio
async def test_research_node_with_tools_populates_tool_calls():
    llm = _SimpleLLM()
    provider = MockToolProvider()
    state: PipelineState = {
        "topic": "agentic AI systems",
        "total_tokens": 0,
        "completed_agents": [],
        "tool_calls": [],
        "retrieved_context": [],
        "plan": [],
    }
    result = await research_node(state, llm=llm, tool_provider=provider)
    assert len(result["tool_calls"]) >= 2
    tool_names = [tc["tool_name"] for tc in result["tool_calls"]]
    assert "web_search" in tool_names
    assert "retrieve_docs" in tool_names


@pytest.mark.asyncio
async def test_research_node_with_tools_populates_retrieved_context():
    llm = _SimpleLLM()
    provider = MockToolProvider()
    state: PipelineState = {
        "topic": "multi-agent orchestration",
        "total_tokens": 0,
        "completed_agents": [],
        "tool_calls": [],
        "retrieved_context": [],
        "plan": [],
    }
    result = await research_node(state, llm=llm, tool_provider=provider)
    assert isinstance(result["retrieved_context"], list)
    assert len(result["retrieved_context"]) > 0


@pytest.mark.asyncio
async def test_research_node_with_tools_injects_context_into_prompt():
    llm = _SimpleLLM()
    provider = MockToolProvider()
    state: PipelineState = {
        "topic": "RAG pipelines",
        "total_tokens": 0,
        "completed_agents": [],
        "tool_calls": [],
        "retrieved_context": [],
        "plan": [],
    }
    await research_node(state, llm=llm, tool_provider=provider)
    assert "Tool Results" in llm.last_prompt


@pytest.mark.asyncio
async def test_research_node_without_tools_backward_compatible():
    """No tool_provider -> same basic behaviour as the original node."""
    llm = _SimpleLLM()
    state: PipelineState = {
        "topic": "deep learning",
        "total_tokens": 0,
        "completed_agents": [],
        "tool_calls": [],
        "retrieved_context": [],
        "plan": [],
    }
    result = await research_node(state, llm=llm)
    assert result["research_output"]["content"] != ""
    assert result["tool_calls"] == []
    assert result["retrieved_context"] == []


@pytest.mark.asyncio
async def test_research_node_accumulates_prior_tool_calls():
    """Tool calls from prior nodes are preserved in state."""
    llm = _SimpleLLM()
    provider = MockToolProvider()
    prior_call = {"tool_name": "web_search", "input": {}, "output": "[]", "error": ""}
    state: PipelineState = {
        "topic": "LLM evaluation",
        "total_tokens": 0,
        "completed_agents": [],
        "tool_calls": [prior_call],
        "retrieved_context": [],
        "plan": [],
    }
    result = await research_node(state, llm=llm, tool_provider=provider)
    # Prior call preserved + new calls appended
    assert len(result["tool_calls"]) > 1
    assert result["tool_calls"][0]["tool_name"] == "web_search"


# ---------------------------------------------------------------------------
# _tools_for_topic helper
# ---------------------------------------------------------------------------


def test_tools_for_topic_includes_search_and_retrieval():
    pairs = _tools_for_topic("LangGraph multi-agent systems")
    names = [name for name, _ in pairs]
    assert "web_search" in names
    assert "retrieve_docs" in names
