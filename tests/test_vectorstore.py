"""Tests for MockVectorStore and its integration with MockToolProvider."""

from __future__ import annotations

import json

import pytest

from orchestrator.tools import MockToolProvider
from orchestrator.vectorstore import MockVectorStore


# ---------------------------------------------------------------------------
# MockVectorStore
# ---------------------------------------------------------------------------


def test_mock_store_seeded_with_default_corpus():
    store = MockVectorStore()
    assert store.count() >= 5


def test_mock_store_query_returns_results():
    store = MockVectorStore()
    results = store.query("LangGraph agents", top_k=3)
    assert len(results) == 3


def test_mock_store_results_have_expected_keys():
    store = MockVectorStore()
    results = store.query("RAG retrieval")
    first = results[0]
    assert "doc_id" in first
    assert "content" in first
    assert "score" in first


def test_mock_store_scores_between_zero_and_one():
    store = MockVectorStore()
    results = store.query("fine-tuning QLoRA")
    for r in results:
        assert 0.0 <= r["score"] <= 1.0


def test_mock_store_ranked_by_relevance():
    """More relevant query should score higher than unrelated one on same doc."""
    store = MockVectorStore()
    results_relevant = store.query("LangGraph StateGraph conditional routing")
    results_unrelated = store.query("banana smoothie recipe")
    # Top score for relevant query should be higher
    assert results_relevant[0]["score"] >= results_unrelated[0]["score"]


def test_mock_store_add_documents_increases_count():
    store = MockVectorStore(seed_corpus=False)
    assert store.count() == 0
    store.add_documents([{"id": "x1", "text": "hello world"}])
    assert store.count() == 1


def test_mock_store_empty_returns_empty_list():
    store = MockVectorStore(seed_corpus=False)
    results = store.query("anything", top_k=3)
    assert results == []


def test_mock_store_top_k_respected():
    store = MockVectorStore()
    results = store.query("agents", top_k=2)
    assert len(results) == 2


# ---------------------------------------------------------------------------
# MockToolProvider uses vector store for retrieve_docs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_provider_retrieve_docs_uses_vector_store():
    store = MockVectorStore()
    provider = MockToolProvider(vector_store=store)
    result = await provider.execute("retrieve_docs", {"query": "RAG pipeline", "top_k": 3})
    docs = json.loads(result["output"])
    assert len(docs) == 3
    # Results come from real similarity search — first doc should mention RAG
    contents = " ".join(d["content"] for d in docs).lower()
    assert any(kw in contents for kw in ["retrieval", "rag", "vector", "embedding", "agent"])


@pytest.mark.asyncio
async def test_tool_provider_default_store_is_mock():
    """Default MockToolProvider instantiates a MockVectorStore automatically."""
    provider = MockToolProvider()
    result = await provider.execute("retrieve_docs", {"query": "LangGraph"})
    assert result["error"] == ""
    docs = json.loads(result["output"])
    assert len(docs) > 0
