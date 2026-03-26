"""Tool definitions and mock provider for agentic research nodes.

Demonstrates the tool-use pattern: LLM-accessible tools with JSON schemas,
execution logic, and a deterministic mock for testing without external APIs.
"""

from __future__ import annotations

import json
import math
from typing import Any, Protocol

from orchestrator.state import ToolCall


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "web_search",
        "description": "Search the web for recent information about a topic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {
                    "type": "integer",
                    "description": "Max results to return",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "retrieve_docs",
        "description": "Retrieve relevant documents from the knowledge base via semantic search.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Retrieval query"},
                "top_k": {
                    "type": "integer",
                    "description": "Number of documents to retrieve",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculate",
        "description": "Evaluate a basic arithmetic expression (+, -, *, /, %, parentheses).",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression to evaluate",
                },
            },
            "required": ["expression"],
        },
    },
    {
        "name": "summarize",
        "description": "Condense a long text to a maximum word count.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to summarize"},
                "max_words": {
                    "type": "integer",
                    "description": "Maximum words in output",
                    "default": 100,
                },
            },
            "required": ["text"],
        },
    },
]

_MOCK_SEARCH_RESULTS = [
    {
        "title": "Multi-Agent AI Systems in Production",
        "snippet": (
            "LangGraph enables stateful multi-agent orchestration with conditional "
            "routing and parallel execution. Production deployments show 92-96% "
            "coordination efficiency."
        ),
        "url": "https://example.com/multi-agent",
    },
    {
        "title": "Tool Use in LLM Agents",
        "snippet": (
            "Augmenting LLMs with tools like search and retrieval increases factual "
            "accuracy by up to 40% on knowledge-intensive tasks."
        ),
        "url": "https://example.com/tool-use",
    },
    {
        "title": "RAG Architecture Patterns",
        "snippet": (
            "Retrieval-augmented generation combines vector search with language model "
            "synthesis for grounded, verifiable outputs."
        ),
        "url": "https://example.com/rag",
    },
]


class ToolProvider(Protocol):
    """Protocol for tool execution backends."""

    async def execute(self, tool_name: str, inputs: dict[str, Any]) -> ToolCall: ...

    def get_definitions(self) -> list[dict[str, Any]]: ...


class MockToolProvider:
    """Deterministic tool provider for testing and demos (no external APIs needed)."""

    async def execute(self, tool_name: str, inputs: dict[str, Any]) -> ToolCall:
        if tool_name == "web_search":
            limit = inputs.get("max_results", 3)
            output = json.dumps(_MOCK_SEARCH_RESULTS[:limit])
            return ToolCall(tool_name=tool_name, input=inputs, output=output, error="")

        if tool_name == "retrieve_docs":
            query = inputs.get("query", "")
            top_k = inputs.get("top_k", 3)
            docs = [
                {
                    "doc_id": f"doc_{i}",
                    "content": f"Knowledge base passage about {query!r} (rank {i + 1})",
                    "score": round(0.95 - i * 0.08, 2),
                }
                for i in range(top_k)
            ]
            return ToolCall(
                tool_name=tool_name, input=inputs, output=json.dumps(docs), error=""
            )

        if tool_name == "calculate":
            result = self._safe_eval(inputs.get("expression", ""))
            return ToolCall(
                tool_name=tool_name, input=inputs, output=result, error=""
            )

        if tool_name == "summarize":
            text = inputs.get("text", "")
            limit = inputs.get("max_words", 100)
            words = text.split()
            summary = " ".join(words[:limit]) + ("..." if len(words) > limit else "")
            return ToolCall(
                tool_name=tool_name, input=inputs, output=summary, error=""
            )

        return ToolCall(
            tool_name=tool_name,
            input=inputs,
            output="",
            error=f"Unknown tool: {tool_name!r}",
        )

    def get_definitions(self) -> list[dict[str, Any]]:
        return TOOL_DEFINITIONS

    @staticmethod
    def _safe_eval(expression: str) -> str:
        allowed_chars = set("0123456789+-*/()., %")
        if not all(c in allowed_chars for c in expression):
            return "Error: only basic arithmetic operators are allowed"
        try:
            result = eval(expression, {"__builtins__": {}}, {"math": math})  # noqa: S307
            return str(result)
        except Exception as exc:
            return f"Error: {exc}"
