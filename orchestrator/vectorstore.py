"""Vector store abstraction for semantic document retrieval.

Provides two implementations:
- ``MockVectorStore``: in-memory cosine similarity over a small curated corpus.
  No external dependencies — used by default in tests and demos.
- ``ChromaVectorStore``: persistent ChromaDB + sentence-transformers backend.
  Install optional deps first: ``pip install -e ".[rag]"``

The ``retrieve_docs`` tool in MockToolProvider can be swapped to use either
backend by passing a VectorStore to the tool provider.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Protocol


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class VectorStore(Protocol):
    """Minimal interface for vector-based document retrieval."""

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        """Add documents to the store.  Each dict must have ``id`` and ``text``."""
        ...

    def query(self, query_text: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Return top-k documents ranked by similarity to query_text."""
        ...

    def count(self) -> int:
        """Return the number of documents in the store."""
        ...


# ---------------------------------------------------------------------------
# MockVectorStore — in-memory, no deps
# ---------------------------------------------------------------------------

# Small curated corpus seeded into every MockVectorStore
_DEFAULT_CORPUS: list[dict[str, str]] = [
    {
        "id": "doc_001",
        "text": (
            "LangGraph enables stateful multi-agent orchestration with conditional "
            "routing. It uses a StateGraph that compiles to an executable workflow "
            "supporting both sequential and parallel agent execution."
        ),
    },
    {
        "id": "doc_002",
        "text": (
            "Tool-augmented LLM agents call external tools (web search, code execution, "
            "database queries) to ground responses in verifiable information. "
            "Tool definitions use JSON Schema for input validation."
        ),
    },
    {
        "id": "doc_003",
        "text": (
            "Retrieval-augmented generation (RAG) combines vector similarity search "
            "with large language model generation. Documents are embedded with "
            "sentence-transformers and indexed for fast approximate nearest-neighbour lookup."
        ),
    },
    {
        "id": "doc_004",
        "text": (
            "Production LLMOps requires observability: distributed tracing with "
            "OpenTelemetry, metrics with Prometheus, and experiment tracking with "
            "LangSmith or Langfuse for prompt regression testing."
        ),
    },
    {
        "id": "doc_005",
        "text": (
            "Fine-tuning with QLoRA uses 4-bit quantization and low-rank adapter "
            "weights (LoRA) to reduce GPU memory requirements while preserving "
            "model quality. DPO further aligns the model via preference pairs."
        ),
    },
]


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer used for TF-IDF approximation."""
    return [w.strip(".,!?;:\"'()[]") for w in text.lower().split() if len(w) > 1]


def _tf_idf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    counts: dict[str, int] = defaultdict(int)
    for t in tokens:
        counts[t] += 1
    n = len(tokens) or 1
    return {t: (c / n) * idf.get(t, 1.0) for t, c in counts.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    shared = set(a) & set(b)
    if not shared:
        return 0.0
    dot = sum(a[k] * b[k] for k in shared)
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class MockVectorStore:
    """In-memory TF-IDF vector store seeded with a curated AI/agents corpus.

    No external dependencies. Used for tests and offline demos.
    """

    def __init__(self, seed_corpus: bool = True) -> None:
        self._docs: list[dict[str, Any]] = []
        self._idf: dict[str, float] = {}
        if seed_corpus:
            self.add_documents(_DEFAULT_CORPUS)

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        """Add documents and recompute IDF weights."""
        self._docs.extend(documents)
        self._recompute_idf()

    def query(self, query_text: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Return top-k documents by TF-IDF cosine similarity."""
        if not self._docs:
            return []
        q_vec = _tf_idf_vector(_tokenize(query_text), self._idf)
        scored = [
            (
                _cosine(q_vec, _tf_idf_vector(_tokenize(d["text"]), self._idf)),
                d,
            )
            for d in self._docs
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"doc_id": d["id"], "content": d["text"], "score": round(score, 4)}
            for score, d in scored[:top_k]
        ]

    def count(self) -> int:
        return len(self._docs)

    def _recompute_idf(self) -> None:
        n = len(self._docs)
        df: dict[str, int] = defaultdict(int)
        for doc in self._docs:
            for token in set(_tokenize(doc["text"])):
                df[token] += 1
        self._idf = {t: math.log((n + 1) / (cnt + 1)) + 1.0 for t, cnt in df.items()}


# ---------------------------------------------------------------------------
# ChromaVectorStore — real embeddings via ChromaDB + sentence-transformers
# ---------------------------------------------------------------------------


class ChromaVectorStore:
    """ChromaDB-backed vector store using sentence-transformers embeddings.

    Requires: ``pip install -e ".[rag]"``

    Example::

        store = ChromaVectorStore()
        store.add_documents([{"id": "1", "text": "Hello world"}])
        results = store.query("greeting", top_k=1)
    """

    def __init__(
        self,
        collection_name: str = "multi_agent_demo",
        embedding_model: str = "all-MiniLM-L6-v2",
        seed_corpus: bool = True,
    ) -> None:
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "ChromaVectorStore requires optional deps: pip install -e '.[rag]'"
            ) from exc

        self._model = SentenceTransformer(embedding_model)
        self._client = chromadb.Client()
        self._collection = self._client.get_or_create_collection(collection_name)
        if seed_corpus:
            self.add_documents(_DEFAULT_CORPUS)

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        texts = [d["text"] for d in documents]
        ids = [d["id"] for d in documents]
        embeddings = self._model.encode(texts).tolist()
        self._collection.add(documents=texts, embeddings=embeddings, ids=ids)

    def query(self, query_text: str, top_k: int = 3) -> list[dict[str, Any]]:
        embedding = self._model.encode([query_text]).tolist()
        results = self._collection.query(
            query_embeddings=embedding,
            n_results=min(top_k, self._collection.count()),
        )
        docs = results["documents"][0]
        ids = results["ids"][0]
        distances = results["distances"][0]
        return [
            {"doc_id": ids[i], "content": docs[i], "score": round(1 - distances[i], 4)}
            for i in range(len(docs))
        ]

    def count(self) -> int:
        return self._collection.count()
