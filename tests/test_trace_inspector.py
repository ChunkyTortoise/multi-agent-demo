"""Tests for the Trace Inspector sample data.

These tests guarantee the sample trace JSON parses and conforms to the
schema the Streamlit page expects. If a future contributor edits the
JSON and breaks the shape, CI catches it before the demo loads.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACES_PATH = REPO_ROOT / "data" / "sample_traces.json"

REQUIRED_RUN_FIELDS = {
    "run_id",
    "topic",
    "outcome",
    "total_latency_ms",
    "total_tokens",
    "total_cost_usd",
    "revision_count",
    "spans",
}
REQUIRED_SPAN_FIELDS = {
    "span_id",
    "agent_name",
    "input_snippet",
    "output_snippet",
    "latency_ms",
    "tokens",
    "cost_estimate_usd",
    "status",
}
VALID_OUTCOMES = {"success", "success_after_revision", "failure"}
VALID_STATUSES = {"ok", "error", "no_results"}


@pytest.fixture(scope="module")
def traces() -> dict:
    """Load and parse the sample trace JSON once per test module."""
    assert TRACES_PATH.exists(), f"Sample trace file missing at {TRACES_PATH}"
    with open(TRACES_PATH) as fh:
        return json.load(fh)


def test_traces_parse(traces: dict) -> None:
    """The JSON must parse and expose a top-level runs list."""
    assert "runs" in traces, "Top-level 'runs' key missing"
    assert isinstance(traces["runs"], list), "'runs' must be a list"
    assert len(traces["runs"]) >= 3, "Need at least 3 sample runs for the demo"


def test_runs_have_required_fields(traces: dict) -> None:
    """Every run must carry the fields the Streamlit page reads."""
    for run in traces["runs"]:
        missing = REQUIRED_RUN_FIELDS - set(run.keys())
        assert not missing, f"Run {run.get('run_id')} missing fields: {missing}"
        assert run["outcome"] in VALID_OUTCOMES, (
            f"Run {run['run_id']} has unknown outcome {run['outcome']}"
        )
        assert isinstance(run["spans"], list) and run["spans"], (
            f"Run {run['run_id']} must have at least one span"
        )


def test_spans_have_required_fields(traces: dict) -> None:
    """Every span across every run must conform to the inspector schema."""
    for run in traces["runs"]:
        for span in run["spans"]:
            missing = REQUIRED_SPAN_FIELDS - set(span.keys())
            assert not missing, (
                f"Span {span.get('span_id')} in run {run['run_id']} missing: {missing}"
            )
            assert span["status"] in VALID_STATUSES, (
                f"Span {span['span_id']} has unknown status {span['status']}"
            )
            assert isinstance(span["latency_ms"], (int, float))
            assert isinstance(span["tokens"], (int, float))
            assert isinstance(span["cost_estimate_usd"], (int, float))
            assert span["latency_ms"] >= 0
            assert span["tokens"] >= 0
            assert span["cost_estimate_usd"] >= 0


def test_run_totals_match_span_sums(traces: dict) -> None:
    """Run-level totals must match the sum of their spans within tight tolerance."""
    for run in traces["runs"]:
        latency_sum = sum(s["latency_ms"] for s in run["spans"])
        token_sum = sum(s["tokens"] for s in run["spans"])
        cost_sum = sum(s["cost_estimate_usd"] for s in run["spans"])
        assert run["total_latency_ms"] == latency_sum, (
            f"Run {run['run_id']} latency mismatch: "
            f"declared {run['total_latency_ms']} vs sum {latency_sum}"
        )
        assert run["total_tokens"] == token_sum, (
            f"Run {run['run_id']} token mismatch: "
            f"declared {run['total_tokens']} vs sum {token_sum}"
        )
        assert abs(run["total_cost_usd"] - cost_sum) < 1e-5, (
            f"Run {run['run_id']} cost mismatch: "
            f"declared {run['total_cost_usd']} vs sum {cost_sum}"
        )


def test_includes_one_of_each_outcome(traces: dict) -> None:
    """Demo needs visible variety: success, retry, and failure represented."""
    outcomes = {r["outcome"] for r in traces["runs"]}
    assert "success" in outcomes
    assert "success_after_revision" in outcomes
    assert "failure" in outcomes
