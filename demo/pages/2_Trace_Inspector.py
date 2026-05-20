"""Streamlit page: Trace Inspector.

Loads pre-recorded agent runs from data/sample_traces.json and renders a
per-span timeline so reviewers can audit what each agent did, what tool
it called, how long it took, and what it cost.

This MVP reads from a static JSON file. Next iteration wires the same UI
to a live SQLite/Langfuse backend so every demo run produces a fresh trace.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from demo.theme import apply_theme

# Sample trace file lives at repo_root/data/sample_traces.json.
# This file is demo/pages/2_Trace_Inspector.py, so parents[2] is repo root.
TRACES_PATH = Path(__file__).resolve().parents[2] / "data" / "sample_traces.json"


@st.cache_data
def load_traces(path: Path) -> dict:
    """Read the trace JSON and return the parsed dict."""
    with open(path) as fh:
        return json.load(fh)


def render_span(span: dict) -> None:
    """Render one span as an expandable card."""
    status = span.get("status", "ok")
    agent = span.get("agent_name", "?")
    tool = span.get("tool_called") or "(no tool)"
    latency = span.get("latency_ms", 0)
    tokens = span.get("tokens", 0)
    cost = span.get("cost_estimate_usd", 0.0)

    status_icon = {"ok": "OK", "error": "ERR", "no_results": "EMPTY"}.get(
        status, status.upper()
    )

    header = (
        f"[{status_icon}] {agent}  |  tool={tool}  |  "
        f"{latency} ms  |  {tokens} tok  |  ${cost:.5f}"
    )

    with st.expander(header, expanded=False):
        st.markdown("**Input**")
        st.code(span.get("input_snippet", ""), language="text")
        st.markdown("**Output**")
        st.code(span.get("output_snippet", ""), language="text")
        cols = st.columns(4)
        cols[0].metric("Latency", f"{latency} ms")
        cols[1].metric("Tokens", f"{tokens:,}")
        cols[2].metric("Cost", f"${cost:.5f}")
        cols[3].metric("Status", status_icon)


def render_run_summary(run: dict) -> None:
    """Render top-level metrics for the selected run."""
    cols = st.columns(5)
    cols[0].metric("Outcome", run.get("outcome", "?"))
    cols[1].metric("Total latency", f"{run.get('total_latency_ms', 0)} ms")
    cols[2].metric("Total tokens", f"{run.get('total_tokens', 0):,}")
    cols[3].metric("Total cost", f"${run.get('total_cost_usd', 0.0):.5f}")
    cols[4].metric("Revisions", run.get("revision_count", 0))


def main() -> None:
    """Streamlit entrypoint."""
    apply_theme()
    st.set_page_config(page_title="Trace Inspector", layout="wide")

    st.title("Trace Inspector")
    st.caption(
        "Per-span timeline of recent agent runs. Click a run, then expand any "
        "span to see input, output, latency, and cost."
    )

    if not TRACES_PATH.exists():
        st.error(f"No trace file found at {TRACES_PATH}")
        st.stop()

    traces = load_traces(TRACES_PATH)
    runs = traces.get("runs", [])

    if not runs:
        st.warning("Trace file loaded but contains no runs.")
        st.stop()

    # Run picker. Show outcome + topic so the choice is informative.
    run_options = {
        f"{r['run_id']}  ({r['outcome']})  {r['topic'][:60]}": r for r in runs
    }
    selected_label = st.selectbox("Pick a run", list(run_options.keys()))
    selected_run = run_options[selected_label]

    st.subheader(selected_run["topic"])
    st.caption(f"run_id: `{selected_run['run_id']}`  |  started: {selected_run.get('started_at', '?')}")

    render_run_summary(selected_run)

    st.markdown("---")
    st.markdown("### Span timeline")
    st.caption(
        f"{len(selected_run.get('spans', []))} spans ordered by start time. "
        "Each entry shows agent, tool, latency, tokens, and cost."
    )

    for span in selected_run.get("spans", []):
        render_span(span)

    st.markdown("---")
    with st.expander("Raw run JSON", expanded=False):
        st.json(selected_run)


if __name__ == "__main__":
    main()
else:
    # Streamlit imports this module on page load, so we run main() at import.
    main()
