"""Streamlit app: Live Multi-Agent Orchestrator demo."""

from __future__ import annotations

import asyncio
import json
import os

import streamlit as st

from demo.mock_llm import get_llm
from demo.theme import apply_theme
from orchestrator.graph import ContentPipeline
from orchestrator.tools import MockToolProvider

AGENT_LABELS = {
    "researcher": "Research",
    "drafter": "Draft",
    "reviewer": "Review",
    "publisher": "Publish",
    "planner": "Plan",
    "sub_researcher": "Sub-Research",
    "aggregator": "Aggregate",
}

# Nodes that get the "planner" card colour (amber)
_PLANNER_NODES = {"planner"}
# Nodes that get the "parallel" card colour (purple)
_PARALLEL_NODES = {"sub_researcher", "aggregator"}

ARCHITECTURE_DIAGRAM = """
```
                    +-------------------+
                    |   Content Topic   |
                    +--------+----------+
                             |
              [use_planner & complex topic?]
                /                         \\
              YES                          NO
               |                           |
     +---------v----------+                |
     |      PLANNER        |                |
     | (Decompose topic)   |                |
     +---------+-----------+                |
               |                           |
   [use_parallel & N > 1 sub-tasks?]       |
       /                  \\                |
     YES                   NO              |
      |                     |              |
+-----v------+      +-------v--------------v----+
| SUB-RESEARCH|      |        RESEARCHER          |
|  (x N)      |      |  + web_search()            |
+-----+-------+      |  + retrieve_docs()         |
      |              +-------------+--------------+
+-----v------+                     |
| AGGREGATOR  |                     |
+-----+-------+                     |
      |                             |
      +----------+------------------+
                 |
        +--------v----------+
        |      DRAFTER       |
        +--------+----------+
                 |
        +--------v----------+
        |     REVIEWER       |<---+
        +--------+----------+    |
                 |          score < 0.7
            [quality?]    & revisions < 2
            /        \\        |
          PASS      FAIL------+
            |
   +--------v----------+
   |    PUBLISHER       |
   +-------------------+

Orchestration: LangGraph StateGraph (conditional routing + Send() fan-out)
Tools:         web_search · retrieve_docs · calculate · summarize
Vector store:  MockVectorStore (TF-IDF) or ChromaVectorStore (optional)
LLM:           Claude Haiku (real) or MockLLM (demo)
```
"""


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def get_active_sequence(use_planner: bool, use_parallel: bool) -> list[str]:
    """Return the ordered list of node names for the current pipeline mode."""
    seq: list[str] = []
    if use_planner:
        seq.append("planner")
    if use_parallel:
        seq.extend(["sub_researcher", "aggregator"])
    else:
        seq.append("researcher")
    seq.extend(["drafter", "reviewer", "publisher"])
    return seq


def _card_html(label: str, status: str, node_name: str) -> str:
    """Build an agent-card div with appropriate CSS classes."""
    extra = ""
    if node_name in _PLANNER_NODES:
        extra = " planner"
    elif node_name in _PARALLEL_NODES:
        extra = " parallel"

    if status == "done":
        css = f"agent-card done{extra}"
        text = f"{label} ✓"
        color = ""
    elif status == "running":
        css = f"agent-card running{extra}"
        text = f"{label}…"
        color = ""
    elif status == "error":
        css = "agent-card"
        text = f"{label} ✗"
        color = ' style="color:#ef4444;"'
    else:
        css = f"agent-card{extra}"
        text = label
        color = ' style="color:#8B949E;"'

    return (
        f'<div class="{css}">'
        f'<p{color} style="text-align:center;margin:0;font-weight:600;">{text}</p>'
        f"</div>"
    )


def render_pipeline_status(active_sequence: list[str], node_status: dict[str, str]) -> None:
    """Render the visual pipeline with dynamic column count."""
    cols = st.columns(len(active_sequence))
    for i, name in enumerate(active_sequence):
        label = AGENT_LABELS.get(name, name.title())
        status = node_status.get(name, "idle")
        with cols[i]:
            st.markdown(_card_html(label, status, name), unsafe_allow_html=True)


def render_tool_calls(tool_calls: list[dict]) -> None:
    if not tool_calls:
        return
    with st.expander(f"Tool Calls ({len(tool_calls)})", expanded=False):
        for i, tc in enumerate(tool_calls):
            tool_name = tc.get("tool_name", "unknown")
            st.markdown(f"**{i + 1}. `{tool_name}`**")
            col1, col2 = st.columns(2)
            with col1:
                st.caption("Input")
                raw_input = tc.get("input", {})
                if isinstance(raw_input, str):
                    try:
                        raw_input = json.loads(raw_input)
                    except (json.JSONDecodeError, TypeError):
                        pass
                st.json(raw_input)
            with col2:
                st.caption("Output")
                output = tc.get("output", "")
                if isinstance(output, str):
                    try:
                        parsed = json.loads(output)
                        st.json(parsed)
                    except (json.JSONDecodeError, TypeError):
                        st.code(output[:500])
                else:
                    st.json(output)
            if tc.get("error"):
                st.error(tc["error"])
            if i < len(tool_calls) - 1:
                st.markdown("---")


def render_plan(plan: list[str]) -> None:
    if not plan:
        return
    with st.expander(f"Research Plan ({len(plan)} sub-tasks)", expanded=True):
        for i, task in enumerate(plan):
            st.markdown(f"{i + 1}. {task}")


def render_parallel_results(parallel_results: list[dict]) -> None:
    if not parallel_results:
        return
    with st.expander(
        f"Parallel Research ({len(parallel_results)} sub-researchers)", expanded=False
    ):
        for r in parallel_results:
            st.markdown(f"**Angle:** {r.get('task', 'N/A')}")
            st.caption(f"Tokens: {r.get('tokens_used', 0):,}")
            content = r.get("content", "")
            preview = content[:400] + ("…" if len(content) > 400 else "")
            st.markdown(preview)
            st.markdown("---")


def render_review_quality(review_score: float, revision_count: int) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Review Score", f"{review_score:.2f}")
        st.progress(min(review_score, 1.0))
    with col2:
        st.metric("Revisions", revision_count)
        if revision_count == 0:
            st.info("Passed review on first attempt")
        else:
            st.warning(f"Revised {revision_count} time{'s' if revision_count != 1 else ''} before passing")


def render_cost_sidebar(total_tokens: int) -> None:
    est_cost = total_tokens * 0.5 / 1_000_000
    st.sidebar.markdown("## Cost Tracker")
    st.sidebar.metric("Total Tokens", f"{total_tokens:,}")
    st.sidebar.metric("Estimated Cost", f"${est_cost:.6f}")


# ---------------------------------------------------------------------------
# Streaming pipeline runner
# ---------------------------------------------------------------------------


async def run_pipeline_streaming(
    topic: str,
    use_planner: bool,
    use_parallel: bool,
    pipeline_placeholder,
    details_placeholder,
    output_placeholder,
) -> dict:
    """Run the pipeline and drive UI updates from astream events."""
    llm = get_llm()
    tool_provider = MockToolProvider()
    pipeline = ContentPipeline(
        llm=llm,
        tool_provider=tool_provider,
        use_planner=use_planner,
        use_parallel=use_parallel,
    )

    active_seq = get_active_sequence(use_planner, use_parallel)
    node_status: dict[str, str] = {name: "idle" for name in active_seq}

    # Accumulated state
    accumulated: dict = {
        "tool_calls": [],
        "plan": [],
        "parallel_results": [],
        "total_tokens": 0,
        "review_score": 0.0,
        "revision_count": 0,
        "publish_output": {},
        "error": "",
    }

    # Mark first node as running
    if active_seq:
        node_status[active_seq[0]] = "running"
    with pipeline_placeholder.container():
        render_pipeline_status(active_seq, node_status)

    async for node_name, delta in pipeline.astream(topic):
        # Mark completed node done
        display_name = node_name
        if display_name in node_status:
            node_status[display_name] = "done"

        # Accumulate state fields
        if "tool_calls" in delta and delta["tool_calls"]:
            accumulated["tool_calls"].extend(delta["tool_calls"])
        if "plan" in delta and delta["plan"]:
            accumulated["plan"] = delta["plan"]
        if "parallel_results" in delta and delta["parallel_results"]:
            accumulated["parallel_results"].extend(delta["parallel_results"])
        if "total_tokens" in delta:
            accumulated["total_tokens"] = delta["total_tokens"]
        if "review_score" in delta:
            accumulated["review_score"] = delta["review_score"]
        if "revision_count" in delta:
            accumulated["revision_count"] = delta["revision_count"]
        if "publish_output" in delta and delta["publish_output"]:
            accumulated["publish_output"] = delta["publish_output"]
        if "error" in delta and delta["error"]:
            accumulated["error"] = delta["error"]

        # Mark next idle node as running
        try:
            idx = active_seq.index(display_name)
            if idx + 1 < len(active_seq):
                next_node = active_seq[idx + 1]
                if node_status.get(next_node) == "idle":
                    node_status[next_node] = "running"
        except ValueError:
            pass

        # Refresh pipeline cards
        with pipeline_placeholder.container():
            render_pipeline_status(active_seq, node_status)

        # Refresh details panel
        with details_placeholder.container():
            render_plan(accumulated["plan"])
            render_tool_calls(accumulated["tool_calls"])
            render_parallel_results(accumulated["parallel_results"])

    # Final output
    publish = accumulated.get("publish_output", {})
    content = publish.get("content", "")
    if content:
        with output_placeholder.container():
            st.markdown("---")
            st.markdown("### Final Published Output")
            st.markdown(
                f'<div class="output-box">{content}</div>',
                unsafe_allow_html=True,
            )

    return accumulated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="Multi-Agent Orchestrator",
        page_icon="",
        layout="wide",
    )
    apply_theme()

    st.title("Live Multi-Agent Orchestrator")
    st.caption("LangGraph · Tool Use · Planning · Parallel Execution · Vector Store")

    # LLM mode indicator
    if os.environ.get("ANTHROPIC_API_KEY"):
        st.sidebar.success("Mode: Claude Haiku (real)")
    else:
        st.sidebar.info("Mode: MockLLM (demo)")
        st.sidebar.caption("Set ANTHROPIC_API_KEY for real Claude output")

    # Pipeline option toggles
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Pipeline Options")
    use_planner = st.sidebar.checkbox(
        "Enable Planning",
        value=False,
        help="Decompose complex topics into sub-tasks before research",
    )
    use_parallel = st.sidebar.checkbox(
        "Enable Parallel Execution",
        value=False,
        help="Run sub-tasks in parallel via LangGraph Send() — auto-enables planning",
    )
    if use_parallel:
        use_planner = True

    # Topic input
    topic = st.text_input(
        "Topic to research and publish",
        placeholder="e.g., The future of multi-agent AI systems",
    )

    run_button = st.button("Run Pipeline", type="primary", disabled=not topic)

    # Placeholders
    pipeline_placeholder = st.empty()
    details_placeholder = st.empty()
    output_placeholder = st.empty()

    # Skeleton while idle
    active_seq = get_active_sequence(use_planner, use_parallel)
    if not run_button:
        _skeleton = '<div class="skeleton"></div>' * len(active_seq)
        pipeline_placeholder.markdown(
            f'<div style="display:grid;grid-template-columns:repeat({len(active_seq)},1fr);gap:0.5rem;">'
            f"{_skeleton}</div>",
            unsafe_allow_html=True,
        )

    # Architecture expander
    with st.expander("Architecture"):
        st.markdown(ARCHITECTURE_DIAGRAM)

    if run_button and topic:
        result = asyncio.run(
            run_pipeline_streaming(
                topic,
                use_planner,
                use_parallel,
                pipeline_placeholder,
                details_placeholder,
                output_placeholder,
            )
        )

        # Sidebar cost
        render_cost_sidebar(result.get("total_tokens", 0))

        # Review quality
        st.markdown("### Pipeline Quality")
        render_review_quality(
            result.get("review_score", 0.0),
            result.get("revision_count", 0),
        )

        if result.get("error"):
            st.error(f"Pipeline error: {result['error']}")
        else:
            st.success("Pipeline complete!")


if __name__ == "__main__":
    main()
