"""Streamlit app: Live Multi-Agent Orchestrator demo."""

from __future__ import annotations

import asyncio
import time

import streamlit as st

from demo.mock_llm import get_llm
from demo.theme import apply_theme
from mesh.coordinator import MeshCoordinator
from mesh.registry import AgentStatus
from orchestrator.graph import AGENT_SEQUENCE, ContentPipeline

AGENT_LABELS = {
    "researcher": "Research",
    "drafter": "Draft",
    "reviewer": "Review",
    "publisher": "Publish",
}

STATUS_COLORS = {
    AgentStatus.IDLE: "gray",
    AgentStatus.RUNNING: "blue",
    AgentStatus.DONE: "green",
    AgentStatus.ERROR: "red",
}

STATUS_EMOJI = {
    AgentStatus.IDLE: "",
    AgentStatus.RUNNING: "...",
    AgentStatus.DONE: " [done]",
    AgentStatus.ERROR: " [error]",
}

ARCHITECTURE_DIAGRAM = """
```
                    +-------------------+
                    |   Content Topic   |
                    +--------+----------+
                             |
                    +--------v----------+
                    |   RESEARCHER      |
                    |   (Gather facts)  |
                    +--------+----------+
                             |
                    +--------v----------+
               +--->|   DRAFTER         |
               |    |   (Write article) |
               |    +--------+----------+
               |             |
               |    +--------v----------+
               |    |   REVIEWER        |
               |    |   (Quality check) |
               |    +--------+----------+
               |             |
               |        score < 0.7
               |        & revisions < 2?
               |       /            \\
               +--- YES              NO ---+
                                           |
                                  +--------v----------+
                                  |   PUBLISHER       |
                                  |   (Format & ship) |
                                  +--------+----------+
                                           |
                                  +--------v----------+
                                  |   PUBLISHED       |
                                  +-------------------+

    Orchestration: LangGraph StateGraph (conditional routing)
    Coordination:  Mesh Coordinator (health, cost, routing)
    LLM Backend:   Claude Haiku (real) or MockLLM (demo)
```
"""


def render_pipeline_status(agents: list, current: str | None) -> None:
    """Render the visual pipeline with status boxes."""
    cols = st.columns(len(AGENT_SEQUENCE))
    for i, name in enumerate(AGENT_SEQUENCE):
        agent = next((a for a in agents if a.name == name), None)
        status = agent.status if agent else AgentStatus.IDLE
        label = AGENT_LABELS[name]
        suffix = STATUS_EMOJI[status]

        with cols[i]:
            if status == AgentStatus.DONE:
                st.markdown(
                    f'<div class="agent-card done"><p style="text-align:center;margin:0;font-weight:600;">{label}{suffix}</p></div>',
                    unsafe_allow_html=True,
                )
            elif status == AgentStatus.RUNNING:
                st.markdown(
                    f'<div class="agent-card running"><p style="text-align:center;margin:0;font-weight:600;">{label}{suffix}</p></div>',
                    unsafe_allow_html=True,
                )
            elif status == AgentStatus.ERROR:
                st.markdown(
                    f'<div class="agent-card" style="border-color:rgba(239,68,68,0.6);background:rgba(239,68,68,0.06);"><p style="text-align:center;margin:0;color:#ef4444;">{label} [error]</p></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="agent-card"><p style="text-align:center;margin:0;color:#8B949E;">{label}</p></div>',
                    unsafe_allow_html=True,
                )


def render_agent_table(agents: list) -> None:
    """Render agent status table."""
    rows = []
    for a in agents:
        rows.append(
            {
                "Agent": a.name.title(),
                "Status": a.status.value.upper(),
                "Tokens": a.metrics.tokens_used,
                "Latency": f"{a.metrics.avg_latency_s:.2f}s" if a.metrics.completed_tasks else "-",
            }
        )
    if rows:
        st.table(rows)


def render_cost_sidebar(coordinator: MeshCoordinator) -> None:
    """Render cost tracker in sidebar."""
    status = coordinator.get_status()
    st.sidebar.markdown("## Cost Tracker")
    st.sidebar.metric("Total Tokens", f"{status['tokens']['total']:,}")
    st.sidebar.metric("Estimated Cost", f"${status['cost']['total']:.6f}")
    st.sidebar.metric(
        "Budget",
        f"${status['cost']['budget_limit']:.2f}",
        delta="OK" if status["cost"]["within_budget"] else "OVER",
        delta_color="normal" if status["cost"]["within_budget"] else "inverse",
    )

    st.sidebar.markdown("### Per-Agent Tokens")
    for name, tokens in status["tokens"]["by_agent"].items():
        st.sidebar.text(f"  {name}: {tokens:,}")


async def run_pipeline_with_tracking(
    topic: str,
    coordinator: MeshCoordinator,
    pipeline_placeholder,
    table_placeholder,
    output_placeholder,
) -> None:
    """Run pipeline with real-time UI updates via mesh coordinator."""
    llm = get_llm()

    for agent_name in AGENT_SEQUENCE:
        # Mark agent as running
        coordinator.start_agent(agent_name)

        # Update pipeline visual
        with pipeline_placeholder.container():
            render_pipeline_status(
                coordinator.registry.all_agents(), agent_name
            )

        # Update agent table
        with table_placeholder.container():
            render_agent_table(coordinator.registry.all_agents())

        # Execute agent via LLM
        start = time.time()
        try:
            from orchestrator.nodes import _build_prompt

            # Build prompt with prior output
            prior = ""
            if agent_name == "drafter":
                r = coordinator.registry.get("researcher")
                prior = r.output if r else ""
            elif agent_name == "reviewer":
                d = coordinator.registry.get("drafter")
                prior = d.output if d else ""
            elif agent_name == "publisher":
                r = coordinator.registry.get("reviewer")
                prior = r.output if r else ""

            prompt = _build_prompt(agent_name, topic, prior)
            result = await llm.generate(prompt, agent_name)
            latency = time.time() - start

            coordinator.complete_agent(
                agent_name,
                tokens=result.get("tokens_used", 0),
                latency_s=latency,
                output=result.get("content", ""),
            )
        except Exception as exc:
            coordinator.fail_agent(agent_name, str(exc))

        # Update visuals after completion
        with pipeline_placeholder.container():
            render_pipeline_status(
                coordinator.registry.all_agents(), agent_name
            )
        with table_placeholder.container():
            render_agent_table(coordinator.registry.all_agents())

    # Show final output
    publisher = coordinator.registry.get("publisher")
    if publisher and publisher.status == AgentStatus.DONE:
        with output_placeholder.container():
            st.markdown("---")
            st.markdown("### Final Published Output")
            st.markdown(
                f'<div class="output-box">{publisher.output}</div>',
                unsafe_allow_html=True,
            )


def render_review_quality(result: dict) -> None:
    """Display review score and revision count after pipeline run."""
    score = result.get("review_score", 0.0)
    revisions = result.get("revision_count", 0)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Review Score", f"{score:.2f}")
        st.progress(min(score, 1.0))
    with col2:
        st.metric("Revisions", revisions)
        if revisions == 0:
            st.info("Passed review on first attempt")
        else:
            st.warning(f"Revised {revisions} time{'s' if revisions != 1 else ''} before passing")


def main() -> None:
    st.set_page_config(
        page_title="Multi-Agent Orchestrator",
        page_icon="",
        layout="wide",
    )
    apply_theme()

    st.title("Live Multi-Agent Orchestrator")
    st.caption("LangGraph + Mesh Coordinator")

    # Show LLM mode
    import os
    if os.environ.get("ANTHROPIC_API_KEY"):
        st.sidebar.success("Mode: Claude Haiku (real)")
    else:
        st.sidebar.info("Mode: MockLLM (demo)")
        st.sidebar.caption("Set ANTHROPIC_API_KEY for real Claude output")

    # Input
    topic = st.text_input(
        "Topic to research and publish",
        placeholder="e.g., The future of multi-agent AI systems",
    )

    run_button = st.button("Run Pipeline", type="primary", disabled=not topic)

    # Placeholders for real-time updates
    pipeline_placeholder = st.empty()
    table_placeholder = st.empty()
    output_placeholder = st.empty()

    # Show skeleton while waiting for first run
    if not run_button:
        _skeleton = '<div class="skeleton"></div>' * len(AGENT_SEQUENCE)
        pipeline_placeholder.markdown(
            f'<div style="display:grid;grid-template-columns:repeat({len(AGENT_SEQUENCE)},1fr);gap:0.5rem;">{_skeleton}</div>',
            unsafe_allow_html=True,
        )

    # Architecture expander
    with st.expander("Architecture"):
        st.markdown(ARCHITECTURE_DIAGRAM)

    if run_button and topic:
        # Initialize coordinator
        coordinator = MeshCoordinator(budget_limit=1.0)
        coordinator.register_agents(list(AGENT_SEQUENCE))

        # Initial render
        with pipeline_placeholder.container():
            render_pipeline_status(coordinator.registry.all_agents(), None)

        # Render sidebar
        render_cost_sidebar(coordinator)

        # Run pipeline with visual tracking
        asyncio.run(
            run_pipeline_with_tracking(
                topic,
                coordinator,
                pipeline_placeholder,
                table_placeholder,
                output_placeholder,
            )
        )

        # Run LangGraph pipeline to get review quality metrics
        llm = get_llm()
        pipeline = ContentPipeline(llm=llm)
        result = asyncio.run(pipeline.run(topic))

        # Final sidebar update
        render_cost_sidebar(coordinator)

        # Show review quality metrics
        render_review_quality(result)

        st.success("Pipeline complete!")


if __name__ == "__main__":
    main()
