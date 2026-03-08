![Tests](https://img.shields.io/badge/tests-28%20passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41-red)
![CI](https://github.com/ChunkyTortoise/multi-agent-demo/actions/workflows/ci.yml/badge.svg)

# Multi-Agent Orchestrator Demo

Live demo of a multi-agent content pipeline using LangGraph state machines and a mesh coordinator for health/cost tracking.

## Architecture

```
Topic Input
    |
    v
[Researcher] --> [Drafter] --> [Reviewer] --> [Publisher]
    |                |              |              |
    +--- Mesh Coordinator (health, cost, routing) -+
```

**4 sequential agents**: Researcher, Drafter, Reviewer, Publisher
- **LangGraph StateGraph** manages the workflow (modeled after EnterpriseHub's `LeadQualificationOrchestrator`)
- **Mesh Coordinator** handles agent registration, health tracking, cost monitoring (simplified from EnterpriseHub's `AgentMeshCoordinator`)
- **MockLLM** generates realistic outputs without API keys
- **ClaudeLLM** uses real Claude Haiku when `ANTHROPIC_API_KEY` is set

## Quick Start

```bash
pip install -e ".[dev]"

# Run tests
pytest tests/ -x -q

# Launch Streamlit demo
streamlit run demo/app.py

# With real Claude (optional)
ANTHROPIC_API_KEY=sk-... streamlit run demo/app.py
```

## Project Structure

```
orchestrator/
    graph.py       # LangGraph state machine
    nodes.py       # Agent node functions
    state.py       # TypedDict state definition
mesh/
    coordinator.py # Mesh coordinator (health, cost, routing)
    registry.py    # Agent registry and metrics
demo/
    app.py         # Streamlit UI
    mock_llm.py    # Mock + real LLM providers
tests/
    test_graph.py         # Pipeline state machine tests
    test_coordinator.py   # Coordinator and registry tests
```

## Key Design Decisions

1. **Decoupled from EnterpriseHub**: No GHL, MCP, Redis, or PostgreSQL dependencies
2. **Same patterns**: TypedDict state, StateGraph, conditional edges, ainvoke()
3. **Dual LLM mode**: MockLLM for demos, ClaudeLLM for real output
4. **Cost tracking**: Blended Haiku rate ($0.50/1M tokens)
