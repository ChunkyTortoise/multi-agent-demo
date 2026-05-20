# Multi-Agent Demo

## Stack
LangGraph | Streamlit | Anthropic (claude-haiku-4-5-20251001) | Python

## Architecture
LangGraph ContentPipeline with 7 nodes: Planner → Sub-Research×N → Aggregator → Research → Draft → Review → Publish. Planning and parallel execution are opt-in feature flags. MockLLM works without API key; ClaudeLLM used when ANTHROPIC_API_KEY set.
- `demo/app.py` — Streamlit UI with sidebar toggles for planner/parallel modes
- `orchestrator/` — ContentPipeline, nodes, planner, tools, vectorstore, state
- `mesh/` -- Scaffold MeshCoordinator + registry, NOT wired into the graph (see README "Known scope gaps")

## Deploy
Streamlit Cloud — https://multi-agent-demo-xjvogxpydrv6cfnxvqftpx.streamlit.app
Repo file: `demo/app.py`. Needs `ANTHROPIC_API_KEY` in Streamlit secrets.

## Test
```pytest tests/  # 89 tests```

## Key Env
ANTHROPIC_API_KEY
