# Multi-Agent Demo

## Stack
LangGraph | Streamlit | Anthropic (claude-haiku-4-5-20251001) | Python

## Architecture
LangGraph ContentPipeline with 4 nodes: Research → Draft → Review → Publish. MeshCoordinator for multi-agent orchestration. MockLLM works without API key; ClaudeLLM used when ANTHROPIC_API_KEY set.
- `demo/app.py` — Streamlit UI entry point
- `agents/` — LangGraph node definitions
- `pipeline/` — ContentPipeline and MeshCoordinator

## Deploy
Streamlit Cloud — https://multi-agent-demo-xjvogxpydrv6cfnxvqftpx.streamlit.app
Repo file: `demo/app.py`. Needs `ANTHROPIC_API_KEY` in Streamlit secrets.

## Test
```pytest tests/  # 33 tests```

## Key Env
ANTHROPIC_API_KEY
