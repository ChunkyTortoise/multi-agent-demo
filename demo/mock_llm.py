"""Mock and real LLM providers for the demo.

MockLLM generates realistic-looking outputs without API keys.
ClaudeLLM uses real Claude claude-haiku-4-5-20251001 when ANTHROPIC_API_KEY is set.
"""

from __future__ import annotations

import asyncio
import os
import random
from typing import Optional

from orchestrator.state import AgentOutput

# Realistic mock outputs by agent role
_MOCK_OUTPUTS: dict[str, list[str]] = {
    "planner": [
        (
            "1. Research current market size and growth trends\n"
            "2. Analyze key technical developments and emerging frameworks\n"
            "3. Identify major industry players and competitive dynamics\n"
            "4. Examine real-world implementation challenges and ROI metrics"
        ),
    ],
    "researcher": [
        (
            "## Research Findings\n\n"
            "**Key Statistics:**\n"
            "- Market size grew 23% YoY to $4.2B in 2025\n"
            "- 67% of enterprises adopted AI-driven workflows\n"
            "- Average ROI of 340% reported by early adopters\n\n"
            "**Expert Insights:**\n"
            "Dr. Sarah Chen (MIT AI Lab): 'Multi-agent systems represent the next "
            "paradigm shift in enterprise automation. We're seeing coordination "
            "efficiency rates of 92-96% in production deployments.'\n\n"
            "**Trends:**\n"
            "1. Agentic AI replacing traditional RPA workflows\n"
            "2. LangGraph adoption surging for state machine orchestration\n"
            "3. Cost optimization through intelligent routing and caching\n"
            "4. Real-time monitoring dashboards becoming standard"
        ),
    ],
    "drafter": [
        (
            "# The Rise of Multi-Agent AI Systems\n\n"
            "## Introduction\n"
            "The enterprise AI landscape is undergoing a fundamental transformation. "
            "Multi-agent systems, once confined to research labs, are now powering "
            "production workflows across industries.\n\n"
            "## The Numbers Tell the Story\n"
            "With market growth of 23% year-over-year and a staggering 340% average "
            "ROI, the business case for multi-agent orchestration is clear. "
            "67% of enterprises have already adopted AI-driven workflows.\n\n"
            "## How It Works\n"
            "Modern multi-agent systems use state machine orchestration (like LangGraph) "
            "to coordinate specialized agents. Each agent handles one task — research, "
            "drafting, review, or publishing — and passes results to the next.\n\n"
            "## Looking Ahead\n"
            "As coordination efficiency reaches 92-96%, the question is no longer "
            "'should we adopt multi-agent AI?' but 'how quickly can we deploy it?'"
        ),
    ],
    "reviewer": [
        (
            "# The Rise of Multi-Agent AI Systems\n\n"
            "## Introduction\n"
            "The enterprise AI landscape is undergoing a fundamental transformation. "
            "Multi-agent systems, once confined to research labs, now power production "
            "workflows across industries with remarkable efficiency.\n\n"
            "## The Numbers Tell the Story\n"
            "Market growth of 23% year-over-year and 340% average ROI make the "
            "business case for multi-agent orchestration compelling. Two-thirds of "
            "enterprises have already adopted AI-driven workflows.\n\n"
            "## How It Works\n"
            "Modern systems use state machine orchestration (e.g., LangGraph) to "
            "coordinate specialized agents. Each handles a discrete task — research, "
            "drafting, review, or publishing — passing structured results downstream.\n\n"
            "## Looking Ahead\n"
            "With coordination efficiency at 92-96% in production, adoption is "
            "accelerating. The remaining question: how quickly can organizations deploy?"
        ),
    ],
    "publisher": [
        (
            "---\n"
            "# PUBLISHED: The Rise of Multi-Agent AI Systems\n"
            "**By AI Content Pipeline | March 2026**\n\n"
            "---\n\n"
            "**TL;DR:** Multi-agent AI systems deliver 340% ROI with 92-96% "
            "coordination efficiency. Here's why enterprises are adopting them.\n\n"
            "---\n\n"
            "## Introduction\n"
            "The enterprise AI landscape is undergoing a fundamental transformation. "
            "Multi-agent systems now power production workflows with remarkable "
            "efficiency.\n\n"
            "## Key Metrics\n"
            "| Metric | Value |\n"
            "|--------|-------|\n"
            "| Market Growth | 23% YoY |\n"
            "| Average ROI | 340% |\n"
            "| Enterprise Adoption | 67% |\n"
            "| Coordination Efficiency | 92-96% |\n\n"
            "## Architecture\n"
            "State machine orchestration coordinates specialized agents in sequence: "
            "Research -> Draft -> Review -> Publish.\n\n"
            "## Conclusion\n"
            "Multi-agent AI is no longer experimental — it's production-ready "
            "and delivering measurable results.\n\n"
            "---\n"
            "*Published via Live Multi-Agent Orchestrator*"
        ),
    ],
}


class MockLLM:
    """Generates realistic mock outputs with simulated latency and token counts."""

    def __init__(self, delay_range: tuple[float, float] = (0.5, 1.5)) -> None:
        self.delay_range = delay_range

    async def generate(self, prompt: str, agent_name: str) -> AgentOutput:
        """Generate mock output with simulated processing time."""
        delay = random.uniform(*self.delay_range)
        await asyncio.sleep(delay)

        outputs = _MOCK_OUTPUTS.get(agent_name, ["Mock output for " + agent_name])
        content = random.choice(outputs)

        # Simulate token counts based on content length
        tokens = len(content.split()) * 2 + len(prompt.split())

        return AgentOutput(content=content, tokens_used=tokens)


class ClaudeLLM:
    """Real Claude claude-haiku-4-5-20251001 provider via Anthropic SDK."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        import anthropic

        self.client = anthropic.AsyncAnthropic(
            api_key=api_key or os.environ["ANTHROPIC_API_KEY"]
        )

    async def generate(self, prompt: str, agent_name: str) -> AgentOutput:
        """Generate real output using Claude claude-haiku-4-5-20251001."""
        response = await self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        content = response.content[0].text
        tokens_used = (response.usage.input_tokens + response.usage.output_tokens)

        return AgentOutput(content=content, tokens_used=tokens_used)


# Multi-model support
# Claude (default): model="claude-haiku-4-5-20251001"  (requires ANTHROPIC_API_KEY)
# GPT-4.1: model="gpt-4.1"  (requires OPENAI_API_KEY — use openai.AsyncOpenAI)
# GLM-4 Plus (Zhipu AI): model="glm-4-plus"  (requires ZHIPUAI_API_KEY)
#   OpenAI-compatible API: https://open.bigmodel.cn/api/paas/v4/
#   Usage: openai.AsyncOpenAI(api_key=os.environ["ZHIPUAI_API_KEY"],
#                             base_url="https://open.bigmodel.cn/api/paas/v4/")
# GLM-4 Flash (fast/cheap variant): model="glm-4-flash"  (requires ZHIPUAI_API_KEY)


def get_llm() -> MockLLM | ClaudeLLM:
    """Return ClaudeLLM if API key is available, otherwise MockLLM."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        return ClaudeLLM(api_key=api_key)
    return MockLLM()
