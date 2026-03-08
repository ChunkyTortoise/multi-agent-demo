"""TypedDict state definition for the content pipeline."""

from typing import TypedDict


class AgentOutput(TypedDict, total=False):
    """Output from a single agent execution."""

    content: str
    tokens_used: int
    error: str


class PipelineState(TypedDict, total=False):
    """State passed through every node of the content pipeline graph."""

    # Input
    topic: str

    # Agent outputs
    research_output: AgentOutput
    draft_output: AgentOutput
    review_output: AgentOutput
    publish_output: AgentOutput

    # Pipeline metadata
    current_agent: str
    completed_agents: list[str]
    total_tokens: int
    error: str

    # Revision loop
    revision_count: int  # number of revision loops completed
    review_score: float  # quality score from reviewer (0.0-1.0)
