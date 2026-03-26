"""TypedDict state definition for the content pipeline."""

import operator
from typing import Annotated, Any, TypedDict


class AgentOutput(TypedDict, total=False):
    """Output from a single agent execution."""

    content: str
    tokens_used: int
    error: str


class ToolCall(TypedDict, total=False):
    """Record of a single tool invocation during agent execution."""

    tool_name: str
    input: dict[str, Any]
    output: str
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

    # Tool use (populated by research_node when tool_provider is supplied)
    tool_calls: list[ToolCall]  # all tool invocations recorded in order
    retrieved_context: list[dict[str, Any]]  # documents from retrieve_docs tool

    # Planning (populated by planner_node when use_planner=True)
    plan: list[str]  # ordered list of research sub-tasks from the planner
    use_planner: bool  # whether to run planner_node before researcher

    # Parallel execution (populated when use_parallel=True)
    # Annotated reducer: each sub_researcher appends its result via operator.add
    parallel_results: Annotated[list[dict[str, Any]], operator.add]
    use_parallel: bool  # whether to fan-out sub-tasks to parallel researchers
