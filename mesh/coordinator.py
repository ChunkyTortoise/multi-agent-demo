"""Simplified agent mesh coordinator.

Distilled from EnterpriseHub's AgentMeshCoordinator. Preserves:
- Agent registration and health tracking
- Task routing (sequential in this demo)
- Cost tracking and budget controls
- Performance metrics aggregation

Removed EnterpriseHub-specific dependencies:
- GHL/MCP/ProgressiveSkills integrations
- Background monitoring loops
- HTTP health checks
- User quotas and SLA enforcement
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from mesh.registry import AgentRegistry, AgentStatus

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Result from a coordinated task execution."""

    agent_name: str
    success: bool
    output: str
    tokens_used: int
    latency_s: float
    error: Optional[str] = None


class MeshCoordinator:
    """Coordinates agent execution with health and cost tracking.

    Usage::

        coordinator = MeshCoordinator()
        coordinator.register_agents(["researcher", "drafter", "reviewer", "publisher"])

        # Execute and track
        coordinator.start_agent("researcher")
        result = await run_agent(...)
        coordinator.complete_agent("researcher", tokens=150, output="...")
    """

    def __init__(self, budget_limit: float = 1.0) -> None:
        self.registry = AgentRegistry()
        self.budget_limit = budget_limit
        self.task_results: list[TaskResult] = []

    def register_agents(self, names: list[str]) -> None:
        """Register a list of agents."""
        for name in names:
            self.registry.register(name)
            logger.info("Agent registered: %s", name)

    def start_agent(self, name: str) -> None:
        """Mark agent as running."""
        self.registry.set_status(name, AgentStatus.RUNNING)
        logger.info("Agent started: %s", name)

    def complete_agent(
        self, name: str, tokens: int, latency_s: float, output: str
    ) -> TaskResult:
        """Record agent completion and return result."""
        self.registry.record_completion(name, tokens, latency_s, output)
        result = TaskResult(
            agent_name=name,
            success=True,
            output=output,
            tokens_used=tokens,
            latency_s=latency_s,
        )
        self.task_results.append(result)
        logger.info(
            "Agent completed: %s (tokens=%d, latency=%.2fs)",
            name,
            tokens,
            latency_s,
        )
        return result

    def fail_agent(self, name: str, error: str) -> TaskResult:
        """Record agent failure."""
        self.registry.record_failure(name, error)
        result = TaskResult(
            agent_name=name,
            success=False,
            output="",
            tokens_used=0,
            latency_s=0,
            error=error,
        )
        self.task_results.append(result)
        logger.error("Agent failed: %s — %s", name, error)
        return result

    def check_budget(self) -> bool:
        """Check if total cost is within budget."""
        return self.registry.total_cost <= self.budget_limit

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive mesh status (mirrors EnterpriseHub's get_mesh_status)."""
        agents = self.registry.all_agents()
        return {
            "agents": {
                "total": len(agents),
                "idle": sum(1 for a in agents if a.status == AgentStatus.IDLE),
                "running": sum(
                    1 for a in agents if a.status == AgentStatus.RUNNING
                ),
                "done": sum(1 for a in agents if a.status == AgentStatus.DONE),
                "error": sum(
                    1 for a in agents if a.status == AgentStatus.ERROR
                ),
            },
            "tokens": {
                "total": self.registry.total_tokens,
                "by_agent": {
                    a.name: a.metrics.tokens_used for a in agents
                },
            },
            "cost": {
                "total": self.registry.total_cost,
                "budget_limit": self.budget_limit,
                "within_budget": self.check_budget(),
            },
            "performance": {
                "avg_latency_s": (
                    sum(a.metrics.avg_latency_s for a in agents) / len(agents)
                    if agents
                    else 0
                ),
                "success_rate": (
                    sum(a.metrics.success_rate for a in agents) / len(agents)
                    if agents
                    else 0
                ),
            },
        }
