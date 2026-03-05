"""Agent registry and health tracking.

Simplified from EnterpriseHub's AgentMeshCoordinator, removing GHL/MCP/
ProgressiveSkills dependencies while preserving the core patterns:
- Agent registration with status tracking
- Health monitoring via heartbeats
- Cost and token tracking per agent
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass
class AgentMetrics:
    """Per-agent performance metrics."""

    tokens_used: int = 0
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_latency_s: float = 0.0
    last_heartbeat: float = field(default_factory=time.time)

    @property
    def avg_latency_s(self) -> float:
        if self.completed_tasks == 0:
            return 0.0
        return self.total_latency_s / self.completed_tasks

    @property
    def success_rate(self) -> float:
        if self.total_tasks == 0:
            return 100.0
        return (self.completed_tasks / self.total_tasks) * 100

    @property
    def estimated_cost(self) -> float:
        """Estimate cost at Haiku rates ($0.25/1M input, $1.25/1M output).
        Use blended $0.50/1M for simplicity."""
        return self.tokens_used * 0.5 / 1_000_000


@dataclass
class AgentRecord:
    """Registration record for a mesh agent."""

    name: str
    status: AgentStatus = AgentStatus.IDLE
    metrics: AgentMetrics = field(default_factory=AgentMetrics)
    output: str = ""

    @property
    def is_healthy(self) -> bool:
        """Agent is healthy if heartbeat was within last 60s."""
        return (time.time() - self.metrics.last_heartbeat) < 60


class AgentRegistry:
    """Thread-safe agent registry with health tracking."""

    def __init__(self) -> None:
        self._agents: dict[str, AgentRecord] = {}

    def register(self, name: str) -> AgentRecord:
        """Register an agent and return its record."""
        record = AgentRecord(name=name)
        self._agents[name] = record
        return record

    def get(self, name: str) -> Optional[AgentRecord]:
        return self._agents.get(name)

    def all_agents(self) -> list[AgentRecord]:
        return list(self._agents.values())

    def set_status(self, name: str, status: AgentStatus) -> None:
        if agent := self._agents.get(name):
            agent.status = status
            agent.metrics.last_heartbeat = time.time()

    def record_completion(
        self, name: str, tokens: int, latency_s: float, output: str
    ) -> None:
        """Record successful task completion."""
        if agent := self._agents.get(name):
            agent.status = AgentStatus.DONE
            agent.metrics.tokens_used += tokens
            agent.metrics.total_tasks += 1
            agent.metrics.completed_tasks += 1
            agent.metrics.total_latency_s += latency_s
            agent.metrics.last_heartbeat = time.time()
            agent.output = output

    def record_failure(self, name: str, error: str) -> None:
        """Record task failure."""
        if agent := self._agents.get(name):
            agent.status = AgentStatus.ERROR
            agent.metrics.total_tasks += 1
            agent.metrics.failed_tasks += 1
            agent.metrics.last_heartbeat = time.time()
            agent.output = f"ERROR: {error}"

    @property
    def total_tokens(self) -> int:
        return sum(a.metrics.tokens_used for a in self._agents.values())

    @property
    def total_cost(self) -> float:
        return sum(a.metrics.estimated_cost for a in self._agents.values())
