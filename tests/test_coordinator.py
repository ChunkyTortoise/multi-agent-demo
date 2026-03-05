"""Tests for the mesh coordinator and agent registry."""

from __future__ import annotations

import pytest

from mesh.coordinator import MeshCoordinator
from mesh.registry import AgentRecord, AgentRegistry, AgentStatus


class TestAgentRegistry:
    """Tests for AgentRegistry."""

    def test_register_agent(self):
        registry = AgentRegistry()
        agent = registry.register("researcher")
        assert agent.name == "researcher"
        assert agent.status == AgentStatus.IDLE

    def test_get_agent(self):
        registry = AgentRegistry()
        registry.register("researcher")
        agent = registry.get("researcher")
        assert agent is not None
        assert agent.name == "researcher"

    def test_get_unknown_agent_returns_none(self):
        registry = AgentRegistry()
        assert registry.get("unknown") is None

    def test_all_agents(self):
        registry = AgentRegistry()
        registry.register("a")
        registry.register("b")
        assert len(registry.all_agents()) == 2

    def test_set_status(self):
        registry = AgentRegistry()
        registry.register("researcher")
        registry.set_status("researcher", AgentStatus.RUNNING)
        assert registry.get("researcher").status == AgentStatus.RUNNING

    def test_record_completion(self):
        registry = AgentRegistry()
        registry.register("researcher")
        registry.record_completion("researcher", tokens=100, latency_s=1.5, output="done")

        agent = registry.get("researcher")
        assert agent.status == AgentStatus.DONE
        assert agent.metrics.tokens_used == 100
        assert agent.metrics.completed_tasks == 1
        assert agent.output == "done"

    def test_record_failure(self):
        registry = AgentRegistry()
        registry.register("researcher")
        registry.record_failure("researcher", "timeout")

        agent = registry.get("researcher")
        assert agent.status == AgentStatus.ERROR
        assert agent.metrics.failed_tasks == 1

    def test_total_tokens(self):
        registry = AgentRegistry()
        registry.register("a")
        registry.register("b")
        registry.record_completion("a", tokens=100, latency_s=1.0, output="x")
        registry.record_completion("b", tokens=200, latency_s=1.0, output="y")
        assert registry.total_tokens == 300

    def test_total_cost(self):
        registry = AgentRegistry()
        registry.register("a")
        registry.record_completion("a", tokens=1_000_000, latency_s=1.0, output="x")
        # At $0.50/1M tokens
        assert abs(registry.total_cost - 0.5) < 0.001

    def test_success_rate(self):
        registry = AgentRegistry()
        registry.register("a")
        registry.record_completion("a", tokens=10, latency_s=1.0, output="ok")
        registry.record_completion("a", tokens=10, latency_s=1.0, output="ok")
        assert registry.get("a").metrics.success_rate == 100.0

    def test_success_rate_with_failures(self):
        registry = AgentRegistry()
        registry.register("a")
        registry.record_completion("a", tokens=10, latency_s=1.0, output="ok")
        registry.record_failure("a", "err")
        assert registry.get("a").metrics.success_rate == 50.0

    def test_avg_latency(self):
        registry = AgentRegistry()
        registry.register("a")
        registry.record_completion("a", tokens=10, latency_s=2.0, output="ok")
        registry.record_completion("a", tokens=10, latency_s=4.0, output="ok")
        assert abs(registry.get("a").metrics.avg_latency_s - 3.0) < 0.001


class TestMeshCoordinator:
    """Tests for MeshCoordinator."""

    def test_register_agents(self):
        coord = MeshCoordinator()
        coord.register_agents(["a", "b", "c"])
        assert len(coord.registry.all_agents()) == 3

    def test_start_agent(self):
        coord = MeshCoordinator()
        coord.register_agents(["researcher"])
        coord.start_agent("researcher")
        assert coord.registry.get("researcher").status == AgentStatus.RUNNING

    def test_complete_agent(self):
        coord = MeshCoordinator()
        coord.register_agents(["researcher"])
        coord.start_agent("researcher")
        result = coord.complete_agent("researcher", tokens=50, latency_s=1.0, output="done")

        assert result.success is True
        assert result.tokens_used == 50
        assert len(coord.task_results) == 1

    def test_fail_agent(self):
        coord = MeshCoordinator()
        coord.register_agents(["researcher"])
        coord.start_agent("researcher")
        result = coord.fail_agent("researcher", "boom")

        assert result.success is False
        assert result.error == "boom"

    def test_budget_check_within(self):
        coord = MeshCoordinator(budget_limit=1.0)
        coord.register_agents(["a"])
        coord.complete_agent("a", tokens=100, latency_s=1.0, output="x")
        assert coord.check_budget() is True

    def test_budget_check_exceeded(self):
        coord = MeshCoordinator(budget_limit=0.0001)
        coord.register_agents(["a"])
        coord.complete_agent("a", tokens=1_000_000, latency_s=1.0, output="x")
        assert coord.check_budget() is False

    def test_get_status(self):
        coord = MeshCoordinator()
        coord.register_agents(["a", "b"])
        coord.start_agent("a")
        coord.complete_agent("a", tokens=100, latency_s=1.0, output="x")

        status = coord.get_status()
        assert status["agents"]["total"] == 2
        assert status["agents"]["done"] == 1
        assert status["agents"]["idle"] == 1
        assert status["tokens"]["total"] == 100
        assert status["cost"]["within_budget"] is True

    def test_full_pipeline_tracking(self):
        """Simulate a full 4-agent pipeline through the coordinator."""
        coord = MeshCoordinator()
        agents = ["researcher", "drafter", "reviewer", "publisher"]
        coord.register_agents(agents)

        for name in agents:
            coord.start_agent(name)
            coord.complete_agent(name, tokens=50, latency_s=0.5, output=f"{name} done")

        status = coord.get_status()
        assert status["agents"]["done"] == 4
        assert status["tokens"]["total"] == 200
        assert len(coord.task_results) == 4
