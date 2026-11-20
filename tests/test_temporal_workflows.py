"""Workflow tests against Temporal's time-skipping test environment.

These spin up the downloadable test server via
``WorkflowEnvironment.start_time_skipping`` — no docker, no cluster.
If the test-server binary cannot be downloaded (offline sandbox), the
whole module is skipped and the activity tests still cover the logic.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from agentforge.store import RunStore
from agentforge.temporal.activities import AgentActivities
from agentforge.temporal.models import AgentRunInput, ApprovalSignalPayload
from agentforge.temporal.worker import build_demo_registry
from agentforge.temporal.workflows import AgentRunWorkflow

pytestmark = pytest.mark.asyncio


@pytest.fixture(scope="module")
def event_loop_policy():
    return asyncio.DefaultEventLoopPolicy()


async def _start_env() -> WorkflowEnvironment:
    try:
        return await asyncio.wait_for(
            WorkflowEnvironment.start_time_skipping(), timeout=90
        )
    except Exception as exc:  # binary download blocked, sandbox, etc.
        pytest.skip(f"Temporal test server unavailable: {exc}")


def _worker(env: WorkflowEnvironment, store: RunStore) -> Worker:
    activities = AgentActivities(registry=build_demo_registry(), store=store)
    return Worker(
        env.client,
        task_queue="test-agentforge",
        workflows=[AgentRunWorkflow],
        activities=activities.all(),
    )


async def test_run_completes_without_approval_gate():
    env = await _start_env()
    async with env:
        store = RunStore()
        async with _worker(env, store):
            run_id = f"t-{uuid.uuid4().hex[:8]}"
            result = await env.client.execute_workflow(
                AgentRunWorkflow.run,
                AgentRunInput(run_id=run_id,
                              task="count the words in this sentence"),
                id=run_id,
                task_queue="test-agentforge",
            )
            assert result.status == "completed"
            assert result.steps[0].tool == "word_count"
            assert '"words": 6' in result.steps[0].observation
            assert store.get_run(run_id)["status"] == "completed"


async def test_approval_gate_edit_then_execute():
    env = await _start_env()
    async with env:
        store = RunStore()
        async with _worker(env, store):
            run_id = f"t-{uuid.uuid4().hex[:8]}"
            handle = await env.client.start_workflow(
                AgentRunWorkflow.run,
                AgentRunInput(
                    run_id=run_id,
                    task="count the words",
                    approval_tools=["word_count"],
                ),
                id=run_id,
                task_queue="test-agentforge",
            )
            # Wait until the workflow reports it is parked at the gate.
            for _ in range(200):
                state = await handle.query(AgentRunWorkflow.run_state)
                if state.awaiting_approval:
                    break
                await asyncio.sleep(0.05)
            assert state.awaiting_approval
            assert store.get_run(run_id)["status"] == "awaiting_approval"

            await handle.signal(
                AgentRunWorkflow.approve,
                ApprovalSignalPayload(
                    verdict="edit", edits={"text": "one two three"},
                    by="reviewer",
                ),
            )
            result = await handle.result()
            assert result.status == "completed"
            assert result.steps[0].status == "edited"
            assert '"words": 3' in result.steps[0].observation
            assert store.pending_approvals(run_id) == []


async def test_approval_gate_reject_ends_run():
    env = await _start_env()
    async with env:
        store = RunStore()
        async with _worker(env, store):
            run_id = f"t-{uuid.uuid4().hex[:8]}"
            handle = await env.client.start_workflow(
                AgentRunWorkflow.run,
                AgentRunInput(
                    run_id=run_id,
                    task="count the words",
                    approval_tools=["word_count"],
                ),
                id=run_id,
                task_queue="test-agentforge",
            )
            for _ in range(200):
                state = await handle.query(AgentRunWorkflow.run_state)
                if state.awaiting_approval:
                    break
                await asyncio.sleep(0.05)
            assert state.awaiting_approval
            await handle.signal(
                AgentRunWorkflow.approve,
                ApprovalSignalPayload(verdict="reject", by="reviewer"),
            )
            result = await handle.result()
            assert result.status == "rejected"
            assert result.steps[0].status == "rejected"
            assert store.get_run(run_id)["status"] == "failed"
