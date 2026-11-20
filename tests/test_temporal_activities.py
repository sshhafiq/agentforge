"""Unit tests for the Temporal activities (no Temporal server needed).

Activities are plain async methods bound to a registry + store, so they
can be exercised directly. Workflow-level tests live in
``test_temporal_workflows.py``.
"""

from __future__ import annotations

import asyncio

import pytest
from temporalio.exceptions import ApplicationError

from agentforge.store import RunStore
from agentforge.temporal.activities import AgentActivities, heuristic_planner
from agentforge.temporal.models import (
    ExecuteToolInput,
    FinishRunInput,
    PlanStepsInput,
    RecordRunStartedInput,
    RequestApprovalInput,
    ResolveApprovalInput,
)
from agentforge.temporal.worker import build_demo_registry


@pytest.fixture()
def activities() -> AgentActivities:
    return AgentActivities(registry=build_demo_registry(), store=RunStore())


def run(coro):
    return asyncio.run(coro)


# ----------------------------------------------------------------------
# planner
# ----------------------------------------------------------------------
def test_heuristic_planner_matches_tools_by_keyword():
    registry = build_demo_registry()
    plan = heuristic_planner("please count the words in this text", registry, 16)
    # "words"/"text" match all three demo tools (name/description),
    # planned in registration order.
    assert [s.tool for s in plan] == ["word_count", "top_words", "longest_word"]
    assert plan[0].arguments["text"] == "please count the words in this text"


def test_heuristic_planner_no_match_empty_plan():
    registry = build_demo_registry()
    assert heuristic_planner("xyzzy plover", registry, 16) == []


def test_heuristic_planner_respects_max_steps():
    registry = build_demo_registry()
    plan = heuristic_planner("count words, most frequent words", registry, 1)
    assert len(plan) == 1


# ----------------------------------------------------------------------
# lifecycle + tool execution
# ----------------------------------------------------------------------
def test_run_lifecycle_is_recorded_in_store(activities):
    store = activities._store
    run(activities.record_run_started(
        RecordRunStartedInput(run_id="r1", name="demo", task="count words")))
    assert store.has_run("r1")
    assert store.get_run("r1")["status"] == "running"

    result = run(activities.execute_tool(
        ExecuteToolInput(run_id="r1", step_index=0, tool="word_count",
                         arguments={"text": "one two three"})))
    assert '"words": 3' in result.observation
    assert store.count_events_by_type("r1")["step_completed"] == 1

    run(activities.finish_run(
        FinishRunInput(run_id="r1", status="completed", answer="done",
                       steps_completed=1)))
    assert store.get_run("r1")["status"] == "completed"
    assert store.load_checkpoint("r1")["state"]["answer"] == "done"


def test_execute_tool_validation_error_is_non_retryable(activities):
    run(activities.record_run_started(
        RecordRunStartedInput(run_id="r2", name="demo", task="x")))
    with pytest.raises(ApplicationError) as excinfo:
        run(activities.execute_tool(
            ExecuteToolInput(run_id="r2", step_index=0, tool="word_count",
                             arguments={"wrong": "arg"})))
    assert excinfo.value.non_retryable is True


def test_execute_tool_unknown_tool_is_non_retryable(activities):
    run(activities.record_run_started(
        RecordRunStartedInput(run_id="r3", name="demo", task="x")))
    with pytest.raises(ApplicationError) as excinfo:
        run(activities.execute_tool(
            ExecuteToolInput(run_id="r3", step_index=0, tool="nope",
                             arguments={})))
    assert excinfo.value.non_retryable is True


def test_plan_steps_records_event(activities):
    run(activities.record_run_started(
        RecordRunStartedInput(run_id="r4", name="demo", task="count words")))
    plan = run(activities.plan_steps(
        PlanStepsInput(run_id="r4", task="count the words", max_steps=4)))
    assert plan[0].tool == "word_count"
    types = activities._store.count_events_by_type("r4")
    assert types["plan_created"] == 1


# ----------------------------------------------------------------------
# approval bookkeeping
# ----------------------------------------------------------------------
def test_approval_roundtrip(activities):
    store = activities._store
    run(activities.record_run_started(
        RecordRunStartedInput(run_id="r5", name="demo", task="count words")))
    approval_id = run(activities.request_approval(
        RequestApprovalInput(run_id="r5", step_index=0, tool="word_count",
                             arguments={"text": "hi"})))
    assert store.get_run("r5")["status"] == "awaiting_approval"
    pending = store.pending_approvals("r5")
    assert len(pending) == 1 and pending[0]["id"] == approval_id
    assert "word_count" in pending[0]["question"]

    run(activities.resolve_approval(
        ResolveApprovalInput(run_id="r5", approval_id=approval_id,
                             decision={"verdict": "approve", "by": "sam"})))
    assert store.pending_approvals("r5") == []
    assert store.get_run("r5")["status"] == "running"
