"""Tests for the SQLite run store."""

from agentforge.store import RunStore
from agentforge.engine import Event


def test_run_lifecycle():
    store = RunStore()
    store.create_run("r1", "my-workflow")
    run = store.get_run("r1")
    assert run["name"] == "my-workflow"
    assert run["status"] == "running"

    store.update_run_status("r1", "completed")
    assert store.get_run("r1")["status"] == "completed"
    assert store.get_run("missing") is None


def test_list_runs_filters_by_status():
    store = RunStore()
    store.create_run("r1", "a")
    store.create_run("r2", "b")
    store.update_run_status("r2", "paused")
    assert {r["id"] for r in store.list_runs()} == {"r1", "r2"}
    assert [r["id"] for r in store.list_runs("paused")] == ["r2"]


def test_event_append_and_readback():
    store = RunStore()
    store.create_run("r1", "w")
    store.append_event(Event(seq=0, run_id="r1", type="workflow_started",
                             payload={"a": 1}, ts=1.0))
    store.append_event(Event(seq=1, run_id="r1", type="step_completed",
                             payload={"b": [1, 2]}, ts=2.0))
    events = store.get_events("r1")
    assert len(events) == 2
    assert events[0].type == "workflow_started"
    assert events[0].payload == {"a": 1}
    assert events[1].payload == {"b": [1, 2]}
    assert store.count_events("r1") == 2
    assert store.get_events("other") == []


def test_checkpoint_roundtrip():
    store = RunStore()
    store.create_run("r1", "w")
    assert store.load_checkpoint("r1") is None
    store.save_checkpoint("r1", 2, {"x": 1, "nested": {"y": [1]}})
    loaded = store.load_checkpoint("r1")
    assert loaded["step_index"] == 2
    assert loaded["state"] == {"x": 1, "nested": {"y": [1]}}
    # Overwrite keeps the latest.
    store.save_checkpoint("r1", 3, {"x": 2})
    assert store.load_checkpoint("r1")["state"] == {"x": 2}


def test_approval_lifecycle():
    store = RunStore()
    store.create_run("r1", "w")
    aid = store.save_approval("r1", "gate", "Ship it?", {"plan": "v2"})
    pending = store.pending_approvals("r1")
    assert len(pending) == 1
    assert pending[0]["id"] == aid
    assert pending[0]["context"] == {"plan": "v2"}
    assert pending[0]["decision"] is None

    store.resolve_approval(aid, {"verdict": "approve"})
    assert store.pending_approvals("r1") == []
    # Resolved approvals are still queryable across runs.
    assert store.pending_approvals() == []
