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


def test_count_pending_approvals():
    store = RunStore()
    store.create_run("r1", "w")
    store.create_run("r2", "w")
    assert store.count_pending_approvals() == 0
    first = store.save_approval("r1", "gate", "Ship it?")
    store.save_approval("r1", "gate2", "Deploy?")
    store.save_approval("r2", "gate", "Ship it?")
    assert store.count_pending_approvals() == 3
    assert store.count_pending_approvals("r1") == 2
    assert store.count_pending_approvals("missing") == 0
    store.resolve_approval(first, {"verdict": "approve"})
    assert store.count_pending_approvals() == 2
    assert store.count_pending_approvals("r1") == 1


def test_delete_run_removes_related_rows():
    store = RunStore()
    store.create_run("r1", "a")
    store.create_run("r2", "b")
    store.append_event(Event(seq=0, run_id="r1", type="workflow_started",
                             payload={}, ts=1.0))
    store.save_checkpoint("r1", 1, {"x": 1})
    store.save_approval("r1", "gate", "Ship it?")
    assert store.delete_run("r1") is True
    assert store.get_run("r1") is None
    assert store.get_events("r1") == []
    assert store.load_checkpoint("r1") is None
    assert store.pending_approvals("r1") == []
    # The other run is untouched, and a second delete reports False.
    assert store.get_run("r2") is not None
    assert store.delete_run("r1") is False
    assert store.delete_run("missing") is False


def test_count_runs_by_status():
    store = RunStore()
    assert store.count_runs_by_status() == {}
    store.create_run("r1", "a")
    store.create_run("r2", "b")
    store.create_run("r3", "c")
    store.update_run_status("r2", "completed")
    store.update_run_status("r3", "completed")
    assert store.count_runs_by_status() == {"running": 1, "completed": 2}


def test_count_runs():
    store = RunStore()
    assert store.count_runs() == 0
    store.create_run("r1", "a")
    store.create_run("r2", "b")
    assert store.count_runs() == 2
    store.update_run_status("r2", "completed")
    assert store.count_runs() == 2
    store.delete_run("r1")
    assert store.count_runs() == 1


def test_has_run():
    store = RunStore()
    assert store.has_run("missing") is False
    store.create_run("r1", "demo")
    assert store.has_run("r1") is True
    store.delete_run("r1")
    assert store.has_run("r1") is False


def test_list_run_ids_in_creation_order():
    store = RunStore()
    assert store.list_run_ids() == []
    store.create_run("r1", "a")
    store.create_run("r2", "b")
    store.create_run("r3", "c")
    store.update_run_status("r2", "completed")
    assert store.list_run_ids() == ["r1", "r2", "r3"]
    assert store.list_run_ids("completed") == ["r2"]
    assert store.list_run_ids("missing-status") == []
    store.delete_run("r1")
    assert store.list_run_ids() == ["r2", "r3"]


def test_has_checkpoint():
    store = RunStore()
    assert store.has_checkpoint("r1") is False
    store.create_run("r1", "demo")
    assert store.has_checkpoint("r1") is False
    store.save_checkpoint("r1", 1, {"x": 1})
    assert store.has_checkpoint("r1") is True
    store.delete_run("r1")
    assert store.has_checkpoint("r1") is False


def test_delete_checkpoint():
    store = RunStore()
    store.create_run("r1", "demo")
    assert store.delete_checkpoint("r1") is False
    store.save_checkpoint("r1", 2, {"x": 2})
    assert store.delete_checkpoint("r1") is True
    assert store.has_checkpoint("r1") is False
    assert store.load_checkpoint("r1") is None
    # The run itself survives, and a second delete reports False.
    assert store.get_run("r1") is not None
    assert store.delete_checkpoint("r1") is False


def test_count_checkpoints():
    store = RunStore()
    assert store.count_checkpoints() == 0
    store.create_run("r1", "a")
    store.create_run("r2", "b")
    store.save_checkpoint("r1", 1, {"x": 1})
    assert store.count_checkpoints() == 1
    # Saving again for the same run replaces, it does not add.
    store.save_checkpoint("r1", 2, {"x": 2})
    assert store.count_checkpoints() == 1
    store.save_checkpoint("r2", 1, {"y": 1})
    assert store.count_checkpoints() == 2
    store.delete_checkpoint("r1")
    assert store.count_checkpoints() == 1
    store.delete_run("r2")
    assert store.count_checkpoints() == 0


def test_list_checkpoint_run_ids_sorted():
    store = RunStore()
    assert store.list_checkpoint_run_ids() == []
    store.create_run("r2", "b")
    store.create_run("r1", "a")
    store.create_run("r3", "c")
    store.save_checkpoint("r2", 1, {"x": 1})
    store.save_checkpoint("r1", 1, {"y": 1})
    assert store.list_checkpoint_run_ids() == ["r1", "r2"]
    store.delete_checkpoint("r1")
    assert store.list_checkpoint_run_ids() == ["r2"]
    store.delete_run("r2")
    assert store.list_checkpoint_run_ids() == []


def test_checkpoint_step_reports_saved_step():
    store = RunStore()
    assert store.checkpoint_step("r1") is None
    store.create_run("r1", "demo")
    assert store.checkpoint_step("r1") is None
    store.save_checkpoint("r1", 4, {"x": 1})
    assert store.checkpoint_step("r1") == 4
    # Saving again replaces the step, and deleting clears it.
    store.save_checkpoint("r1", 7, {"x": 2})
    assert store.checkpoint_step("r1") == 7
    store.delete_checkpoint("r1")
    assert store.checkpoint_step("r1") is None


def test_has_events_tracks_stored_events():
    store = RunStore()
    assert store.has_events("r1") is False
    store.create_run("r1", "w")
    assert store.has_events("r1") is False
    store.append_event(Event(seq=0, run_id="r1", type="workflow_started",
                             payload={}, ts=1.0))
    assert store.has_events("r1") is True
    assert store.has_events("other") is False


def test_latest_event_type_reports_most_recent_event():
    store = RunStore()
    assert store.latest_event_type("r1") is None
    store.create_run("r1", "w")
    assert store.latest_event_type("r1") is None
    store.append_event(Event(seq=0, run_id="r1", type="workflow_started",
                             payload={}, ts=1.0))
    assert store.latest_event_type("r1") == "workflow_started"
    store.append_event(Event(seq=1, run_id="r1", type="step_completed",
                             payload={}, ts=2.0))
    assert store.latest_event_type("r1") == "step_completed"
    assert store.latest_event_type("other") is None


def test_count_events_by_type_groups_stored_events():
    store = RunStore()
    assert store.count_events_by_type("r1") == {}
    store.create_run("r1", "w")
    assert store.count_events_by_type("r1") == {}
    store.append_event(Event(seq=0, run_id="r1", type="workflow_started",
                             payload={}, ts=1.0))
    store.append_event(Event(seq=1, run_id="r1", type="step_completed",
                             payload={}, ts=2.0))
    store.append_event(Event(seq=2, run_id="r1", type="step_completed",
                             payload={}, ts=3.0))
    assert store.count_events_by_type("r1") == {
        "workflow_started": 1,
        "step_completed": 2,
    }
    assert store.count_events_by_type("other") == {}


def test_first_event_type_reports_earliest_event():
    store = RunStore()
    assert store.first_event_type("r1") is None
    store.create_run("r1", "w")
    assert store.first_event_type("r1") is None
    store.append_event(Event(seq=0, run_id="r1", type="workflow_started",
                             payload={}, ts=1.0))
    assert store.first_event_type("r1") == "workflow_started"
    store.append_event(Event(seq=1, run_id="r1", type="step_completed",
                             payload={}, ts=2.0))
    assert store.first_event_type("r1") == "workflow_started"
    assert store.first_event_type("other") is None
