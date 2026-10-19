"""Tests for the durable workflow engine: retries, checkpoints, replay."""

import pytest

from agentforge.engine import (
    EventType,
    RetryPolicy,
    RunStatus,
    Step,
    WorkflowContext,
    WorkflowEngine,
)
from agentforge.store import RunStore


def make_engine(**kwargs):
    kwargs.setdefault("sleep_fn", lambda s: None)  # never actually sleep
    kwargs.setdefault("store", RunStore())
    return WorkflowEngine(**kwargs)


def test_happy_path_records_events():
    engine = make_engine()
    engine.register_workflow("w", [
        Step("a", lambda ctx: {"x": 1}),
        Step("b", lambda ctx: {"y": ctx.state["x"] + 1}),
    ])
    result = engine.run("w")
    assert result.status == RunStatus.COMPLETED
    assert result.state == {"x": 1, "y": 2}
    types = [e.type for e in result.events]
    assert types[0] == EventType.WORKFLOW_STARTED.value
    assert types[-1] == EventType.WORKFLOW_COMPLETED.value
    assert types.count(EventType.STEP_COMPLETED.value) == 2


def test_step_retry_succeeds_after_failures():
    attempts = []

    def flaky(ctx):
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("boom")
        return {"ok": True}

    engine = make_engine()
    engine.register_workflow("w", [
        Step("flaky", flaky, retry=RetryPolicy(max_attempts=3, jitter=False)),
    ])
    result = engine.run("w")
    assert result.status == RunStatus.COMPLETED
    assert result.state == {"ok": True}
    failed = [e for e in result.events
              if e.type == EventType.STEP_FAILED.value]
    assert len(failed) == 2
    assert [e.payload["attempt"] for e in failed] == [1, 2]


def test_step_retry_exhaustion_fails_run():
    def always_fails(ctx):
        raise RuntimeError("permanent")

    engine = make_engine()
    engine.register_workflow("w", [
        Step("bad", always_fails,
             retry=RetryPolicy(max_attempts=2, jitter=False)),
    ])
    result = engine.run("w")
    assert result.status == RunStatus.FAILED
    assert "permanent" in result.error
    types = [e.type for e in result.events]
    assert EventType.STEP_EXHAUSTED.value in types
    assert types[-1] == EventType.WORKFLOW_FAILED.value


def test_backoff_delays_are_exponential():
    policy = RetryPolicy(max_attempts=4, base_delay=1.0, max_delay=100.0,
                         jitter=False)
    assert policy.delay_for(1) == 0.0
    assert policy.delay_for(2) == 1.0
    assert policy.delay_for(3) == 2.0
    assert policy.delay_for(4) == 4.0


def test_backoff_jitter_stays_within_bounds():
    import random
    policy = RetryPolicy(max_attempts=3, base_delay=2.0, jitter=True)
    rng = random.Random(42)
    for _ in range(50):
        d = policy.delay_for(3, rng)
        assert 0.0 <= d <= 4.0


def test_sleep_called_with_backoff():
    sleeps = []
    attempts = []

    def fail_twice(ctx):
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("x")
        return {}

    engine = make_engine(sleep_fn=sleeps.append)
    engine.register_workflow("w", [
        Step("s", fail_twice,
             retry=RetryPolicy(max_attempts=3, base_delay=1.0, jitter=False)),
    ])
    engine.run("w")
    delays = [s for s in sleeps if isinstance(s, float)]
    assert delays == [1.0, 2.0]


def test_replay_rebuilds_state_without_rerunning():
    calls = []

    def tracked(ctx):
        calls.append(1)
        return {"n": len(calls)}

    engine = make_engine()
    engine.register_workflow("w", [Step("t", tracked)])
    result = engine.run("w")
    assert len(calls) == 1

    replayed = engine.replay_state(result.run_id)
    assert replayed == {"n": 1}
    assert len(calls) == 1  # replay must not re-execute steps


def test_checkpoint_and_resume_from_store(tmp_path):
    db = str(tmp_path / "durable.db")
    engine = WorkflowEngine(store=RunStore(db), sleep_fn=lambda s: None)
    engine.register_workflow("w", [
        Step("one", lambda ctx: {"a": 1}, checkpoint_after=True),
        Step("two", lambda ctx: {"b": 2}),
    ])
    result = engine.run("w")
    assert result.status == RunStatus.COMPLETED

    # A brand-new engine over the same store can replay the run.
    engine2 = WorkflowEngine(store=RunStore(db), sleep_fn=lambda s: None)
    assert engine2.replay_state(result.run_id) == {"a": 1, "b": 2}


def test_non_dict_step_result_stored_under_step_name():
    engine = make_engine()
    engine.register_workflow("w", [Step("greet", lambda ctx: "hello")])
    result = engine.run("w")
    assert result.state == {"greet": "hello"}


def test_unknown_workflow_raises():
    engine = make_engine()
    with pytest.raises(ValueError, match="unknown workflow"):
        engine.run("nope")


def test_pause_is_not_retried():
    from agentforge.engine import PauseWorkflow

    attempts = []

    def pauser(ctx):
        attempts.append(1)
        raise PauseWorkflow("are you sure?")

    engine = make_engine()
    engine.register_workflow("w", [
        Step("p", pauser, retry=RetryPolicy(max_attempts=5, jitter=False)),
    ])
    result = engine.run("w")
    assert result.status == RunStatus.PAUSED
    assert len(attempts) == 1  # pausing must never be treated as failure
