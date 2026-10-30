"""Durable workflow engine: event-sourced execution with checkpoint/restore,
per-step retries, and replay.

The engine treats a workflow as an ordered list of named steps. Every state
transition is recorded as an event in an append-only log, which makes
execution *durable*: if the process crashes (or a human-in-the-loop gate
pauses it), the workflow can resume from the last completed step without
re-running side effects. The same log powers *replay* — reconstructing the
workflow's state purely from history.

Design notes
------------
- Steps are plain callables ``fn(ctx) -> Any``. Their return value is merged
  into the workflow's shared state dict (if it is a dict) or stored under the
  step's name.
- Retries use exponential backoff with full jitter. The sleep function is
  injectable so tests never actually wait.
- When a ``store`` (SQLite) is attached, every event is persisted as it is
  appended, so ``resume(run_id)`` survives process restarts.
"""

from __future__ import annotations

import json
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from agentforge.store import RunStore


class EventType(str, Enum):
    """Every state transition the engine can record."""

    WORKFLOW_STARTED = "workflow_started"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"  # an attempt failed; a retry will follow
    STEP_EXHAUSTED = "step_exhausted"  # all attempts failed
    CHECKPOINT = "checkpoint"
    PAUSED = "paused"
    RESUMED = "resumed"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"


class RunStatus(str, Enum):
    """Terminal (or pausable) status of a workflow run."""

    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class PauseWorkflow(Exception):
    """Raised by a step (via ``ctx.request_approval``) to pause the run.

    The engine catches this, persists a checkpoint, and returns a
    ``RunResult`` with status ``PAUSED``. Call ``engine.resume(run_id,
    decision)`` to continue.
    """

    def __init__(self, question: str, context: Optional[dict] = None) -> None:
        super().__init__(question)
        self.question = question
        self.context = context or {}


@dataclass
class Event:
    """A single entry in a run's append-only event log."""

    seq: int
    run_id: str
    type: str
    payload: dict = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Event":
        return cls(**data)


class EventLog:
    """In-memory append-only log of :class:`Event` objects."""

    def __init__(self) -> None:
        self._events: list[Event] = []

    def append(self, run_id: str, type: str, payload: Optional[dict] = None) -> Event:
        event = Event(seq=len(self._events), run_id=run_id, type=type,
                      payload=payload or {})
        self._events.append(event)
        return event

    def events(self, run_id: Optional[str] = None) -> list[Event]:
        if run_id is None:
            return list(self._events)
        return [e for e in self._events if e.run_id == run_id]

    def events_of_type(self, type: str, run_id: Optional[str] = None) -> list[Event]:
        """Events of one type, optionally restricted to a single run."""
        return [e for e in self.events(run_id) if e.type == type]

    def to_json(self) -> str:
        return json.dumps([e.to_dict() for e in self._events])

    @classmethod
    def from_json(cls, raw: str) -> "EventLog":
        log = cls()
        for data in json.loads(raw):
            log._events.append(Event.from_dict(data))
        return log


@dataclass
class RetryPolicy:
    """Per-step retry configuration (exponential backoff with jitter)."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter: bool = True

    def delay_for(self, attempt: int, rng: random.Random = random) -> float:
        """Delay before attempt N (1-indexed). Attempt 1 never waits."""
        if attempt <= 1:
            return 0.0
        delay = min(self.base_delay * (2 ** (attempt - 2)), self.max_delay)
        if self.jitter:
            delay = rng.uniform(0, delay)
        return delay


@dataclass
class Step:
    """One named unit of work in a workflow."""

    name: str
    fn: Callable[["WorkflowContext"], Any]
    retry: Optional[RetryPolicy] = None
    checkpoint_after: bool = False


class WorkflowContext:
    """The handle a step receives: shared state plus engine services."""

    def __init__(self, engine: "WorkflowEngine", run_id: str, state: dict) -> None:
        self._engine = engine
        self.run_id = run_id
        self.state = state

    def request_approval(self, question: str, context: Optional[dict] = None) -> Any:
        """Pause the workflow until a human approves/rejects.

        Never returns normally — always raises :class:`PauseWorkflow`.
        After :meth:`WorkflowEngine.resume` the *same step* runs again and
        this call returns the human's decision instead.

        Note: a step should request at most one approval. The resumed
        decision is delivered to the next ``request_approval`` call; split
        multiple gates into separate steps.
        """
        decision = self._engine.peek_pending_decision(self.run_id)
        if decision is not None:
            return decision
        raise PauseWorkflow(question, context)

    def checkpoint(self) -> None:
        """Persist a checkpoint of the current state right now."""
        self._engine.checkpoint(self.run_id)


@dataclass
class RunResult:
    run_id: str
    status: RunStatus
    state: dict
    error: Optional[str] = None
    events: list[Event] = field(default_factory=list)


class WorkflowEngine:
    """Executes registered workflows durably."""

    def __init__(
        self,
        store: Optional[RunStore] = None,
        default_retry: Optional[RetryPolicy] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.store = store or RunStore()
        self.default_retry = default_retry or RetryPolicy()
        self.sleep_fn = sleep_fn
        self.rng = rng or random.Random()
        self._workflows: dict[str, list[Step]] = {}
        self._pending_decisions: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # registration
    # ------------------------------------------------------------------
    def register_workflow(self, name: str, steps: list[Step]) -> None:
        if not steps:
            raise ValueError("a workflow needs at least one step")
        self._workflows[name] = steps

    # ------------------------------------------------------------------
    # execution
    # ------------------------------------------------------------------
    def run(
        self,
        name: str,
        initial_state: Optional[dict] = None,
        run_id: Optional[str] = None,
    ) -> RunResult:
        steps = self._require_workflow(name)
        run_id = run_id or uuid.uuid4().hex[:12]
        state = dict(initial_state or {})
        log = EventLog()

        self.store.create_run(run_id, name)
        self._record(log, run_id, EventType.WORKFLOW_STARTED,
                     {"workflow": name, "initial_state": state})

        try:
            state = self._execute_from(log, run_id, steps, state, start_index=0)
        except PauseWorkflow as pause:
            self._pause(log, run_id, steps, state, pause)
            return RunResult(run_id, RunStatus.PAUSED, state,
                             events=log.events(run_id))
        except Exception as exc:  # noqa: BLE001 — surfaced in RunResult
            self._record(log, run_id, EventType.WORKFLOW_FAILED,
                         {"error": str(exc)})
            self.store.update_run_status(run_id, RunStatus.FAILED.value)
            return RunResult(run_id, RunStatus.FAILED, state, error=str(exc),
                             events=log.events(run_id))

        self._record(log, run_id, EventType.WORKFLOW_COMPLETED, {"state": state})
        self.store.update_run_status(run_id, RunStatus.COMPLETED.value)
        return RunResult(run_id, RunStatus.COMPLETED, state,
                         events=log.events(run_id))

    def resume(self, run_id: str, decision: Any = None) -> RunResult:
        """Resume a paused run, optionally delivering a HITL decision."""
        run = self.store.get_run(run_id)
        if run is None:
            raise ValueError(f"unknown run_id: {run_id}")
        if run["status"] != RunStatus.PAUSED.value:
            raise ValueError(f"run {run_id} is not paused (status={run['status']})")

        steps = self._require_workflow(run["name"])
        checkpoint = self.store.load_checkpoint(run_id)
        if checkpoint is None:
            raise ValueError(f"no checkpoint found for paused run {run_id}")

        if decision is not None:
            self._pending_decisions[run_id] = decision

        log = self._load_log(run_id)
        state = dict(checkpoint["state"])
        start_index = checkpoint["step_index"]
        self._record(log, run_id, EventType.RESUMED,
                     {"from_step": steps[start_index].name,
                      "decision": decision})

        try:
            state = self._execute_from(log, run_id, steps, state,
                                       start_index=start_index)
        except PauseWorkflow as pause:
            self._pause(log, run_id, steps, state, pause)
            return RunResult(run_id, RunStatus.PAUSED, state,
                             events=log.events(run_id))
        except Exception as exc:  # noqa: BLE001
            self._record(log, run_id, EventType.WORKFLOW_FAILED,
                         {"error": str(exc)})
            self.store.update_run_status(run_id, RunStatus.FAILED.value)
            return RunResult(run_id, RunStatus.FAILED, state, error=str(exc),
                             events=log.events(run_id))

        self._record(log, run_id, EventType.WORKFLOW_COMPLETED, {"state": state})
        self.store.update_run_status(run_id, RunStatus.COMPLETED.value)
        return RunResult(run_id, RunStatus.COMPLETED, state,
                         events=log.events(run_id))

    # ------------------------------------------------------------------
    # replay
    # ------------------------------------------------------------------
    def replay_state(self, run_id: str) -> dict:
        """Rebuild a run's state purely from its event log.

        No step function is executed — this is how you audit or debug a run
        without re-triggering side effects.
        """
        state: dict = {}
        for event in self._load_log(run_id).events(run_id):
            if event.type == EventType.WORKFLOW_STARTED.value:
                state.update(event.payload.get("initial_state", {}))
            elif event.type == EventType.STEP_COMPLETED.value:
                update = event.payload.get("state_update")
                if isinstance(update, dict):
                    state.update(update)
            elif event.type == EventType.CHECKPOINT.value:
                state.update(event.payload.get("state", {}))
        return state

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _require_workflow(self, name: str) -> list[Step]:
        try:
            return self._workflows[name]
        except KeyError:
            raise ValueError(f"unknown workflow: {name}") from None

    def _execute_from(
        self,
        log: EventLog,
        run_id: str,
        steps: list[Step],
        state: dict,
        start_index: int,
    ) -> dict:
        ctx = WorkflowContext(self, run_id, state)
        for index in range(start_index, len(steps)):
            step = steps[index]
            policy = step.retry or self.default_retry
            self._record(log, run_id, EventType.STEP_STARTED,
                         {"step": step.name, "attempt": 1})
            result = self._run_step_with_retries(log, run_id, step, ctx, policy)
            # The step consumed any delivered HITL decision — drop it so a
            # later gate in a later step can't see a stale decision.
            self._pending_decisions.pop(run_id, None)
            if isinstance(result, dict):
                state.update(result)
                self._record(log, run_id, EventType.STEP_COMPLETED,
                             {"step": step.name, "state_update": result})
            else:
                state[step.name] = result
                self._record(log, run_id, EventType.STEP_COMPLETED,
                             {"step": step.name,
                              "state_update": {step.name: result}})
            if step.checkpoint_after:
                self.checkpoint(run_id, step_index=index + 1)
                self._record(log, run_id, EventType.CHECKPOINT,
                             {"step": step.name, "state": dict(state)})
        return state

    def _run_step_with_retries(
        self,
        log: EventLog,
        run_id: str,
        step: Step,
        ctx: WorkflowContext,
        policy: RetryPolicy,
    ) -> Any:
        last_error: Optional[Exception] = None
        for attempt in range(1, policy.max_attempts + 1):
            if attempt > 1:
                delay = policy.delay_for(attempt, self.rng)
                self.sleep_fn(delay)
                self._record(log, run_id, EventType.STEP_STARTED,
                             {"step": step.name, "attempt": attempt})
            try:
                return step.fn(ctx)
            except PauseWorkflow:
                raise  # pausing is not a failure — never retry it
            except Exception as exc:  # noqa: BLE001 — retried per policy
                last_error = exc
                self._record(log, run_id, EventType.STEP_FAILED,
                             {"step": step.name, "attempt": attempt,
                              "error": str(exc)})
        self._record(log, run_id, EventType.STEP_EXHAUSTED,
                     {"step": step.name, "attempts": policy.max_attempts})
        raise last_error  # type: ignore[misc]

    def _pause(
        self,
        log: EventLog,
        run_id: str,
        steps: list[Step],
        state: dict,
        pause: PauseWorkflow,
    ) -> None:
        # Find which step raised: it is the step currently executing. We
        # persist the *index of that step* so resume re-runs it — this time
        # request_approval returns the stored decision instead of raising.
        current = self._current_step_name(log, run_id)
        index = next(i for i, s in enumerate(steps) if s.name == current)
        # A fresh pause supersedes any undelivered decision.
        self._pending_decisions.pop(run_id, None)
        self.store.save_approval(
            run_id=run_id,
            step_name=current,
            question=pause.question,
            context=pause.context,
        )
        self.checkpoint(run_id, step_index=index)
        self._record(log, run_id, EventType.PAUSED,
                     {"step": current, "question": pause.question})
        self.store.update_run_status(run_id, RunStatus.PAUSED.value)

    def _current_step_name(self, log: EventLog, run_id: str) -> str:
        started = [e for e in log.events(run_id)
                   if e.type == EventType.STEP_STARTED.value]
        completed = {e.payload["step"] for e in log.events(run_id)
                     if e.type == EventType.STEP_COMPLETED.value}
        for event in reversed(started):
            if event.payload["step"] not in completed:
                return event.payload["step"]
        raise RuntimeError("could not determine the currently executing step")

    # ------------------------------------------------------------------
    # helpers used by WorkflowContext / HITL
    # ------------------------------------------------------------------
    def checkpoint(self, run_id: str, step_index: Optional[int] = None) -> None:
        # step_index defaults to "resume from scratch" only when called
        # outside _execute_from; _pause always passes it explicitly.
        state = self._live_state(run_id)
        self.store.save_checkpoint(run_id, step_index or 0, state)

    def consume_pending_decision(self, run_id: str) -> Any:
        """Remove and return a pending HITL decision (rarely needed).

        Prefer :meth:`peek_pending_decision`: the engine pops the decision
        itself once the step that consumed it completes successfully, so
        retries of the step are redelivered the same decision instead of
        pausing again.
        """
        return self._pending_decisions.pop(run_id, None)

    def peek_pending_decision(self, run_id: str) -> Any:
        return self._pending_decisions.get(run_id)

    def _live_state(self, run_id: str) -> dict:
        # Best-effort snapshot: rebuild from the log so far. The context's
        # state dict is the same object the engine threads through, but on
        # resume there is no live context — the log is the source of truth.
        return self.replay_state(run_id)

    def _record(
        self, log: EventLog, run_id: str, type: EventType, payload: dict
    ) -> None:
        event = log.append(run_id, type.value, payload)
        self.store.append_event(event)

    def _load_log(self, run_id: str) -> EventLog:
        log = EventLog()
        for event in self.store.get_events(run_id):
            # Re-sequence in storage order; EventLog.append assigns seq.
            log.append(event.run_id, event.type, event.payload)
        return log
