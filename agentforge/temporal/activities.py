"""Activities for the Temporal agent workflow.

Activities are where side effects live: planning, tool execution through a
:class:`~agentforge.tools.ToolRegistry`, and bookkeeping through a
:class:`~agentforge.store.RunStore`. The workflow itself stays pure and
replayable; everything in this module runs in normal (unsandboxed) code.

Planning
--------
The default planner is a deterministic heuristic: it matches task keywords
against the registry (via :meth:`ToolRegistry.search`) and fills required
arguments from the task text. It is deliberately simple — the seam is the
``planner`` callable accepted by :class:`AgentActivities`, where an
LLM-backed planner (e.g. one driving :class:`agentforge.agent.Agent`)
plugs in without touching the workflow.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional

from temporalio import activity
from temporalio.exceptions import ApplicationError

from agentforge.engine import Event, EventType
from agentforge.store import RunStore
from agentforge.tools import ToolRegistry, ToolValidationError
from agentforge.temporal.models import (
    ExecuteToolInput,
    FinishRunInput,
    PlanStepsInput,
    PlannedStep,
    RecordRunStartedInput,
    RequestApprovalInput,
    ResolveApprovalInput,
    ToolExecutionResult,
)

#: Planner protocol: task text + registry -> ordered tool calls.
Planner = Callable[[str, ToolRegistry, int], list[PlannedStep]]

_MAX_OBSERVATION_CHARS = 4000
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "with",
    "please", "then", "this", "that", "it", "its", "is", "are", "be",
}


def _fill_arguments(tool_obj: Any, task: str) -> dict[str, Any]:
    """Fill a tool's required arguments with plausible values from the task.

    Strings get the full task text (most agent tools take one free-text
    input); numbers/bools get schema defaults when present, else neutral
    values. Optional parameters keep their schema defaults by omission.
    """
    schema = tool_obj.parameters
    props: dict[str, Any] = schema.get("properties", {})
    args: dict[str, Any] = {}
    for name in schema.get("required", []):
        prop = props.get(name, {})
        if "default" in prop:
            args[name] = prop["default"]
        elif prop.get("type") == "string" or not prop.get("type"):
            args[name] = task
        elif prop.get("type") == "integer":
            args[name] = 1
        elif prop.get("type") == "number":
            args[name] = 1.0
        elif prop.get("type") == "boolean":
            args[name] = True
        elif prop.get("type") == "array":
            args[name] = []
        elif prop.get("type") == "object":
            args[name] = {}
    return args


def heuristic_planner(
    task: str, registry: ToolRegistry, max_steps: int
) -> list[PlannedStep]:
    """Keyword planner: one step per tool the task plausibly calls for.

    Each content word of the task is searched against tool names and
    descriptions; matching tools are planned once, in registration order,
    with arguments filled by :func:`_fill_arguments`. If nothing matches,
    the plan is empty and the run completes with no steps.
    """
    words = [
        w
        for w in re.findall(r"[a-z0-9]+", task.lower())
        if w not in _STOPWORDS and len(w) > 2
    ]
    matched: list[str] = []
    for word in words:
        for tool_obj in registry.search(word):
            if tool_obj.name not in matched:
                matched.append(tool_obj.name)
    plan: list[PlannedStep] = []
    for name in registry.names():  # stable registration order
        if name in matched:
            tool_obj = registry.get(name)
            plan.append(
                PlannedStep(
                    tool=name,
                    arguments=_fill_arguments(tool_obj, task),
                    reason=f"task keywords matched tool '{name}'",
                )
            )
        if len(plan) >= max_steps:
            break
    return plan


class AgentActivities:
    """Activity implementations bound to one registry + store pair.

    A worker constructs one instance with its own registry and store and
    registers the methods; tests can construct it directly with an
    in-memory store and a registry of deterministic tools.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        store: RunStore,
        planner: Optional[Planner] = None,
    ) -> None:
        self._registry = registry
        self._store = store
        self._planner = planner or heuristic_planner

    # ------------------------------------------------------------------
    # bookkeeping
    # ------------------------------------------------------------------
    def _event(self, run_id: str, type_: str, payload: dict) -> None:
        self._store.append_event(
            Event(seq=0, run_id=run_id, type=type_, payload=payload)
        )

    @activity.defn
    async def record_run_started(self, input: RecordRunStartedInput) -> None:
        if not self._store.has_run(input.run_id):
            self._store.create_run(input.run_id, input.name)
        self._store.update_run_status(input.run_id, "running")
        self._event(
            input.run_id,
            EventType.WORKFLOW_STARTED.value,
            {"task": input.task, "engine": "temporal"},
        )

    @activity.defn
    async def finish_run(self, input: FinishRunInput) -> None:
        status = {
            "completed": "completed",
            "failed": "failed",
            "rejected": "failed",
        }.get(input.status, input.status)
        self._store.update_run_status(input.run_id, status)
        self._store.save_checkpoint(
            input.run_id,
            input.steps_completed,
            {"status": input.status, "answer": input.answer},
        )
        event_type = (
            EventType.WORKFLOW_COMPLETED.value
            if input.status == "completed"
            else EventType.WORKFLOW_FAILED.value
        )
        self._event(
            input.run_id, event_type,
            {"status": input.status, "answer": input.answer},
        )

    # ------------------------------------------------------------------
    # planning
    # ------------------------------------------------------------------
    @activity.defn
    async def plan_steps(self, input: PlanStepsInput) -> list[PlannedStep]:
        plan = self._planner(input.task, self._registry, input.max_steps)
        self._event(
            input.run_id,
            "plan_created",
            {"steps": [s.tool for s in plan], "count": len(plan)},
        )
        return plan

    # ------------------------------------------------------------------
    # tool execution
    # ------------------------------------------------------------------
    @activity.defn
    async def execute_tool(self, input: ExecuteToolInput) -> ToolExecutionResult:
        """Execute one tool call through the registry.

        Validation failures are deterministic — retrying cannot fix bad
        arguments — so they raise a non-retryable :class:`ApplicationError`.
        Any other exception propagates and is retried by the workflow's
        retry policy (flaky tools get their backoff).
        """
        self._event(
            input.run_id,
            EventType.STEP_STARTED.value,
            {"step": input.step_index, "tool": input.tool},
        )
        try:
            result = self._registry.call(input.tool, dict(input.arguments))
        except ToolValidationError as exc:
            raise ApplicationError(
                str(exc), type="ToolValidationError", non_retryable=True
            ) from exc
        except KeyError as exc:
            raise ApplicationError(
                str(exc), type="UnknownTool", non_retryable=True
            ) from exc
        observation = (
            result if isinstance(result, str) else json.dumps(result, default=str)
        )
        truncated = len(observation) > _MAX_OBSERVATION_CHARS
        if truncated:
            observation = observation[:_MAX_OBSERVATION_CHARS]
        self._event(
            input.run_id,
            EventType.STEP_COMPLETED.value,
            {"step": input.step_index, "tool": input.tool},
        )
        return ToolExecutionResult(observation=observation, truncated=truncated)

    # ------------------------------------------------------------------
    # human approval bookkeeping (the wait itself is a workflow signal)
    # ------------------------------------------------------------------
    @activity.defn
    async def request_approval(self, input: RequestApprovalInput) -> int:
        question = (
            f"Approve tool call '{input.tool}' with arguments "
            f"{json.dumps(input.arguments, default=str)}?"
        )
        approval_id = self._store.save_approval(
            input.run_id,
            step_name=input.tool,
            question=question,
            context={"step": input.step_index, "arguments": input.arguments},
        )
        self._store.update_run_status(input.run_id, "awaiting_approval")
        self._event(
            input.run_id,
            EventType.PAUSED.value,
            {"step": input.step_index, "tool": input.tool,
             "approval_id": approval_id},
        )
        return approval_id

    @activity.defn
    async def resolve_approval(self, input: ResolveApprovalInput) -> None:
        self._store.resolve_approval(input.approval_id, input.decision)
        self._store.update_run_status(input.run_id, "running")
        self._event(
            input.run_id,
            EventType.RESUMED.value,
            {"approval_id": input.approval_id,
             "verdict": input.decision.get("verdict")},
        )

    def all(self) -> list:
        """Every activity callable, for one-shot worker registration."""
        return [
            self.record_run_started,
            self.finish_run,
            self.plan_steps,
            self.execute_tool,
            self.request_approval,
            self.resolve_approval,
        ]
