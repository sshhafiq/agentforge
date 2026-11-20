"""Serializable data model for the Temporal agent workflow.

Everything here is a plain dataclass with JSON-friendly field types so the
Temporal Python SDK's default converter can carry it across workflow /
activity boundaries. No agentforge imports: this module is safe to load
inside the workflow sandbox.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class PlannedStep:
    """One tool call in a plan, produced by the planning activity."""

    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass
class AgentRunInput:
    """Input to :class:`AgentRunWorkflow`."""

    run_id: str
    task: str
    #: Tool names that must be approved by a human (via the ``approve``
    #: signal) before they execute. Mirrors the engine's HITL gate.
    approval_tools: list[str] = field(default_factory=list)
    #: Hard cap on executed steps, regardless of plan length.
    max_steps: int = 16
    #: Human-readable run name recorded in the RunStore.
    run_name: str = "agent-run"


@dataclass
class StepRecord:
    """Outcome of one executed (or skipped) plan step."""

    index: int
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    status: str = "completed"  # completed | failed | rejected | edited
    observation: str = ""
    error: Optional[str] = None


@dataclass
class AgentRunResult:
    """Terminal result of an agent run."""

    run_id: str
    status: str  # completed | failed | rejected
    steps: list[StepRecord] = field(default_factory=list)
    answer: str = ""


@dataclass
class ApprovalSignalPayload:
    """Payload of the ``approve`` signal.

    Mirrors :func:`agentforge.hitl.make_decision`'s decision dict:
    ``verdict`` is one of ``approve`` / ``reject`` / ``edit``; ``edits``
    overrides tool arguments when the verdict is ``edit``.
    """

    verdict: str
    edits: dict[str, Any] = field(default_factory=dict)
    by: str = "human"
    note: str = ""

    def to_decision(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "edits": dict(self.edits),
            "by": self.by,
            "note": self.note,
        }


@dataclass
class RunState:
    """Live run state, returned by the workflow's ``run_state`` query."""

    run_id: str
    status: str  # planning | running | awaiting_approval | done
    total_steps: int = 0
    completed_steps: int = 0
    current_tool: Optional[str] = None
    awaiting_approval: bool = False
    pending_question: Optional[str] = None


# ----------------------------------------------------------------------
# Activity argument bundles (one dataclass per activity keeps the
# workflow code readable and the payloads self-describing in the UI).
# ----------------------------------------------------------------------


@dataclass
class RecordRunStartedInput:
    run_id: str
    name: str
    task: str


@dataclass
class PlanStepsInput:
    run_id: str
    task: str
    max_steps: int = 16


@dataclass
class ExecuteToolInput:
    run_id: str
    step_index: int
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolExecutionResult:
    observation: str
    truncated: bool = False


@dataclass
class RequestApprovalInput:
    run_id: str
    step_index: int
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResolveApprovalInput:
    run_id: str
    approval_id: int
    decision: dict[str, Any] = field(default_factory=dict)


@dataclass
class FinishRunInput:
    run_id: str
    status: str
    answer: str = ""
    steps_completed: int = 0
