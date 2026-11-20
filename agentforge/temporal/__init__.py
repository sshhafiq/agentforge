"""Temporal integration: run agentforge workflows as durable Temporal workflows.

The pure-Python engine in :mod:`agentforge.engine` gives durable execution
inside one process. This subpackage maps the same concepts onto a real
Temporal cluster:

- :class:`~agentforge.temporal.workflows.AgentRunWorkflow` — the durable
  orchestration: plan, optionally pause for human approval (signal), then
  execute each tool call as an activity with its own retry policy.
- :class:`~agentforge.temporal.activities.AgentActivities` — the side-effecting
  pieces: planning, tool execution through a :class:`ToolRegistry`, and run
  bookkeeping through a :class:`RunStore`.
- :mod:`agentforge.temporal.worker` — a runnable worker + demo client.

Requires the ``temporal`` extra: ``pip install agentforge[temporal]``.
"""

from agentforge.temporal.models import (
    AgentRunInput,
    AgentRunResult,
    ApprovalSignalPayload,
    PlannedStep,
    RunState,
    StepRecord,
)

__all__ = [
    "AgentRunInput",
    "AgentRunResult",
    "ApprovalSignalPayload",
    "PlannedStep",
    "RunState",
    "StepRecord",
]
