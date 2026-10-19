"""agentforge: durable AI agent workflows with Temporal-style orchestration.

A lightweight, dependency-free (stdlib + click) framework for building
long-running, fault-tolerant AI agent workflows:

- **Durable execution** — every step is recorded in an append-only event log,
  so a crashed or paused workflow can resume exactly where it left off.
- **Retries** — per-step retry policies with exponential backoff and jitter.
- **Replay** — rebuild workflow state purely from the event log, without
  re-executing side effects.
- **Human-in-the-loop** — approval gates pause a workflow until a human
  approves, rejects, or edits the proposed action.
- **ReAct agents** — a model-agnostic agent loop with pluggable LLM backends
  and JSON-schema-validated tools.
"""

from agentforge.engine import (
    Event,
    EventLog,
    EventType,
    PauseWorkflow,
    RetryPolicy,
    RunResult,
    RunStatus,
    Step,
    WorkflowContext,
    WorkflowEngine,
)
from agentforge.agent import AgentResult, AgentStep, MockModelClient, ModelClient, ReActAgent
from agentforge.tools import Tool, ToolRegistry, ToolValidationError, tool
from agentforge.store import RunStore
from agentforge.hitl import ApprovalDecision, ApprovalRequest

__all__ = [
    "AgentResult",
    "AgentStep",
    "ApprovalDecision",
    "ApprovalRequest",
    "Event",
    "EventLog",
    "EventType",
    "MockModelClient",
    "ModelClient",
    "PauseWorkflow",
    "ReActAgent",
    "RetryPolicy",
    "RunResult",
    "RunStatus",
    "RunStore",
    "Step",
    "Tool",
    "ToolRegistry",
    "ToolValidationError",
    "WorkflowContext",
    "WorkflowEngine",
    "tool",
]

__version__ = "0.1.0"
