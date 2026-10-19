"""Human-in-the-loop approval gates.

A workflow step calls :func:`request_approval` (or
``ctx.request_approval``) when it wants a human to review a proposed
action before it executes. The engine pauses the run, persists a
checkpoint plus an approval record, and returns ``RunStatus.PAUSED``.
Later — from a CLI, a Slack bot, a web UI — a human resolves the
approval and the run resumes exactly where it stopped.

Decisions are one of:

- ``approve`` — continue with the proposed action unchanged.
- ``reject`` — abort the paused step; the workflow fails with a clear error.
- ``edit`` — continue, but with modified parameters supplied by the human.

Example::

    def deploy_step(ctx):
        plan = build_deploy_plan(ctx.state["service"])
        decision = ctx.request_approval(
            f"Deploy {plan['image']} to prod?", context={"plan": plan})
        if decision["verdict"] == "reject":
            raise RuntimeError("deploy rejected by human reviewer")
        plan.update(decision.get("edits", {}))
        return {"deployed": do_deploy(plan)}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ApprovalDecision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    EDIT = "edit"


@dataclass
class ApprovalRequest:
    """A human-readable approval request created by a paused workflow."""

    id: int
    run_id: str
    step_name: str
    question: str
    context: dict = field(default_factory=dict)
    status: str = "pending"
    decision: Optional[dict] = None

    def summarize(self) -> str:
        lines = [
            f"Approval #{self.id} — run {self.run_id} (step '{self.step_name}')",
            f"  Question: {self.question}",
        ]
        if self.context:
            lines.append(f"  Context: {self.context}")
        lines.append(f"  Status: {self.status}")
        if self.decision:
            lines.append(f"  Decision: {self.decision}")
        return "\n".join(lines)


def make_decision(
    verdict: str,
    edits: Optional[dict] = None,
    by: str = "human",
    note: str = "",
) -> dict:
    """Build a well-formed decision payload for ``engine.resume``."""
    if verdict not in ("approve", "reject", "edit"):
        raise ValueError(f"unknown verdict: {verdict!r}")
    return {"verdict": verdict, "edits": edits or {}, "by": by, "note": note}
