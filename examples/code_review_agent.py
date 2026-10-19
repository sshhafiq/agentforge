"""Example: code-review agent workflow.

Reviews a diff in three durable steps: load the diff, run a ReAct agent
with lint/test tools over it, and file the findings. A human-in-the-loop
gate pauses before findings are posted, so nothing ships without review.

Fully offline: the diff and tool backends are canned data, the model is
:class:`MockModelClient`.
"""

from __future__ import annotations

from agentforge.agent import MockModelClient, ReActAgent
from agentforge.engine import Step, WorkflowContext
from agentforge.tools import ToolRegistry, tool

SAMPLE_DIFF = """\
diff --git a/payments.py b/payments.py
--- a/payments.py
+++ b/payments.py
@@ -12,6 +12,12 @@
 def charge(card_token, amount_cents):
+    # TODO: retry on network errors
     resp = gateway.charge(card_token, amount_cents)
+    if resp.status == "declined":
+        log("card declined")
     return resp
+
+def refund(charge_id, amount_cents):
+    return gateway.refund(charge_id, amount_cents)
"""


@tool(description="Return the diff under review.")
def get_diff() -> str:
    """Return the diff under review."""
    return SAMPLE_DIFF


@tool(description="Run the linter on a file and return findings.")
def lint(file: str) -> str:
    """Run the linter on a file and return findings."""
    return (
        "payments.py:15: TODO comment without ticket reference; "
        "payments.py:21: 'refund' missing error handling for gateway errors"
    )


@tool(description="Run the test suite for a file and return the result.")
def run_tests(file: str) -> str:
    """Run the test suite for a file and return the result."""
    return "3 passed, 1 failed: test_refund_partial_amount (KeyError)"


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    for fn in (get_diff, lint, run_tests):
        registry.register(fn)
    return registry


def load_diff(ctx: WorkflowContext) -> dict:
    return {"diff": SAMPLE_DIFF, "files": ["payments.py"]}


def analyze(ctx: WorkflowContext) -> dict:
    script = [
        "Thought: I need the diff first.\nAction: get_diff {}",
        "Thought: Now lint the changed file.\n"
        'Action: lint {"file": "payments.py"}',
        "Thought: And run the tests.\n"
        'Action: run_tests {"file": "payments.py"}',
    ]
    agent = ReActAgent(MockModelClient(script), _registry(), max_iterations=8)
    result = agent.run("Review the diff in payments.py for bugs and style issues.")
    return {
        "review_notes": result.answer,
        "tool_calls": len(result.steps),
    }


def file_findings(ctx: WorkflowContext) -> dict:
    notes = ctx.state["review_notes"]
    return {
        "findings": (
            "## Code review: payments.py\n\n"
            f"{notes}\n\n"
            "- Missing retry logic on gateway.charge (see TODO)\n"
            "- refund() needs error handling\n"
        )
    }


def post_gate(ctx: WorkflowContext) -> dict:
    decision = ctx.request_approval(
        "Post these review findings to the PR?",
        context={"findings": ctx.state["findings"]},
    )
    if decision.get("verdict") == "reject":
        raise RuntimeError("review rejected by human")
    return {"posted": True}


def build_code_review_workflow() -> list[Step]:
    return [
        Step("load_diff", load_diff),
        Step("analyze", analyze, checkpoint_after=True),
        Step("file_findings", file_findings),
        Step("post_gate", post_gate),
    ]
