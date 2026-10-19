"""Tests for human-in-the-loop pause/resume and the approval store."""

import pytest

from agentforge.engine import RunStatus, Step, WorkflowEngine
from agentforge.hitl import make_decision
from agentforge.store import RunStore


def make_engine(**kwargs):
    kwargs.setdefault("sleep_fn", lambda s: None)
    kwargs.setdefault("store", RunStore())
    return WorkflowEngine(**kwargs)


def gated_workflow():
    def prepare(ctx):
        return {"draft": "deploy v2"}

    def gate(ctx):
        decision = ctx.request_approval("Ship it?",
                                        context={"draft": "deploy v2"})
        if decision.get("verdict") == "reject":
            raise RuntimeError("rejected by reviewer")
        return {"verdict": decision.get("verdict")}

    def ship(ctx):
        return {"shipped": ctx.state["draft"]}

    return [Step("prepare", prepare), Step("gate", gate), Step("ship", ship)]


def test_pause_creates_approval_and_checkpoint():
    engine = make_engine()
    engine.register_workflow("deploy", gated_workflow())
    result = engine.run("deploy")

    assert result.status == RunStatus.PAUSED
    assert result.state["draft"] == "deploy v2"

    approvals = engine.store.pending_approvals(result.run_id)
    assert len(approvals) == 1
    assert approvals[0]["question"] == "Ship it?"
    assert approvals[0]["step_name"] == "gate"

    checkpoint = engine.store.load_checkpoint(result.run_id)
    assert checkpoint is not None
    assert checkpoint["state"]["draft"] == "deploy v2"


def test_resume_with_approve_completes():
    engine = make_engine()
    engine.register_workflow("deploy", gated_workflow())
    paused = engine.run("deploy")
    assert paused.status == RunStatus.PAUSED

    approval = engine.store.pending_approvals(paused.run_id)[0]
    decision = make_decision("approve", by="qa-lead")
    engine.store.resolve_approval(approval["id"], decision)

    result = engine.resume(paused.run_id, decision)
    assert result.status == RunStatus.COMPLETED
    assert result.state["verdict"] == "approve"
    assert result.state["shipped"] == "deploy v2"
    assert engine.store.pending_approvals(paused.run_id) == []


def test_resume_with_reject_fails_run():
    engine = make_engine()
    engine.register_workflow("deploy", gated_workflow())
    paused = engine.run("deploy")

    result = engine.resume(paused.run_id, make_decision("reject"))
    assert result.status == RunStatus.FAILED
    assert "rejected by reviewer" in result.error


def test_resume_with_edit_passes_edits_through():
    seen = {}

    def gate(ctx):
        decision = ctx.request_approval("ok?", context={})
        seen.update(decision.get("edits", {}))
        return {}

    engine = make_engine()
    engine.register_workflow("w", [Step("gate", gate)])
    paused = engine.run("w")
    engine.resume(paused.run_id,
                  make_decision("edit", edits={"image": "v3"}, by="sam"))
    assert seen == {"image": "v3"}


def test_resume_unknown_run_raises():
    engine = make_engine()
    with pytest.raises(ValueError, match="unknown run_id"):
        engine.resume("does-not-exist")


def test_resume_non_paused_run_raises():
    engine = make_engine()
    engine.register_workflow("w", [Step("a", lambda ctx: {})])
    result = engine.run("w")
    assert result.status == RunStatus.COMPLETED
    with pytest.raises(ValueError, match="not paused"):
        engine.resume(result.run_id)


def test_pause_survives_process_restart(tmp_path):
    """A paused run resumes on a brand-new engine over the same DB file."""
    db = str(tmp_path / "hitl.db")

    engine = WorkflowEngine(store=RunStore(db), sleep_fn=lambda s: None)
    engine.register_workflow("deploy", gated_workflow())
    paused = engine.run("deploy")
    assert paused.status == RunStatus.PAUSED

    # Simulate a restart: new engine, same store file.
    engine2 = WorkflowEngine(store=RunStore(db), sleep_fn=lambda s: None)
    engine2.register_workflow("deploy", gated_workflow())
    result = engine2.resume(paused.run_id, make_decision("approve"))
    assert result.status == RunStatus.COMPLETED
    assert result.state["shipped"] == "deploy v2"


def test_make_decision_rejects_bad_verdict():
    with pytest.raises(ValueError, match="unknown verdict"):
        make_decision("maybe")


def test_approval_summarize():
    from agentforge.hitl import ApprovalRequest

    req = ApprovalRequest(id=7, run_id="abc", step_name="gate",
                          question="Ship it?",
                          context={"draft": "v2"})
    text = req.summarize()
    assert "Approval #7" in text
    assert "Ship it?" in text
    assert "pending" in text
