"""Command-line interface for agentforge.

Commands
--------
run <workflow>     Execute a registered example workflow.
runs               List recorded runs.
approvals          List pending human-in-the-loop approvals.
approve <id>       Resolve an approval and resume its workflow.
"""

from __future__ import annotations

import json
import sys

import click

from agentforge.engine import RunStatus, WorkflowEngine
from agentforge.hitl import make_decision
from agentforge.store import RunStore


def _engine_for(store_path: str) -> WorkflowEngine:
    from examples import build_all  # deferred: examples need the package

    engine = WorkflowEngine(store=RunStore(store_path))
    for name, steps in build_all().items():
        engine.register_workflow(name, steps)
    return engine


@click.group()
@click.option("--store", "store_path", default="agentforge.db",
              help="SQLite store path (default: agentforge.db)")
@click.pass_context
def main(ctx: click.Context, store_path: str) -> None:
    ctx.ensure_object(dict)
    ctx.obj["store_path"] = store_path


@main.command()
@click.argument("workflow")
@click.pass_context
def run(ctx: click.Context, workflow: str) -> None:
    """Run a workflow by name (e.g. ``agentforge run research``)."""
    engine = _engine_for(ctx.obj["store_path"])
    try:
        result = engine.run(workflow)
    except ValueError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(2)
    click.echo(f"run_id:  {result.run_id}")
    click.echo(f"status:  {result.status.value}")
    if result.status == RunStatus.PAUSED:
        click.echo("The workflow is paused waiting for human approval.")
        click.echo("Use `agentforge approvals` to review it.")
    elif result.status == RunStatus.FAILED:
        click.echo(f"error:   {result.error}", err=True)
        sys.exit(1)
    else:
        click.echo("final state:")
        click.echo(json.dumps(result.state, indent=2, default=str))


@main.command(name="list")
@click.pass_context
def list_runs(ctx: click.Context) -> None:
    """List recorded workflow runs."""
    store = RunStore(ctx.obj["store_path"])
    runs = store.list_runs()
    if not runs:
        click.echo("no runs recorded yet")
        return
    for r in runs:
        click.echo(f"{r['id']}  {r['name']:<12} {r['status']}")


@main.command()
@click.pass_context
def approvals(ctx: click.Context) -> None:
    """List pending human-in-the-loop approvals."""
    from agentforge.hitl import ApprovalRequest

    store = RunStore(ctx.obj["store_path"])
    pending = store.pending_approvals()
    if not pending:
        click.echo("no pending approvals")
        return
    for item in pending:
        req = ApprovalRequest(**{k: v for k, v in item.items()
                                 if k in ApprovalRequest.__dataclass_fields__})
        click.echo(req.summarize())
        click.echo()


@main.command()
@click.argument("approval_id", type=int)
@click.option("--reject", is_flag=True, help="reject instead of approve")
@click.option("--edits", default="{}",
              help='JSON edits for an "edit" verdict, e.g. \'{"x": 1}\'')
@click.pass_context
def approve(ctx: click.Context, approval_id: int, reject: bool,
            edits: str) -> None:
    """Resolve an approval and resume its workflow."""
    store_path = ctx.obj["store_path"]
    store = RunStore(store_path)
    pending = {a["id"]: a for a in store.pending_approvals()}
    if approval_id not in pending:
        click.echo(f"error: no pending approval #{approval_id}", err=True)
        sys.exit(2)
    item = pending[approval_id]
    try:
        edits_obj = json.loads(edits)
    except json.JSONDecodeError as exc:
        click.echo(f"error: --edits is not valid JSON: {exc}", err=True)
        sys.exit(2)

    verdict = "reject" if reject else ("edit" if edits_obj else "approve")
    decision = make_decision(verdict, edits_obj)
    store.resolve_approval(approval_id, decision)

    engine = _engine_for(store_path)
    result = engine.resume(item["run_id"], decision)
    click.echo(f"run_id: {result.run_id}  status: {result.status.value}")
    if result.status == RunStatus.FAILED:
        click.echo(f"error: {result.error}", err=True)
        sys.exit(1)
    click.echo(json.dumps(result.state, indent=2, default=str))


if __name__ == "__main__":
    main()
