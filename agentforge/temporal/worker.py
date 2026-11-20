"""Worker + demo client for the Temporal agent workflow.

Run a local Temporal server first (from the repo root)::

    docker compose up -d          # temporalio/auto-setup on :7233

then, in one terminal::

    python -m agentforge.temporal.worker worker

and in another::

    python -m agentforge.temporal.worker demo \
        --task "count the words in this sentence" --approve-tool word_count

The demo starts a run, polls the ``run_state`` query, auto-approves any
approval gate (a real deployment would put a human in that loop), and
prints the terminal result. The demo tools are small deterministic text
utilities registered through the standard :class:`ToolRegistry`, so the
whole path — plan, approval, tool execution, RunStore bookkeeping — is
exercised end to end with no model API keys.
"""

from __future__ import annotations

import argparse
import asyncio
import uuid

from temporalio.client import Client
from temporalio.worker import Worker

from agentforge.store import RunStore
from agentforge.temporal.activities import AgentActivities
from agentforge.temporal.models import AgentRunInput, ApprovalSignalPayload
from agentforge.temporal.workflows import AgentRunWorkflow
from agentforge.tools import ToolRegistry, tool

TASK_QUEUE = "agentforge-agent-runs"


# ----------------------------------------------------------------------
# demo tools (deterministic text utilities)
# ----------------------------------------------------------------------
@tool(description="Count the words in a block of text.")
def word_count(text: str) -> dict:
    words = text.split()
    return {"words": len(words), "characters": len(text)}


@tool(description="Find the most frequent words in a block of text.")
def top_words(text: str, top_k: int = 3) -> dict:
    counts: dict[str, int] = {}
    for word in text.lower().split():
        counts[word] = counts.get(word, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return {"top": ranked[:top_k], "unique": len(counts)}


@tool(description="Find the longest word in a block of text.")
def longest_word(text: str) -> dict:
    words = text.split()
    longest = max(words, key=len) if words else ""
    return {"longest": longest, "length": len(longest)}


def build_demo_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for fn in (word_count, top_words, longest_word):
        registry.register(fn)
    return registry


def build_worker(
    client: Client,
    store: RunStore,
    registry: ToolRegistry,
    task_queue: str = TASK_QUEUE,
) -> Worker:
    activities = AgentActivities(registry=registry, store=store)
    return Worker(
        client,
        task_queue=task_queue,
        workflows=[AgentRunWorkflow],
        activities=activities.all(),
    )


async def _run_worker(address: str, namespace: str, store_path: str) -> None:
    client = await Client.connect(address, namespace=namespace)
    store = RunStore(store_path)
    worker = build_worker(client, store, build_demo_registry())
    print(f"agentforge worker listening on task queue '{TASK_QUEUE}'")
    await worker.run()


async def _run_demo(args: argparse.Namespace) -> None:
    client = await Client.connect(args.address, namespace=args.namespace)
    store = RunStore(args.store)
    worker = build_worker(client, store, build_demo_registry())
    async with worker:
        run_id = f"demo-{uuid.uuid4().hex[:8]}"
        handle = await client.start_workflow(
            AgentRunWorkflow.run,
            AgentRunInput(
                run_id=run_id,
                task=args.task,
                approval_tools=args.approve_tool,
                run_name="demo",
            ),
            id=run_id,
            task_queue=TASK_QUEUE,
        )
        print(f"started run {run_id}")
        while True:
            state = await handle.query(AgentRunWorkflow.run_state)
            print(f"  state: {state.status} "
                  f"({state.completed_steps}/{state.total_steps} steps)")
            if state.awaiting_approval:
                print(f"  approving gate: {state.pending_question}")
                await handle.signal(
                    AgentRunWorkflow.approve,
                    ApprovalSignalPayload(verdict="approve", by="demo"),
                )
            try:
                result = await asyncio.wait_for(
                    asyncio.shield(handle.result()), timeout=1.0
                )
                break
            except asyncio.TimeoutError:
                continue
        print(f"finished: {result.status} — {result.answer}")
        for step in result.steps:
            print(f"  [{step.index}] {step.tool}: {step.status} "
                  f"-> {step.observation[:120]}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="agentforge-temporal")
    parser.add_argument("--address", default="localhost:7233")
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--store", default="agentforge-runs.db",
                        help="SQLite path for the RunStore")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("worker", help="run the worker in the foreground")

    demo = sub.add_parser("demo", help="run a worker + one demo run")
    demo.add_argument("--task", default="count the words in this sentence")
    demo.add_argument("--approve-tool", action="append", default=[],
                      help="tool name that requires approval (repeatable)")

    args = parser.parse_args()
    if args.command == "worker":
        asyncio.run(_run_worker(args.address, args.namespace, args.store))
    else:
        asyncio.run(_run_demo(args))


if __name__ == "__main__":
    main()
