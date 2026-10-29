# agentforge

Durable AI agent workflows with Temporal-style orchestration — in pure Python,
with zero required dependencies beyond `click`.

`agentforge` is a small framework for building **long-running, fault-tolerant
agent workflows**: multi-step LLM processes that survive crashes, retry flaky
steps with backoff, pause for human approval, and can be replayed from their
event history without re-executing side effects.

## Why

Most agent demos are happy-path scripts: if step 3 of 8 fails, you start over.
Real agent systems (customer support, code review, research) run for minutes
or hours and touch the real world — they need **durable execution**:

- every step is recorded in an append-only **event log**,
- a crashed process **resumes** from the last completed step,
- flaky steps **retry** with exponential backoff + jitter,
- risky actions pause for **human approval** before executing.

That's the model Temporal popularized. `agentforge` brings it to Python agent
workflows in ~1,500 lines you can read in an afternoon.

## Architecture

```
                        +-------------------+
                        |   CLI / your app  |
                        +--------+----------+
                                 |
                    +------------v-------------+
                    |     WorkflowEngine     |
                    |  - runs ordered Steps  |
                    |  - retries w/ backoff  |
                    |  - pause/resume (HITL) |
                    +---+--------------+-----+
                        |              |
            +-----------v----+  +------v-----------+
            |    EventLog    |  |  WorkflowContext |
            | append-only    |  |  shared state    |
            | history of the |  |  request_approval|
            | run            |  |  checkpoint()    |
            +-----------+----+  +------------------+
                        |
               +--------v---------+
               |    RunStore      |
               |  SQLite: runs,   |
               |  events,         |
               |  checkpoints,    |
               |  approvals       |
               +------------------+

  +----------------+      +----------------+      +--------------+
  |  ReActAgent    |----->| ToolRegistry   |----->| ModelClient  |
  | Thought/Action/|      | @tool decorator|      | (Protocol)   |
  | Observation    |      | JSON-schema    |      | Mock or real |
  | loop           |      | validated calls|      | LLM backend  |
  +----------------+      +----------------+      +--------------+
```

**Event sourcing.** The engine never mutates run state in place — it appends
events (`workflow_started`, `step_started`, `step_completed`, `paused`, …).
State is a fold over events, so `engine.replay_state(run_id)` rebuilds it
without re-running a single step function. Checkpoints are just named
snapshots of that fold, used to resume after pause/crash.

**Retries.** Each `Step` carries a `RetryPolicy` (max attempts, base delay,
cap, jitter). `PauseWorkflow` is never retried — pausing isn't failure.

**HITL.** `ctx.request_approval(question)` raises `PauseWorkflow`; the engine
persists a checkpoint + approval row and returns `PAUSED`. Resolving the
approval (CLI, bot, UI) calls `engine.resume(run_id, decision)`; the paused
step re-runs and `request_approval` returns the decision dict instead.

## Quickstart

```bash
pip install -e ".[dev]"   # or: pip install agentforge
pytest                      # 49 tests, all offline
```

Define a workflow:

```python
from agentforge import Step, WorkflowContext, WorkflowEngine, tool, ToolRegistry
from agentforge.agent import MockModelClient, ReActAgent

@tool(description="Search the docs.")
def search_docs(query: str) -> str:
    return f"docs for {query}"

def research(ctx: WorkflowContext):
    agent = ReActAgent(MockModelClient([...scripted replies...]), registry)
    return {"answer": agent.run("summarize the docs").answer}

def review_gate(ctx: WorkflowContext):
    decision = ctx.request_approval("Publish the summary?")
    if decision["verdict"] == "reject":
        raise RuntimeError("rejected by reviewer")
    return {"published": True}

engine = WorkflowEngine()
engine.register_workflow("research", [
    Step("research", research, checkpoint_after=True),
    Step("review_gate", review_gate),
])

result = engine.run("research")
if result.status == "paused":
    # ... human reviews in the CLI: agentforge approvals / agentforge approve 1
    result = engine.resume(result.run_id, {"verdict": "approve"})
```

Or run the bundled examples:

```bash
agentforge run research        # pauses at a publish approval gate
agentforge approvals           # list pending approvals
agentforge approve 1           # approve & resume to completion
agentforge run code-review     # diff review agent with a post gate
agentforge list                # list recorded runs
```

## Persistence

Runs, events, checkpoints, and approvals live in a SQLite store, so a
paused workflow survives a process restart: re-run the same CLI command
and `agentforge approve` resumes it from the stored checkpoint. The CLI
defaults to `agentforge.db` in the working directory; pass `--store` to
point it at a different path (e.g. `agentforge --store /tmp/runs.db run
research`).

## Example output

```
$ agentforge run research
run_id:  472320666615
status:  paused
The workflow is paused waiting for human approval.
Use `agentforge approvals` to review it.

$ agentforge approvals
Approval #1 — run 472320666615 (step 'publish_gate')
  Question: Publish this research briefing?
  Context: {'briefing': '# Research briefing: temporal workflows\n\n- ...'}
  Status: pending

$ agentforge approve 1
run_id: 472320666615  status: completed
{
  "topic": "temporal workflows",
  "briefing": "# Research briefing: temporal workflows\n\n- ...",
  "published": true,
  "verdict": "approve"
}
```

## Project layout

```
agentforge/          # the framework
  engine.py          # durable execution: event log, retries, replay, resume
  agent.py           # ReAct loop + ModelClient protocol + MockModelClient
  tools.py           # @tool decorator, JSON-schema inference & validation
  store.py           # SQLite run store (runs/events/checkpoints/approvals)
  hitl.py            # approval-gate types & decision helpers
  cli.py             # `agentforge run|list|approvals|approve`
examples/
  research_agent.py  # 4-step research workflow with publish gate
  code_review_agent.py  # diff review workflow with post gate
tests/               # 49 pytest tests — engine, agent, tools, HITL, store
```

## Design decisions worth knowing

- **No ORM, no web framework, no LLM SDK.** `sqlite3`, `dataclasses`, and
  `typing.Protocol` do the heavy lifting; the only runtime dep is `click`.
  The `ModelClient` protocol means any LLM (OpenAI, Anthropic, local) plugs
  in with ~10 lines.
- **Tools validate before executing.** The `@tool` decorator infers JSON
  Schema from type hints (`str` → `{"type": "string"}`, `Literal` → `enum`,
  `Optional` → `nullable`), and a small built-in validator rejects malformed
  model output with a clear error the agent can recover from.
- **Decisions are data.** HITL verdicts are plain dicts
  (`{"verdict": "approve"|"reject"|"edit", "edits": {...}}`), so any UI —
  CLI, Slack, web — can resolve approvals without framework coupling.
- **Tests never sleep and never call the network.** The engine takes an
  injectable `sleep_fn`; the mock model replays scripted transcripts.

## Roadmap

- Async step execution and parallel fan-out/fan-in
- Webhook delivery for approval requests (Slack/Discord)
- Postgres store backend
- OpenTelemetry tracing of runs
- A real `OpenAIClient` / `AnthropicClient` implementing `ModelClient`

MIT — built as a portfolio project demonstrating durable execution,
event sourcing, and agent loop design.
