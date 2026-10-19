"""Example: research agent workflow.

A three-step durable workflow that plans search queries, runs them
through a ReAct agent with a search tool, and synthesizes a briefing.

Runs fully offline: the "search backend" is a small in-memory corpus
and the model is :class:`MockModelClient` with a scripted transcript.
Swap in a real ``ModelClient`` and a real search API to go live.
"""

from __future__ import annotations

from agentforge.agent import MockModelClient, ReActAgent
from agentforge.engine import Step, WorkflowContext
from agentforge.tools import ToolRegistry, tool

# ----------------------------------------------------------------------
# offline knowledge base (stand-in for a real search API)
# ----------------------------------------------------------------------
CORPUS = {
    "temporal workflows": (
        "Temporal is a durable execution platform. Workflows are written as "
        "ordinary code; the platform persists event history so execution can "
        "resume after failures without re-running completed activities."
    ),
    "react agents": (
        "ReAct interleaves reasoning traces with actions: the model writes a "
        "Thought, takes an Action via a tool, reads the Observation, and "
        "repeats until it can give a Final Answer."
    ),
    "evals": (
        "LLM evals measure model behavior with task-specific benchmarks. "
        "Good evals are deterministic, versioned, and track regressions "
        "across model iterations."
    ),
}


@tool(description="Search the offline knowledge base for a topic.")
def kb_search(query: str) -> str:
    """Search the offline knowledge base for a topic."""
    q = query.lower()
    for topic, text in CORPUS.items():
        if topic in q or any(w in topic for w in q.split() if len(w) > 4):
            return f"[{topic}] {text}"
    return "No results found."


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(kb_search)
    return registry


# ----------------------------------------------------------------------
# workflow steps
# ----------------------------------------------------------------------
def plan_queries(ctx: WorkflowContext) -> dict:
    topic = ctx.state.get("topic", "temporal workflows")
    return {
        "queries": [topic, f"{topic} best practices"],
        "topic": topic,
    }


def gather_evidence(ctx: WorkflowContext) -> dict:
    """Run a ReAct agent over the knowledge base for each planned query."""
    script = []
    for query in ctx.state["queries"]:
        script.append(
            f"Thought: I should search for '{query}'.\n"
            f'Action: kb_search {{"query": "{query}"}}'
        )
    # After the last observation the mock falls back to a final answer.
    agent = ReActAgent(MockModelClient(script), _registry(), max_iterations=6)
    findings = []
    for query in ctx.state["queries"]:
        result = agent.run(f"Research this topic: {query}")
        findings.append({"query": query, "notes": result.answer,
                         "tool_calls": len(result.steps)})
    return {"findings": findings}


def synthesize(ctx: WorkflowContext) -> dict:
    findings = ctx.state["findings"]
    bullets = "\n".join(f"- {f['query']}: {f['notes']}" for f in findings)
    return {
        "briefing": (
            f"# Research briefing: {ctx.state['topic']}\n\n{bullets}\n"
        )
    }


def publish_gate(ctx: WorkflowContext) -> dict:
    """Human-in-the-loop gate before the briefing is 'published'."""
    decision = ctx.request_approval(
        "Publish this research briefing?",
        context={"briefing": ctx.state["briefing"]},
    )
    if decision.get("verdict") == "reject":
        raise RuntimeError("briefing rejected by reviewer")
    return {"published": True, "verdict": decision.get("verdict")}


def build_research_workflow() -> list[Step]:
    return [
        Step("plan_queries", plan_queries),
        Step("gather_evidence", gather_evidence, checkpoint_after=True),
        Step("synthesize", synthesize),
        Step("publish_gate", publish_gate),
    ]
