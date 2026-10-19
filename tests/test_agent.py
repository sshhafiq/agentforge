"""Tests for the ReAct agent loop and the mock model client."""

from agentforge.agent import MockModelClient, ReActAgent
from agentforge.tools import ToolRegistry, tool


@tool(description="Add two numbers.")
def add(a: int, b: int) -> int:
    return a + b


@tool(description="Look up a user by id.")
def get_user(user_id: str) -> str:
    return f"user:{user_id}"


def make_registry():
    registry = ToolRegistry()
    registry.register(add)
    registry.register(get_user)
    return registry


def test_agent_calls_tool_and_returns_final_answer():
    script = [
        "Thought: I need to add 2 and 3.\nAction: add {\"a\": 2, \"b\": 3}",
        "Thought: The sum is 5.\nFinal Answer: 5",
    ]
    agent = ReActAgent(MockModelClient(script), make_registry())
    result = agent.run("What is 2 + 3?")
    assert result.answer == "5"
    assert len(result.steps) == 2
    assert result.steps[0].action == "add"
    assert result.steps[0].arguments == {"a": 2, "b": 3}
    assert result.steps[0].observation == "Observation: 5"
    assert not result.terminated_early


def test_agent_handles_unknown_tool_gracefully():
    script = [
        "Thought: Let me try this.\nAction: nope {}",
        "Final Answer: I could not do it.",
    ]
    agent = ReActAgent(MockModelClient(script), make_registry())
    result = agent.run("do the thing")
    assert result.steps[0].error is not None
    assert "unknown tool" in result.steps[0].observation
    assert result.answer == "I could not do it."


def test_agent_reports_tool_validation_errors():
    script = [
        "Thought: Wrong types on purpose.\n"
        "Action: add {\"a\": \"two\", \"b\": 3}",
        "Final Answer: Giving up.",
    ]
    agent = ReActAgent(MockModelClient(script), make_registry())
    result = agent.run("add badly")
    assert "expected integer" in result.steps[0].observation
    assert result.steps[0].error is not None


def test_agent_reports_malformed_json():
    script = [
        "Thought: Broken JSON.\nAction: add {not json}",
        "Final Answer: done",
    ]
    agent = ReActAgent(MockModelClient(script), make_registry())
    result = agent.run("x")
    assert "not valid JSON" in result.steps[0].observation


def test_agent_terminates_at_max_iterations():
    script = ["Thought: looping forever.\nAction: add {\"a\": 1, \"b\": 1}"] * 10
    agent = ReActAgent(MockModelClient(script), make_registry(),
                       max_iterations=3)
    result = agent.run("loop")
    assert result.terminated_early
    assert len(result.steps) == 3
    assert "max iterations" in result.answer


def test_agent_handles_unparseable_output():
    script = ["just some rambling with no action", "Final Answer: ok"]
    agent = ReActAgent(MockModelClient(script), make_registry())
    result = agent.run("x")
    assert "couldn't parse" in result.steps[0].observation
    assert result.answer == "ok"


def test_mock_client_falls_back_when_script_exhausted():
    client = MockModelClient(["Final Answer: first"])
    agent = ReActAgent(client, make_registry(), max_iterations=4)
    result = agent.run("task")
    assert result.answer == "first"
    # Second run reuses the client: script exhausted -> fallback answer.
    # (The agent strips the "Final Answer:" prefix from its answer.)
    result2 = agent.run("task")
    assert result2.answer.startswith("Done.")


def test_tool_runtime_errors_become_observations():
    @tool(description="Always blows up.")
    def boom() -> str:
        raise RuntimeError("kaput")

    registry = ToolRegistry()
    registry.register(boom)
    script = [
        "Thought: Try it.\nAction: boom {}",
        "Final Answer: it failed",
    ]
    agent = ReActAgent(MockModelClient(script), registry)
    result = agent.run("x")
    assert "kaput" in result.steps[0].observation
    assert "RuntimeError" in result.steps[0].error
