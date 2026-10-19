"""ReAct-style agent loop with a pluggable model backend.

The agent reasons in Thought / Action / Observation cycles::

    Thought: I need to find the error rate for checkout.
    Action: query_metrics {"service": "checkout", "metric": "error_rate"}
    Observation: {"error_rate": 0.023}

    Thought: 2.3% exceeds the 1% SLO. I have what I need.
    Final Answer: The checkout error rate is 2.3%, above the 1% SLO.

Any LLM can drive the loop by implementing the :class:`ModelClient`
protocol. :class:`MockModelClient` replays a scripted transcript, which
makes agent behavior fully deterministic in tests and examples — no
network calls, no API keys.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from agentforge.tools import ToolRegistry, ToolValidationError


class ModelClient(Protocol):
    """What the agent needs from any LLM backend."""

    def complete(self, messages: list[dict[str, str]],
                 tools: Optional[list[dict]] = None) -> str:
        """Return the model's next message given the conversation so far."""
        ...


class MockModelClient:
    """Deterministic scripted model for tests and offline examples.

    Each call to :meth:`complete` returns the next scripted response.
    When the script is exhausted, it returns a ``Final Answer`` built
    from the last observation, so loops always terminate.
    """

    def __init__(self, script: list[str]) -> None:
        self._script = list(script)
        self.calls: list[dict] = []

    def complete(self, messages: list[dict[str, str]],
                 tools: Optional[list[dict]] = None) -> str:
        self.calls.append({"messages": list(messages),
                           "tools": tools})
        if self._script:
            return self._script.pop(0)
        last_obs = ""
        for msg in reversed(messages):
            if msg["role"] == "user" and msg["content"].startswith("Observation:"):
                last_obs = msg["content"][len("Observation:"):].strip()
                break
        return f"Final Answer: Done. Last observation was: {last_obs}"


_ACTION_RE = re.compile(
    r"Action:\s*(?P<tool>[A-Za-z_][A-Za-z0-9_]*)\s*(?P<args>\{.*\})?",
    re.DOTALL,
)
_FINAL_RE = re.compile(r"Final Answer:\s*(?P<answer>.*)", re.DOTALL)


@dataclass
class AgentStep:
    """One turn of the Thought/Action/Observation loop."""

    thought: str
    action: Optional[str] = None
    arguments: dict = field(default_factory=dict)
    observation: Optional[str] = None
    error: Optional[str] = None


@dataclass
class AgentResult:
    answer: str
    steps: list[AgentStep]
    iterations: int
    terminated_early: bool = False


def _parse_action(text: str) -> tuple[Optional[str], dict]:
    match = _ACTION_RE.search(text)
    if not match:
        return None, {}
    args_raw = (match.group("args") or "{}").strip()
    try:
        args = json.loads(args_raw)
    except json.JSONDecodeError:
        return match.group("tool"), {"_raw": args_raw}
    return match.group("tool"), args if isinstance(args, dict) else {"_raw": args}


def _parse_final(text: str) -> Optional[str]:
    match = _FINAL_RE.search(text)
    return match.group("answer").strip() if match else None


def _render_tools(registry: ToolRegistry) -> str:
    lines = []
    for schema in registry.to_schemas():
        params = schema["parameters"].get("properties", {})
        required = schema["parameters"].get("required", [])
        sig = ", ".join(
            f"{k}{'' if k in required else '=...'}: {v.get('type', 'any')}"
            for k, v in params.items()
        )
        lines.append(f"- {schema['name']}({sig}): {schema['description']}")
    return "\n".join(lines)


class ReActAgent:
    """Thought/Action/Observation loop over a tool registry."""

    SYSTEM_PROMPT = (
        "You are a careful autonomous agent. On each turn, write a short "
        "Thought, then either a single Action or a Final Answer.\n"
        "Action format: Action: <tool_name> {\"arg\": value}\n"
        "Final format: Final Answer: <your answer>\n"
        "Available tools:\n{tools}"
    )

    def __init__(
        self,
        client: ModelClient,
        tools: ToolRegistry,
        max_iterations: int = 8,
    ) -> None:
        self.client = client
        self.tools = tools
        self.max_iterations = max_iterations

    def run(self, task: str) -> AgentResult:
        messages = [
            {"role": "system",
             "content": self.SYSTEM_PROMPT.replace(
                 "{tools}", _render_tools(self.tools))},
            {"role": "user", "content": f"Task: {task}"},
        ]
        steps: list[AgentStep] = []

        for _ in range(self.max_iterations):
            response = self.client.complete(messages, self.tools.to_schemas())
            messages.append({"role": "assistant", "content": response})

            final = _parse_final(response)
            thought = response.split("Action:")[0].split("Final Answer:")[0].strip()
            if final is not None:
                steps.append(AgentStep(thought=thought))
                return AgentResult(answer=final, steps=steps,
                                   iterations=len(steps))

            tool_name, args = _parse_action(response)
            step = AgentStep(thought=thought, action=tool_name, arguments=args)
            if tool_name is None:
                step.error = "no Action or Final Answer found in model output"
                step.observation = (
                    "Error: I couldn't parse your response. Reply with either "
                    "'Action: <tool> {args}' or 'Final Answer: <answer>'."
                )
            elif tool_name not in self.tools:
                step.error = f"unknown tool '{tool_name}'"
                step.observation = (
                    f"Error: unknown tool '{tool_name}'. "
                    f"Available: {sorted(self.tools._tools)}."
                )
            else:
                try:
                    # A non-dict _raw payload means the model wrote invalid
                    # JSON — surface it as a validation-style error.
                    if set(args) == {"_raw"}:
                        raise ToolValidationError(
                            tool_name, [f"arguments are not valid JSON: "
                                        f"{args['_raw']!r}"])
                    result = self.tools.call(tool_name, args)
                    step.observation = f"Observation: {result}"
                except ToolValidationError as exc:
                    step.error = str(exc)
                    step.observation = f"Error: {exc}"
                except Exception as exc:  # noqa: BLE001 — tool runtime error
                    step.error = f"{type(exc).__name__}: {exc}"
                    step.observation = f"Error running {tool_name}: {exc}"
            steps.append(step)
            messages.append({"role": "user", "content": step.observation or ""})

        return AgentResult(
            answer="Terminated: max iterations reached without a final answer.",
            steps=steps,
            iterations=len(steps),
            terminated_early=True,
        )
