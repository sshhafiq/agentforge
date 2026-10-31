"""Tool registry with JSON-schema-validated tool definitions.

Tools are the actions a ReAct agent can take. Each tool declares a JSON
Schema for its arguments; calls are validated *before* the function runs,
so malformed model output fails fast with a clear error instead of
blowing up deep inside business logic.

The ``@tool`` decorator infers the schema from type hints::

    @tool(description="Search the knowledge base.")
    def search(query: str, top_k: int = 3) -> str:
        ...

``query`` becomes a required ``{"type": "string"}`` property, ``top_k`` an
optional ``{"type": "integer", "default": 3}``. Supported hint types: str,
int, float, bool, list, dict, Optional[...], and Literal[...] (as enum).
"""

from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, Union, get_args, get_origin


class ToolValidationError(ValueError):
    """Raised when tool arguments fail JSON Schema validation."""

    def __init__(self, tool_name: str, errors: list[str]) -> None:
        self.tool_name = tool_name
        self.errors = errors
        super().__init__(
            f"invalid arguments for tool '{tool_name}': " + "; ".join(errors)
        )


def _type_to_schema(tp: Any) -> dict:
    """Map a Python type hint to a JSON Schema fragment."""
    if tp is inspect.Parameter.empty or tp is Any:
        return {}
    origin = get_origin(tp)
    if origin is Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:  # Optional[X]
            schema = _type_to_schema(args[0])
            schema["nullable"] = True
            return schema
        return {"anyOf": [_type_to_schema(a) for a in args]}
    if origin is Literal:
        return {"enum": list(get_args(tp))}
    if origin in (list, tuple):
        return {"type": "array"}
    if origin is dict:
        return {"type": "object"}
    mapping = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
    }
    return dict(mapping.get(tp, {}))


def _schema_from_signature(fn: Callable) -> dict:
    sig = inspect.signature(fn)
    properties: dict[str, dict] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        schema = _type_to_schema(param.annotation)
        if param.default is not inspect.Parameter.empty:
            schema["default"] = param.default
        else:
            required.append(name)
        properties[name] = schema
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _check_type(value: Any, expected: str, path: str, errors: list[str]) -> None:
    checks = {
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
        "array": lambda v: isinstance(v, list),
        "object": lambda v: isinstance(v, dict),
        "null": lambda v: v is None,
    }
    check = checks.get(expected)
    if check and not check(value):
        errors.append(
            f"{path}: expected {expected}, got {type(value).__name__}"
        )


def validate_against_schema(
    data: Any, schema: dict, path: str = "args"
) -> list[str]:
    """Minimal JSON Schema validator. Returns a list of error strings."""
    errors: list[str] = []
    if not isinstance(schema, dict):
        return errors

    if "enum" in schema and data not in schema["enum"]:
        errors.append(f"{path}: {data!r} is not one of {schema['enum']}")
    if "anyOf" in schema:
        if not any(not validate_against_schema(data, s, path)
                   for s in schema["anyOf"]):
            errors.append(f"{path}: does not match any allowed schema")
        return errors
    expected = schema.get("type")
    if expected:
        if schema.get("nullable") and data is None:
            return errors
        _check_type(data, expected, path, errors)
        if errors:
            return errors
    if expected == "object" or "properties" in schema:
        if not isinstance(data, dict):
            errors.append(f"{path}: expected object")
            return errors
        props = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in data:
                errors.append(f"{path}: missing required property '{name}'")
        for name, value in data.items():
            if name in props:
                errors.extend(
                    validate_against_schema(value, props[name], f"{path}.{name}")
                )
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: unexpected property '{name}'")
    if expected == "array" and "items" in schema:
        for i, item in enumerate(data):
            errors.extend(
                validate_against_schema(item, schema["items"], f"{path}[{i}]")
            )
    return errors


@dataclass
class Tool:
    """A validated, callable tool."""

    name: str
    description: str
    parameters: dict
    func: Callable[..., Any] = field(repr=False)

    def __call__(self, **kwargs: Any) -> Any:
        errors = validate_against_schema(kwargs, self.parameters)
        if errors:
            raise ToolValidationError(self.name, errors)
        return self.func(**kwargs)

    def to_schema(self) -> dict:
        """Model-facing schema (OpenAI function-calling style)."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


def tool(
    _fn: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Any:
    """Decorator turning a function into a schema-validated :class:`Tool`.

    Usable bare (``@tool``) or with arguments
    (``@tool(name=..., description=...)``).
    """

    def decorate(fn: Callable) -> Tool:
        tool_name = name or fn.__name__
        doc = description or (fn.__doc__ or "").strip().split("\n")[0]
        wrapped = Tool(
            name=tool_name,
            description=doc,
            parameters=_schema_from_signature(fn),
            func=fn,
        )

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return wrapped(*args, **kwargs)

        wrapper.tool = wrapped  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    if _fn is not None:
        return decorate(_fn)
    return decorate


class ToolRegistry:
    """Named collection of tools, exposed to the agent loop."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool_obj: Any) -> Tool:
        """Register a :class:`Tool` or a ``@tool``-decorated function."""
        resolved = getattr(tool_obj, "tool", tool_obj)
        if not isinstance(resolved, Tool):
            raise TypeError(
                f"cannot register {type(tool_obj).__name__}: "
                "expected a Tool or @tool-decorated function"
            )
        if resolved.name in self._tools:
            raise ValueError(f"tool '{resolved.name}' is already registered")
        self._tools[resolved.name] = resolved
        return resolved

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"unknown tool: '{name}'") from None

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        """Registered tool names, in registration order."""
        return list(self._tools)

    def descriptions(self) -> dict[str, str]:
        """Map each registered tool name to its one-line description."""
        return {name: tool.description for name, tool in self._tools.items()}

    def to_schemas(self) -> list[dict]:
        """All tools as model-facing function schemas."""
        return [t.to_schema() for t in self._tools.values()]

    def call(self, name: str, arguments: dict) -> Any:
        """Validate and execute a tool by name."""
        return self.get(name)(**arguments)
