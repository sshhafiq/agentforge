"""Tests for tool schema inference, validation, and the registry."""

from typing import Literal, Optional

import pytest

from agentforge.tools import (
    Tool,
    ToolRegistry,
    ToolValidationError,
    tool,
    validate_against_schema,
)


def test_schema_inferred_from_type_hints():
    @tool(description="Search things.")
    def search(query: str, top_k: int = 5, verbose: bool = False) -> str:
        """Search things."""
        return query

    t = search.tool
    assert t.name == "search"
    params = t.parameters
    assert params["type"] == "object"
    assert params["properties"]["query"] == {"type": "string"}
    assert params["properties"]["top_k"] == {"type": "integer", "default": 5}
    assert params["properties"]["verbose"] == {"type": "boolean", "default": False}
    assert params["required"] == ["query"]
    assert params["additionalProperties"] is False


def test_bare_decorator_uses_function_name_and_docstring():
    @tool
    def do_thing(x: int) -> int:
        """Does the thing."""
        return x

    assert do_thing.tool.name == "do_thing"
    assert do_thing.tool.description == "Does the thing."


def test_custom_name_and_description():
    @tool(name="renamed", description="Custom desc")
    def f(x: str) -> str:
        return x

    assert f.tool.name == "renamed"
    assert f.tool.description == "Custom desc"


def test_optional_and_literal_hints():
    @tool
    def f(mode: Literal["a", "b"], path: Optional[str] = None) -> str:
        return mode

    props = f.tool.parameters["properties"]
    assert props["mode"] == {"enum": ["a", "b"]}
    assert props["path"]["nullable"] is True
    assert "path" not in f.tool.parameters["required"]


def test_valid_call_passes_through():
    @tool
    def add(a: int, b: int) -> int:
        return a + b

    assert add(a=1, b=2) == 3
    assert add.tool(a=1, b=2) == 3


def test_missing_required_argument_raises():
    @tool
    def add(a: int, b: int) -> int:
        return a + b

    with pytest.raises(ToolValidationError) as exc_info:
        add(a=1)
    assert "missing required property 'b'" in str(exc_info.value)


def test_wrong_type_raises():
    @tool
    def add(a: int, b: int) -> int:
        return a + b

    with pytest.raises(ToolValidationError, match="expected integer"):
        add(a="one", b=2)


def test_unexpected_property_raises():
    @tool
    def add(a: int, b: int) -> int:
        return a + b

    with pytest.raises(ToolValidationError, match="unexpected property"):
        add(a=1, b=2, c=3)


def test_bool_is_not_an_integer():
    assert validate_against_schema(True, {"type": "integer"}) != []


def test_enum_validation():
    schema = {"enum": ["x", "y"]}
    assert validate_against_schema("x", schema) == []
    assert validate_against_schema("z", schema) != []


def test_registry_register_and_call():
    registry = ToolRegistry()

    @tool(description="Echo.")
    def echo(text: str) -> str:
        return text

    registry.register(echo)
    assert "echo" in registry
    assert len(registry) == 1
    assert registry.call("echo", {"text": "hi"}) == "hi"


def test_registry_rejects_duplicates():
    registry = ToolRegistry()

    @tool
    def dup() -> str:
        return "x"

    registry.register(dup)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(dup)


def test_registry_unknown_tool():
    registry = ToolRegistry()
    with pytest.raises(KeyError, match="unknown tool"):
        registry.get("missing")


def test_registry_rejects_non_tools():
    registry = ToolRegistry()
    with pytest.raises(TypeError):
        registry.register("not a tool")


def test_to_schemas_shape():
    registry = ToolRegistry()

    @tool(description="Adds.")
    def add(a: int, b: int) -> int:
        return a + b

    registry.register(add)
    schemas = registry.to_schemas()
    assert schemas[0]["name"] == "add"
    assert schemas[0]["description"] == "Adds."
    assert "parameters" in schemas[0]


def test_float_list_and_dict_hints():
    @tool
    def f(temperature: float, tags: list, meta: dict, query: str) -> str:
        return query

    props = f.tool.parameters["properties"]
    assert props["temperature"] == {"type": "number"}
    assert props["tags"] == {"type": "array"}
    assert props["meta"] == {"type": "object"}
    assert set(f.tool.parameters["required"]) == {
        "temperature",
        "tags",
        "meta",
        "query",
    }


def test_nested_object_validation():
    schema = {
        "type": "object",
        "properties": {"cfg": {"type": "object",
                               "properties": {"n": {"type": "integer"}},
                               "required": ["n"]}},
        "required": ["cfg"],
    }
    assert validate_against_schema({"cfg": {"n": 1}}, schema) == []
    errs = validate_against_schema({"cfg": {}}, schema)
    assert any("missing required property 'n'" in e for e in errs)
    errs = validate_against_schema({"cfg": {"n": "x"}}, schema)
    assert any("expected integer" in e for e in errs)
