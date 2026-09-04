"""Dependency-free validator for the JSON-Schema subset used by this project."""

from __future__ import annotations

import re
from typing import Any


class SchemaValidationError(ValueError):
    pass


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def validate_instance(
    value: Any,
    schema: dict[str, Any],
    *,
    path: str = "$",
) -> None:
    for index, item_schema in enumerate(schema.get("allOf", [])):
        validate_instance(value, item_schema, path=f"{path}.allOf[{index}]")
    condition = schema.get("if")
    if condition is not None:
        try:
            validate_instance(value, condition, path=path)
        except SchemaValidationError:
            branch = schema.get("else")
        else:
            branch = schema.get("then")
        if branch is not None:
            validate_instance(value, branch, path=path)

    if "const" in schema and value != schema["const"]:
        raise SchemaValidationError(
            f"{path}: expected constant {schema['const']!r}, got {value!r}"
        )
    if "enum" in schema and value not in schema["enum"]:
        raise SchemaValidationError(f"{path}: {value!r} is not in allowed values")
    expected = schema.get("type")
    if expected:
        candidates = expected if isinstance(expected, list) else [expected]
        if not any(_matches_type(value, item) for item in candidates):
            raise SchemaValidationError(
                f"{path}: expected type {candidates}, got {type(value).__name__}"
            )

    if isinstance(value, dict):
        missing = [key for key in schema.get("required", []) if key not in value]
        if missing:
            raise SchemaValidationError(f"{path}: missing required keys {missing}")
        properties = schema.get("properties", {})
        for key, item in value.items():
            if key in properties:
                validate_instance(item, properties[key], path=f"{path}.{key}")
    elif isinstance(value, list):
        if len(value) < int(schema.get("minItems", 0)):
            raise SchemaValidationError(f"{path}: too few items")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            raise SchemaValidationError(f"{path}: too many items")
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(value):
                validate_instance(item, item_schema, path=f"{path}[{index}]")
    elif isinstance(value, str):
        if len(value) < int(schema.get("minLength", 0)):
            raise SchemaValidationError(f"{path}: string is too short")
        pattern = schema.get("pattern")
        if pattern and not re.search(pattern, value):
            raise SchemaValidationError(f"{path}: string does not match {pattern}")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise SchemaValidationError(f"{path}: value is below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise SchemaValidationError(f"{path}: value is above maximum")
