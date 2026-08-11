"""Static validation for untrusted API-generated Python kernel sources."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List

from .schema import TaskSpec


class SourceValidationError(ValueError):
    """Raised before an invalid source candidate reaches the evaluator."""


class SourceValidator:
    """Enforce task-level source invariants without executing generated code."""

    def validate(self, task: TaskSpec, source_code: str, source_name: str) -> None:
        if Path(source_name).suffix != ".py":
            raise SourceValidationError("the kernel source must be a .py file")
        if "\x00" in source_code:
            raise SourceValidationError("the kernel source contains a null byte")
        maximum = int(task.constraints.get("max_source_bytes", 1_000_000))
        if maximum <= 0:
            raise SourceValidationError("constraints.max_source_bytes must be positive")
        size = len(source_code.encode("utf-8"))
        if size > maximum:
            raise SourceValidationError(
                "the kernel source is %d bytes, above the %d-byte limit"
                % (size, maximum)
            )

        try:
            tree = ast.parse(source_code, filename=source_name)
        except SyntaxError as error:
            location = "line %s" % error.lineno if error.lineno else "unknown line"
            raise SourceValidationError(
                "the kernel source is not valid Python syntax at %s: %s"
                % (location, error.msg)
            ) from error

        if bool(task.constraints.get("preserve_entrypoint", True)):
            names = {
                node.name
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            if task.entrypoint not in names:
                raise SourceValidationError(
                    "the kernel source no longer defines entrypoint %r"
                    % task.entrypoint
                )

        for fragment in self._string_list(task, "required_fragments"):
            if fragment not in source_code:
                raise SourceValidationError(
                    "the kernel source is missing required fragment %r" % fragment
                )
        for fragment in self._string_list(task, "forbidden_fragments"):
            if fragment in source_code:
                raise SourceValidationError(
                    "the kernel source contains forbidden fragment %r" % fragment
                )

    @staticmethod
    def _string_list(task: TaskSpec, name: str) -> List[str]:
        value = task.constraints.get(name, [])
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item for item in value
        ):
            raise SourceValidationError(
                "constraints.%s must be a list of non-empty strings" % name
            )
        return value
