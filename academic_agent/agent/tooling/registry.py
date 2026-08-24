"""One registry shared by Qwen function-calling and the UI buttons."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class ToolSpec:
    name: str
    label: str
    category: str
    description: str
    handler: Callable[..., dict[str, Any]]
    parameters: dict[str, Any] = field(default_factory=dict)
    requires_data: bool = True


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> ToolSpec:
        self._tools[spec.name] = spec
        return spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def all(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def by_category(self) -> dict[str, list[ToolSpec]]:
        result: dict[str, list[ToolSpec]] = {}
        for spec in self._tools.values():
            result.setdefault(spec.category, []).append(spec)
        return result

    def execute(self, name: str, **kwargs: Any) -> dict[str, Any]:
        spec = self.get(name)
        if spec is None:
            return {"success": False, "error": f"未知工具：{name}"}
        try:
            return spec.handler(**kwargs)
        except Exception as exc:
            return {"success": False, "tool": name, "error": str(exc)}


tool_registry = ToolRegistry()
