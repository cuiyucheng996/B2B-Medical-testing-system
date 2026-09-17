"""图执行 harness：interrupt 不算失败；异常转结构化结果，不向上抛成 500。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)

__all__ = ["GraphRunResult", "run_graph_invoke"]

RunEvent = Literal["done", "error", "interrupted"]


@dataclass(frozen=True)
class GraphRunResult:
    values: dict[str, Any]
    event: RunEvent = "done"
    error: dict[str, str] | None = None

    @property
    def ok(self) -> bool:
        return self.event != "error"


def _is_interrupt(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name in {"GraphInterrupt", "NodeInterrupt"}:
        return True
    module = getattr(type(exc), "__module__", "") or ""
    return "Interrupt" in name and "langgraph" in module


def _safe_values(graph: CompiledStateGraph, config: RunnableConfig) -> dict[str, Any]:
    try:
        snapshot = graph.get_state(config)
        return dict(snapshot.values or {})
    except Exception:
        return {}


def run_graph_invoke(
    graph: CompiledStateGraph,
    payload: dict[str, Any] | None,
    *,
    config: RunnableConfig,
    update_then_resume: bool,
) -> GraphRunResult:
    """推进图。中断返回 interrupted；其它异常返回 error + 当前 checkpoint。"""
    try:
        if update_then_resume:
            graph.update_state(config, payload or {})
            raw = graph.invoke(None, config)
        else:
            raw = graph.invoke(payload, config)
        values = dict(raw or {}) if isinstance(raw, dict) else _safe_values(graph, config)
        return GraphRunResult(values=values, event="done")
    except Exception as exc:
        if _is_interrupt(exc):
            return GraphRunResult(values=_safe_values(graph, config), event="interrupted")
        logger.warning("graph invoke failed: %s", exc)
        return GraphRunResult(
            values=_safe_values(graph, config),
            event="error",
            error={
                "code": "graph_invoke_failed",
                "message": "服务暂时不可用，请稍后再试",
            },
        )
