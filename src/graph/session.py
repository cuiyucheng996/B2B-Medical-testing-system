"""B2B 图会话入口。"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator, Iterator
from typing import Any
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import StateSnapshot

from algorithm.run_harness import GraphRunResult, run_graph_invoke
from algorithm.short_memory import clear_b2b_short_memory, sync_b2b_short_memory
from configs.paths import ensure_uvgraph_root

from .builder.consult import build_b2b_graph
from .state import B2BRouterState, initial_b2b_state

__all__ = [
    "astream_b2b",
    "get_b2b_graph",
    "get_b2b_state",
    "invoke_b2b",
    "make_thread_config",
    "new_thread_id",
]

_graph: CompiledStateGraph | None = None


def get_b2b_graph() -> CompiledStateGraph:
    global _graph
    if _graph is None:
        ensure_uvgraph_root()
        from src.graph.checkpoint import get_checkpointer

        _graph = build_b2b_graph(
            checkpointer=get_checkpointer(),
            print_graph=False,
        )
    return _graph


def new_thread_id() -> str:
    return str(uuid4())


def make_thread_config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


def get_b2b_state(thread_id: str) -> StateSnapshot:
    return get_b2b_graph().get_state(make_thread_config(thread_id))


def _messages_from_graph(graph: CompiledStateGraph, config: RunnableConfig) -> list[Any]:
    try:
        values = dict(graph.get_state(config).values or {})
    except Exception:
        return []
    return list(values.get("messages") or [])


def invoke_b2b(
    update: B2BRouterState | dict[str, Any] | None = None,
    *,
    thread_id: str,
) -> GraphRunResult:
    graph = get_b2b_graph()
    config = make_thread_config(thread_id)
    ensure_uvgraph_root()
    from src.algorithm.stream_run import reset_active_thread, set_active_thread

    token = set_active_thread(thread_id)
    try:
        if update is None:
            return run_graph_invoke(
                graph,
                dict(initial_b2b_state()),
                config=config,
                update_then_resume=False,
            )

        snapshot = graph.get_state(config)
        if snapshot.values:
            # 短期记忆只同步 invoke 前的历史，不含本轮用户话
            sync_b2b_short_memory(thread_id, list(snapshot.values.get("messages") or []))
            result = run_graph_invoke(
                graph,
                dict(update),
                config=config,
                update_then_resume=True,
            )
            if dict((graph.get_state(config).values or {})).get("should_end"):
                clear_b2b_short_memory(thread_id)
            return result

        return run_graph_invoke(
            graph,
            {**initial_b2b_state(), **update},
            config=config,
            update_then_resume=False,
        )
    finally:
        reset_active_thread(token)


_STREAM_DONE = object()


def _resolve_stream_input(
    graph: CompiledStateGraph,
    config: RunnableConfig,
    update: B2BRouterState | dict[str, Any] | None,
) -> dict[str, Any] | None:
    if update is None:
        return dict(initial_b2b_state())
    snapshot = graph.get_state(config)
    if snapshot.values:
        graph.update_state(config, update)
        return None
    return {**initial_b2b_state(), **update}


def _iter_b2b_stream(
    update: B2BRouterState | dict[str, Any] | None,
    *,
    thread_id: str,
) -> Iterator[tuple[str, Any]]:
    graph = get_b2b_graph()
    config = make_thread_config(thread_id)
    ensure_uvgraph_root()
    from src.algorithm.stream_run import reset_active_thread, set_active_thread

    token = set_active_thread(thread_id)
    try:
        if update is not None:
            snapshot = graph.get_state(config)
            if snapshot.values:
                sync_b2b_short_memory(thread_id, list(snapshot.values.get("messages") or []))
        input_state = _resolve_stream_input(graph, config, update)
        yield from graph.stream(
            input_state,
            config,
            stream_mode=["custom", "updates"],
        )
        if dict((graph.get_state(config).values or {})).get("should_end"):
            clear_b2b_short_memory(thread_id)
    finally:
        reset_active_thread(token)


async def astream_b2b(
    update: B2BRouterState | dict[str, Any] | None = None,
    *,
    thread_id: str,
) -> AsyncIterator[tuple[str, Any]]:
    """流式推进 B2B 图：custom 给 token，updates 给节点进度。"""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Any] = asyncio.Queue()

    def producer() -> None:
        try:
            for item in _iter_b2b_stream(update, thread_id=thread_id):
                future = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
                future.result(timeout=600)
        except Exception as exc:
            future = asyncio.run_coroutine_threadsafe(queue.put(exc), loop)
            future.result(timeout=30)
        finally:
            future = asyncio.run_coroutine_threadsafe(queue.put(_STREAM_DONE), loop)
            future.result(timeout=30)

    threading.Thread(target=producer, daemon=True, name=f"b2b-stream-{thread_id[:8]}").start()

    while True:
        item = await queue.get()
        if item is _STREAM_DONE:
            break
        if isinstance(item, Exception):
            raise item
        yield item
