"""组装并编译 B2B LangGraph。"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from configs.paths import B2B_ROOT

from ..state import B2BRouterState
from .consult_graph import INTERRUPT_AFTER, add_consult_edges, add_consult_nodes

__all__ = ["build_b2b_graph"]

_DEFAULT_GRAPH_IMAGE = B2B_ROOT / "doc" / "b2b_graph.png"


def _open_image(path: Path) -> None:
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606
    elif sys.platform == "darwin":
        os.system(f"open {path}")  # noqa: S605
    else:
        os.system(f"xdg-open {path}")  # noqa: S605


def build_b2b_graph(
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    print_graph: bool = False,
    graph_image_path: Path | str | None = None,
    open_graph_image: bool = False,
) -> CompiledStateGraph:
    builder = StateGraph(B2BRouterState)
    add_consult_nodes(builder)
    add_consult_edges(builder)

    compile_kwargs: dict[str, Any] = {
        "interrupt_after": list(INTERRUPT_AFTER),
    }
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

    graph = builder.compile(**compile_kwargs)

    if print_graph:
        path = Path(graph_image_path) if graph_image_path is not None else _DEFAULT_GRAPH_IMAGE
        path.parent.mkdir(parents=True, exist_ok=True)
        graph.get_graph().draw_mermaid_png(
            output_file_path=str(path),
            max_retries=5,
            retry_delay=2.0,
        )
        print(f"B2B graph saved: {path}")
        if open_graph_image:
            _open_image(path)

    return graph


if __name__ == "__main__":
    from configs.paths import ensure_b2b_src, ensure_uvgraph_root

    ensure_b2b_src()
    ensure_uvgraph_root()
    from src.graph.checkpoint import get_checkpointer

    build_b2b_graph(checkpointer=get_checkpointer(), print_graph=True, open_graph_image=True)
