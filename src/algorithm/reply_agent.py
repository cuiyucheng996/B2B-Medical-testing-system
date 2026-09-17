"""B2B 单 Agent 答复：对齐主项目 compose，system 按轮次填槽。"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from algorithm.reply_context import (
    build_reply_agent_context,
    build_reply_agent_messages,
    render_reply_agent_system_prompt,
)
from configs.b2b_api import get_b2b_api_config
from configs.paths import ensure_uvgraph_root
from schema.reply_agent import ReplyComposeOutput

logger = logging.getLogger(__name__)

__all__ = ["compose_user_reply"]


def compose_user_reply(
    state: Mapping[str, Any],
    *,
    task: str,
    fallback: str,
) -> str:
    """意图与抽槽之后生成用户可见回复；LLM 不可用或失败时用 fallback。"""
    cfg = get_b2b_api_config()
    if not cfg.get("reply_agent_enabled", True):
        return fallback

    ensure_uvgraph_root()
    try:
        from src.langchain.agent.agent_stream import run_structured_agent_invoke
        from src.llm.chat import get_chat_model
    except Exception:
        logger.warning("reply agent imports failed")
        return fallback

    if get_chat_model() is None:
        return fallback

    ctx = build_reply_agent_context(state, task=task)
    try:
        composed = run_structured_agent_invoke(
            system_prompt=render_reply_agent_system_prompt(ctx),
            agent_messages=build_reply_agent_messages(ctx),
            response_schema=ReplyComposeOutput,
        )
        reply = (composed.assistant_reply or "").strip()
        return reply or fallback
    except Exception:
        logger.warning("reply agent compose failed; using fallback")
        return fallback
