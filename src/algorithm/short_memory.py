"""B2B 短期对话：复用主项目 Redis 滑动窗口。"""

from __future__ import annotations

import logging
from typing import Any

from configs.paths import ensure_uvgraph_root

logger = logging.getLogger(__name__)

__all__ = [
    "clear_b2b_short_memory",
    "load_b2b_short_messages",
    "sync_b2b_short_memory",
]


def sync_b2b_short_memory(thread_id: str, messages: list[Any]) -> int:
    """checkpoint messages → Redis List（最近 SHORT_MAX_ITEMS 组）。"""
    sid = (thread_id or "").strip()
    if not sid:
        return 0
    ensure_uvgraph_root()
    try:
        from src.redis.conversation import sync_conversation_from_messages_sync

        return int(sync_conversation_from_messages_sync(sid, messages) or 0)
    except Exception:
        logger.warning("sync short memory failed thread_id=%s", sid)
        return 0


def load_b2b_short_messages(thread_id: str | None = None) -> list[dict[str, str]]:
    """读 Redis 滑动窗口；无 thread 或失败返回空。"""
    ensure_uvgraph_root()
    sid = (thread_id or "").strip()
    if not sid:
        try:
            from src.algorithm.stream_run import get_active_thread

            sid = (get_active_thread() or "").strip()
        except Exception:
            sid = ""
    if not sid:
        return []
    try:
        from src.redis.conversation import context_messages_sync

        return list(context_messages_sync(sid) or [])
    except Exception:
        logger.warning("load short memory failed thread_id=%s", sid)
        return []


def clear_b2b_short_memory(thread_id: str) -> None:
    sid = (thread_id or "").strip()
    if not sid:
        return
    ensure_uvgraph_root()
    try:
        from src.redis.conversation import clear_conversation_sync

        clear_conversation_sync(sid)
    except Exception:
        logger.warning("clear short memory failed thread_id=%s", sid)
