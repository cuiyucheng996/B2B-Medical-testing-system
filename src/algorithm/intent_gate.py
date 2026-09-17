"""大意图区分：敬语/闲聊 vs 主诉（有效症状描述）。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from configs.paths import ensure_uvgraph_root

__all__ = [
    "IntentCategory",
    "IntentGateResult",
    "classify_user_intent",
]

IntentCategory = Literal["complaint", "chitchat", "honorific", "too_vague"]

_CHITCHAT_PATTERNS = (
    r"^你好$",
    r"^在吗$",
    r"^谢谢",
    r"^多谢",
    r"^再见$",
    r"^拜拜$",
    r"^哈哈",
    r"^好的$",
    r"^嗯$",
    r"^哦$",
)
_HONORIFIC_PATTERNS = (
    r"^您好",
    r"^老师好",
    r"^医生好",
    r"^麻烦您",
    r"^辛苦了",
    r"^打扰了",
)


@dataclass(frozen=True)
class IntentGateResult:
    category: IntentCategory
    is_valid_complaint: bool
    reason: str = ""


def _lexicons() -> tuple[frozenset[str], tuple[str, ...]]:
    ensure_uvgraph_root()
    from src.text.lexicons import BODY_SYMPTOM_HINTS, FILLER_UTTERANCES

    return FILLER_UTTERANCES, BODY_SYMPTOM_HINTS


def _match_patterns(text: str, patterns: tuple[str, ...]) -> bool:
    compact = text.replace(" ", "")
    return any(re.search(pat, compact) for pat in patterns)


def classify_user_intent(text: str) -> IntentGateResult:
    """规则判定用户文字是否为主诉类有效输入。"""
    compact = (text or "").replace(" ", "")
    if not compact:
        return IntentGateResult(
            category="too_vague",
            is_valid_complaint=False,
            reason="文本为空",
        )

    fillers, symptom_hints = _lexicons()

    # 含症状/部位词时优先视为主诉（如「您好医生，我头疼」）
    if any(hint in compact for hint in symptom_hints):
        return IntentGateResult(
            category="complaint",
            is_valid_complaint=True,
            reason="含症状或部位描述",
        )

    if compact in fillers:
        return IntentGateResult(
            category="chitchat",
            is_valid_complaint=False,
            reason="语气词或无效寒暄",
        )

    if _match_patterns(compact, _HONORIFIC_PATTERNS):
        return IntentGateResult(
            category="honorific",
            is_valid_complaint=False,
            reason="敬语寒暄，未描述症状",
        )

    if _match_patterns(compact, _CHITCHAT_PATTERNS):
        return IntentGateResult(
            category="chitchat",
            is_valid_complaint=False,
            reason="闲聊内容，未描述症状",
        )

    if len(compact) < 4:
        return IntentGateResult(
            category="too_vague",
            is_valid_complaint=False,
            reason="描述过短，请补充具体不适",
        )

    return IntentGateResult(
        category="complaint",
        is_valid_complaint=True,
        reason="视为有效主诉描述",
    )
