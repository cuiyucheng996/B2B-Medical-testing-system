"""敏感词与脏格式净化。"""

from __future__ import annotations

import re
import unicodedata

from configs.paths import ensure_uvgraph_root

__all__ = [
    "SensitiveContentError",
    "sanitize_user_text",
]

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200d\ufeff]")
_REPEAT_PUNCT_RE = re.compile(r"([，。！？、,.!?;；:：])\1{2,}")
_SENSITIVE_HINTS = (
    "自杀",
    "杀人",
    "毒品",
    "色情",
    "赌博",
    "枪支",
    "爆炸物",
)


class SensitiveContentError(ValueError):
    """含敏感违规内容。"""


def _load_punctuation_only() -> frozenset[str]:
    ensure_uvgraph_root()
    from src.text.lexicons import PUNCTUATION_ONLY

    return PUNCTUATION_ONLY


def sanitize_user_text(text: str) -> str:
    """净化文本：去控制字符、零宽符、重复标点；纯标点视为空。"""
    cleaned = unicodedata.normalize("NFKC", text or "")
    cleaned = _CONTROL_RE.sub("", cleaned)
    cleaned = _ZERO_WIDTH_RE.sub("", cleaned)
    cleaned = _REPEAT_PUNCT_RE.sub(r"\1", cleaned)
    cleaned = cleaned.strip()

    compact = cleaned.replace(" ", "")
    if compact and all(char in _load_punctuation_only() for char in compact):
        return ""

    return cleaned


def contains_sensitive_content(text: str) -> bool:
    compact = (text or "").replace(" ", "")
    return any(word in compact for word in _SENSITIVE_HINTS)


def guard_sensitive_text(text: str) -> None:
    if contains_sensitive_content(text):
        raise SensitiveContentError("内容含敏感信息，请修改后重试")
