"""LangGraph 前的入口通道规范化：打字 vs ASR。"""

from __future__ import annotations

import re

from configs.b2b_api import get_b2b_api_config
from schema.b2b import DEFAULT_INPUT_CHANNEL, InputChannel

__all__ = [
    "normalize_user_text",
    "parse_input_channel",
    "resolve_input_channel",
]

_SPACE_RE = re.compile(r"\s+")
_ASR_PUNCT_NOISE_RE = re.compile(r"[，。！？、；：""''（）【】《》…·]+")


def parse_input_channel(value: str | None, *, default: InputChannel = DEFAULT_INPUT_CHANNEL) -> InputChannel:
    text = (value or "").strip().lower()
    if text in ("text", "asr"):
        return text  # type: ignore[return-value]
    return default


def resolve_input_channel(
    body_value: str | None = None,
    header_value: str | None = None,
) -> InputChannel:
    """Body 优先，其次 Header ``X-Input-Channel``，最后读配置默认。"""
    cfg = get_b2b_api_config()
    default = parse_input_channel(cfg.get("default_input_channel"), default=DEFAULT_INPUT_CHANNEL)
    if body_value is not None and str(body_value).strip():
        return parse_input_channel(body_value, default=default)
    if header_value is not None and str(header_value).strip():
        return parse_input_channel(header_value, default=default)
    return default


def normalize_user_text(text: str, *, channel: InputChannel) -> str:
    """按入口通道做轻量净化，供后续敏感词/意图管道消费。"""
    normalized = (text or "").strip()
    if not normalized:
        return ""

    cfg = get_b2b_api_config()
    if channel == "asr":
        if cfg["asr_collapse_spaces"]:
            normalized = _SPACE_RE.sub(" ", normalized)
        if cfg["asr_strip_punctuation_noise"]:
            normalized = _ASR_PUNCT_NOISE_RE.sub(" ", normalized)
            normalized = _SPACE_RE.sub(" ", normalized).strip()

    return normalized
