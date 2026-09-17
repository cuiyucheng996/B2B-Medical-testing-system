"""B2B API 接入层：API Key 鉴权与内存滑动窗口限流。"""

from __future__ import annotations

import hmac
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque

from starlette.requests import Request

from configs.b2b_api import get_b2b_api_config

__all__ = [
    "AccessDeniedError",
    "EmptyTextError",
    "RateLimitExceededError",
    "RateLimitInfo",
    "TextTooLongError",
    "check_rate_limit",
    "extract_api_key",
    "reject_empty_text",
    "validate_text_length",
    "verify_api_key",
]


class AccessDeniedError(PermissionError):
    """API Key 缺失或不匹配。"""


class RateLimitExceededError(RuntimeError):
    """请求频率超限。"""

    def __init__(self, message: str, *, retry_after: int = 60) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class TextTooLongError(ValueError):
    """用户文本超过配置上限。"""


class EmptyTextError(ValueError):
    """用户提交了空文本。"""


@dataclass(frozen=True, slots=True)
class RateLimitInfo:
    limit: int
    remaining: int
    window_seconds: int
    retry_after: int


_rate_buckets: dict[str, Deque[float]] = defaultdict(deque)
_rate_lock = threading.Lock()


def extract_api_key(request: Request) -> str:
    header_key = (request.headers.get("X-API-Key") or "").strip()
    if header_key:
        return header_key
    auth = (request.headers.get("Authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return (request.query_params.get("api_key") or "").strip()


def verify_api_key(request: Request) -> None:
    """校验 API Key。yaml 要求鉴权，或已配置 B2B_API_KEY 时必须带对密钥。"""
    cfg = get_b2b_api_config()
    expected = str(cfg.get("api_key") or "").strip()
    required = bool(cfg["api_key_required"]) or bool(expected)
    if not required:
        return
    if not expected:
        raise AccessDeniedError("服务端未配置 B2B_API_KEY")
    provided = extract_api_key(request)
    if not provided or not hmac.compare_digest(provided, expected):
        raise AccessDeniedError("API Key 无效或缺失")


def _rate_limit_client_key(request: Request) -> str:
    api_key = extract_api_key(request)
    if api_key:
        return f"key:{api_key[:16]}"
    if request.client and request.client.host:
        return f"ip:{request.client.host}"
    return "ip:unknown"


def check_rate_limit(request: Request) -> RateLimitInfo | None:
    """按 API Key（优先）或 IP 做滑动窗口限流；关闭时返回 None。"""
    cfg = get_b2b_api_config()
    if not cfg["rate_limit_enabled"]:
        return None

    max_requests = max(1, int(cfg["rate_limit_max_requests"]))
    window_seconds = max(1, int(cfg["rate_limit_window_seconds"]))
    client_key = _rate_limit_client_key(request)
    now = time.monotonic()

    with _rate_lock:
        bucket = _rate_buckets[client_key]
        while bucket and now - bucket[0] > window_seconds:
            bucket.popleft()
        if len(bucket) >= max_requests:
            retry_after = max(1, int(window_seconds - (now - bucket[0])) + 1)
            raise RateLimitExceededError(
                f"请求过于频繁，请 {window_seconds}s 内不超过 {max_requests} 次",
                retry_after=retry_after,
            )
        bucket.append(now)
        remaining = max(0, max_requests - len(bucket))

    return RateLimitInfo(
        limit=max_requests,
        remaining=remaining,
        window_seconds=window_seconds,
        retry_after=window_seconds,
    )


def reject_empty_text(text: str) -> None:
    """拒绝空文本（用户显式提交 message/text 时调用）。"""
    if not (text or "").strip():
        raise EmptyTextError("文本不能为空")


def validate_text_length(text: str) -> None:
    """校验用户自由文本长度（纯选项 step 不调用）。"""
    cfg = get_b2b_api_config()
    max_len = max(1, int(cfg["max_text_length"]))
    if len(text) > max_len:
        raise TextTooLongError(f"文本过长，最多 {max_len} 字")
