"""B2B HTTP 接入：仅拦截 /api/sessions* 的鉴权与限流。"""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from algorithm.access_guard import (
    AccessDeniedError,
    RateLimitExceededError,
    RateLimitInfo,
    check_rate_limit,
    verify_api_key,
)

_SESSIONS_PREFIX = "/api/sessions"
_RATE_HEADERS = ("X-RateLimit-Limit", "X-RateLimit-Remaining", "Retry-After")


def _error_body(code: str, message: str) -> dict:
    return {"ok": False, "error": {"code": code, "message": message}}


def _attach_rate_headers(response: Response, info: RateLimitInfo | None) -> None:
    if info is None:
        return
    response.headers["X-RateLimit-Limit"] = str(info.limit)
    response.headers["X-RateLimit-Remaining"] = str(info.remaining)
    response.headers["Retry-After"] = str(info.retry_after)


async def b2b_sessions_access_guard(request: Request, call_next) -> Response:
    """OPTIONS 放行；/api/sessions 先鉴权再限流，错误返回 JSON。"""
    if request.method == "OPTIONS" or not request.url.path.startswith(_SESSIONS_PREFIX):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store, max-age=0"
        return response

    try:
        verify_api_key(request)
        rate_info = check_rate_limit(request)
    except AccessDeniedError as exc:
        return JSONResponse(
            status_code=401,
            content=_error_body("unauthorized", str(exc)),
            headers={"WWW-Authenticate": "ApiKey"},
        )
    except RateLimitExceededError as exc:
        retry_after = int(getattr(exc, "retry_after", 60) or 60)
        return JSONResponse(
            status_code=429,
            content=_error_body("rate_limited", str(exc)),
            headers={"Retry-After": str(retry_after)},
        )

    response = await call_next(request)
    _attach_rate_headers(response, rate_info)
    return response
