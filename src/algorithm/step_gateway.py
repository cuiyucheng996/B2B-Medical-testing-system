"""B2B step 请求 → 四层管道 → LangGraph update。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import HumanMessage
from starlette.requests import Request

from algorithm.access_guard import (
    EmptyTextError,
    TextTooLongError,
    reject_empty_text,
    validate_text_length,
)
from algorithm.input_pipeline import normalize_user_text, resolve_input_channel
from algorithm.intent_gate import IntentGateResult
from algorithm.text_guard import SensitiveContentError, guard_sensitive_text, sanitize_user_text
from schema.b2b import InputChannel

__all__ = [
    "StepGatewayError",
    "StepPrepareResult",
    "prepare_step_update",
]

GateBlockCategory = Literal["sensitive", "chitchat", "honorific", "too_vague", "empty_text"]


class StepGatewayError(ValueError):
    """step 载荷不合法。"""


@dataclass(frozen=True)
class StepPrepareResult:
    update: dict[str, Any] | None
    blocked: bool = False
    block_category: GateBlockCategory | None = None
    gate_message: str | None = None
    intent: IntentGateResult | None = None
    input_channel: InputChannel | None = None


def _has_option_payload(
    *,
    selected_body_part: str | None,
    selected_disease: str | None,
    selected_symptoms: list[str] | None,
    org_id: str | None,
) -> bool:
    return bool(
        (selected_body_part or "").strip()
        or (selected_disease or "").strip()
        or any(str(item).strip() for item in (selected_symptoms or []))
        or (org_id or "").strip()
    )


def _process_free_text(
    raw_text: str,
    *,
    input_channel: InputChannel,
    skip_intent_gate: bool = True,
) -> tuple[str, str, IntentGateResult | None, StepPrepareResult | None]:
    """入口通道 → 脏格式净化 → 敏感词。闲聊/模糊交给图内规则+BERT。"""
    _ = skip_intent_gate
    reject_empty_text(raw_text)
    validate_text_length(raw_text)

    normalized_channel = normalize_user_text(raw_text, channel=input_channel)
    if not normalized_channel:
        raise EmptyTextError("文本不能为空")

    cleaned = sanitize_user_text(normalized_channel)
    if not cleaned:
        raise EmptyTextError("文本不能为空")

    guard_sensitive_text(cleaned)

    return raw_text.strip(), cleaned, None, None


def prepare_step_update(
    *,
    thread_id: str,
    message: str | None = None,
    text: str | None = None,
    selected_body_part: str | None = None,
    selected_disease: str | None = None,
    selected_symptoms: list[str] | None = None,
    org_id: str | None = None,
    input_channel: InputChannel | None = None,
    request: Request | None = None,
) -> StepPrepareResult:
    """四层管道后生成 graph update；拦截类错误抛 StepGatewayError / 接入层异常。"""
    sid = (thread_id or "").strip()
    if not sid:
        raise StepGatewayError("thread_id 无效")

    raw_message = (message or text or "").strip()
    has_options = _has_option_payload(
        selected_body_part=selected_body_part,
        selected_disease=selected_disease,
        selected_symptoms=selected_symptoms,
        org_id=org_id,
    )
    if not raw_message and not has_options:
        raise StepGatewayError("需要提交选项或消息")

    header_channel = request.headers.get("X-Input-Channel") if request is not None else None
    channel = input_channel or resolve_input_channel(None, header_channel)

    update: dict[str, Any] = {}
    intent: IntentGateResult | None = None

    if raw_message:
        try:
            raw_query, query, intent, blocked = _process_free_text(
                raw_message,
                input_channel=channel,
                skip_intent_gate=has_options,
            )
        except SensitiveContentError as exc:
            return StepPrepareResult(
                update=None,
                blocked=True,
                block_category="sensitive",
                gate_message=str(exc),
                input_channel=channel,
            )
        except EmptyTextError:
            return StepPrepareResult(
                update=None,
                blocked=True,
                block_category="empty_text",
                gate_message="文本不能为空",
                input_channel=channel,
            )

        if blocked is not None:
            return blocked

        update["raw_query"] = raw_query
        update["query"] = query
        update["input_channel"] = channel
        update["messages"] = [HumanMessage(content=query)]

    if selected_body_part:
        update["selected_body_part"] = selected_body_part.strip()
    if selected_disease:
        update["selected_disease"] = selected_disease.strip()
    if selected_symptoms:
        update["selected_symptoms"] = [item.strip() for item in selected_symptoms if item.strip()]
    if org_id:
        update["org_id"] = org_id.strip()

    return StepPrepareResult(
        update=update,
        blocked=False,
        intent=intent,
        input_channel=channel,
    )
