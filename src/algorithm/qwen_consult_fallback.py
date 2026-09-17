"""qwen-turbo 降级：BERT 非主诉或 UIE 抽空时，以大模型意图+span 为准。"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from configs.b2b_api import get_b2b_api_config
from configs.paths import ensure_uvgraph_root

logger = logging.getLogger(__name__)

__all__ = [
    "QwenConsultFallback",
    "judge_extracted_negation",
    "run_qwen_consult_fallback",
]

_POLARITY_SYSTEM = """你判断中医问诊里已抽出的实体在原句中是不是被否定。
对每个实体只看它在原句里的用法：
- 否定/否认该不适（没有头疼、头不痛、胸口不是那么疼）→ negated true，不入槽
- 纠偏后半句才是主诉（不是说头不痛是头很痛）→ 「头」「痛」negated false，「头不痛」true
- 没有力气、睡不着是在说症状 → negated false
- 不是很疼、有点疼是程度，仍在说该症状 → negated false
只输出 JSON，不要其它文字：
{"items":[{"span":"胸口","type":"部位","negated":true,"reason":"不是那么疼是否认程度"}]}"""

FallbackIntent = Literal["complaint", "chitchat", "vague"]

_JSON_RE = re.compile(r"\{[\s\S]*\}")

_SYSTEM = """你是中医问诊预处理。根据用户原话判断意图，并抽出原文中的片段（不要改写成标准名）。
意图只允许：complaint（在说身体不适/受伤/部位症状）、chitchat（寒暄）、vague（完全说不清）。
部位片段应对应头面部/颈部/胸胁部/胃腹部/二阴部/四肢躯干之一所能覆盖的说法，例如腿、胳膊、腰、脑袋。
被「没有/不是/不疼」否定的部位、疾病、症状不要抽取。
只输出一个 JSON，不要其它文字：
{"intent":"complaint","部位":["腿"],"疾病":[],"症状":["破了"],"reason":"简短理由"}
没有的键用空数组。片段必须是用户原句中出现过的连续字。"""


@dataclass
class QwenConsultFallback:
    intent: FallbackIntent
    entities: dict[str, list[str]] = field(default_factory=dict)
    reason: str = ""


def _spans_in_text(text: str, items: Any) -> list[str]:
    out: list[str] = []
    if not isinstance(items, list):
        return out
    for item in items:
        span = str(item or "").strip()
        if span and span in text and span not in out:
            out.append(span)
    return out


def _parse_payload(raw: str, text: str) -> QwenConsultFallback | None:
    match = _JSON_RE.search(raw or "")
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    intent_raw = str(data.get("intent") or "vague").strip()
    intent: FallbackIntent = "vague"
    if intent_raw in ("complaint", "chitchat", "vague"):
        intent = intent_raw  # type: ignore[assignment]
    entities = {
        "部位": _spans_in_text(text, data.get("部位")),
        "疾病": _spans_in_text(text, data.get("疾病")),
        "症状": _spans_in_text(text, data.get("症状")),
    }
    reason = str(data.get("reason") or "qwen-turbo 降级").strip()
    return QwenConsultFallback(intent=intent, entities=entities, reason=reason)


def run_qwen_consult_fallback(text: str, *, major_round: int = 1) -> QwenConsultFallback | None:
    query = (text or "").strip()
    if not query:
        return None
    cfg = get_b2b_api_config()
    if not cfg.get("qwen_fallback_enabled", True):
        return None
    ensure_uvgraph_root()
    try:
        from src.configs import get_app_settings
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
    except Exception:
        logger.warning("qwen fallback imports failed")
        return None

    settings = get_app_settings()
    api_key = (getattr(settings, "dashscope_api_key", None) or "").strip()
    if not api_key:
        logger.warning("qwen fallback skipped: DASHSCOPE_API_KEY empty")
        return None

    model = str(cfg.get("qwen_fallback_model") or "qwen-turbo")
    base_url = str(
        cfg.get("qwen_fallback_base_url")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    user = f"当前问诊第{int(major_round)}轮。用户原话：{query}"
    try:
        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0,
            max_tokens=400,
        )
        resp = llm.invoke(
            [SystemMessage(content=_SYSTEM), HumanMessage(content=user)]
        )
        content = str(getattr(resp, "content", "") or "")
    except Exception:
        logger.warning("qwen-turbo fallback call failed")
        return None
    parsed = _parse_payload(content, query)
    if parsed is None:
        logger.warning("qwen fallback json parse failed")
        return None
    logger.warning(
        "qwen fallback used intent=%s parts=%s symptoms=%s",
        parsed.intent,
        parsed.entities.get("部位"),
        parsed.entities.get("症状"),
    )
    return parsed


def _dashscope_chat():
    cfg = get_b2b_api_config()
    if not cfg.get("qwen_fallback_enabled", True):
        return None
    ensure_uvgraph_root()
    try:
        from src.configs import get_app_settings
        from langchain_openai import ChatOpenAI
    except Exception:
        return None
    settings = get_app_settings()
    api_key = (getattr(settings, "dashscope_api_key", None) or "").strip()
    if not api_key:
        return None
    model = str(cfg.get("qwen_fallback_model") or "qwen-turbo")
    base_url = str(
        cfg.get("qwen_fallback_base_url")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        max_tokens=400,
    )


def judge_extracted_negation(
    text: str,
    extracted: dict[str, list[str]],
) -> list[dict[str, Any]] | None:
    """抽槽之后：qwen 判断每个 span 是否否定。失败返回 None。"""
    query = (text or "").strip()
    rows: list[dict[str, str]] = []
    for key in ("部位", "疾病", "症状"):
        for span in extracted.get(key) or []:
            item = str(span or "").strip()
            if item:
                rows.append({"span": item, "type": key})
    if not query or not rows:
        return []
    llm = _dashscope_chat()
    if llm is None:
        return None
    numbered = "\n".join(
        f"{idx}. type={row['type']} span={row['span']}" for idx, row in enumerate(rows, start=1)
    )
    user = f"用户原话：{query}\n已抽出实体：\n{numbered}"
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        resp = llm.invoke(
            [SystemMessage(content=_POLARITY_SYSTEM), HumanMessage(content=user)]
        )
        content = str(getattr(resp, "content", "") or "")
    except Exception:
        logger.warning("qwen polarity call failed")
        return None
    match = _JSON_RE.search(content)
    if not match:
        logger.warning("qwen polarity json parse failed")
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        logger.warning("qwen polarity json parse failed")
        return None
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return None
    allowed = {(row["type"], row["span"]) for row in rows}
    judged: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        span = str(item.get("span") or "").strip()
        kind = str(item.get("type") or "").strip()
        if (kind, span) not in allowed:
            continue
        judged.append(
            {
                "span": span,
                "type": kind,
                "negated": bool(item.get("negated")),
                "reason": str(item.get("reason") or "").strip(),
            }
        )
    logger.warning("qwen polarity judged=%s", judged)
    return judged
