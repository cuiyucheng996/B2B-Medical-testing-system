"""B2B 文本纠错：复用 B2C business 纠错 Agent / BERT 兜底。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from configs.b2b_api import get_b2b_api_config
from configs.paths import ensure_b2c_src

logger = logging.getLogger(__name__)

_agent_unavailable: bool = False
_bert_unavailable: bool = False

__all__ = [
    "SpellCorrectResult",
    "build_spell_correction_state_update",
    "correct_user_text",
]


@dataclass(frozen=True)
class SpellCorrectResult:
    original_text: str
    corrected_text: str
    changed: bool
    source: str
    confidence: float | None = None
    explanation: str = ""
    detail: dict[str, Any] | None = None


def _correction_detail(
    *,
    confidence: float | None,
    explanation: str,
    errors: list[dict[str, Any]] | None = None,
    chunks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """纠错对账结构；整句原文/纠正只写 raw_query / query。"""
    return {
        "errors": list(errors or []),
        "confidence": confidence if confidence is not None else 1.0,
        "explanation": explanation,
        "chunks": list(chunks or []),
    }


def _detail_without_sentence_dup(detail: dict[str, Any]) -> dict[str, Any]:
    stripped = dict(detail)
    stripped.pop("original_text", None)
    stripped.pop("corrected_text", None)
    return stripped


def build_spell_correction_state_update(result: SpellCorrectResult) -> dict[str, Any]:
    """业务字段 + 纠错对账结构写入 state。"""
    update: dict[str, Any] = {
        "raw_query": result.original_text,
        "query": result.corrected_text,
    }
    if result.detail is not None:
        update["spell_correction_detail"] = _detail_without_sentence_dup(result.detail)
    if result.changed:
        update["spell_correction"] = {
            "source": result.source,
            "confidence": result.confidence,
            "explanation": result.explanation,
        }
    return update


def _bert_correct(text: str) -> SpellCorrectResult | None:
    global _bert_unavailable
    if _bert_unavailable:
        return None
    try:
        ensure_b2c_src()
        from runner.Predictor import get_spell_check_bert_predictor

        corrected = get_spell_check_bert_predictor().predict(text)
        corrected = (corrected or "").strip() or text
        changed = corrected != text
        return SpellCorrectResult(
            original_text=text,
            corrected_text=corrected,
            changed=changed,
            source="bert",
            confidence=None,
            explanation="BERT 逐字纠错",
            detail=_correction_detail(
                confidence=None if changed else 1.0,
                explanation="BERT 逐字纠错",
            ),
        )
    except Exception as exc:
        _bert_unavailable = True
        logger.warning("BERT spell fallback unavailable: %s", exc)
        return None


def _agent_correct(text: str, *, require_high_confidence: bool) -> SpellCorrectResult | None:
    global _agent_unavailable
    if _agent_unavailable:
        return None
    try:
        ensure_b2c_src()
        from agent.spell_check_agent import correct_text

        result = correct_text(
            text,
            require_high_confidence=require_high_confidence,
        )
        corrected = (result.corrected_text or "").strip() or text
        return SpellCorrectResult(
            original_text=text,
            corrected_text=corrected,
            changed=corrected != text,
            source="agent",
            confidence=float(result.confidence),
            explanation=result.explanation or "",
            detail=result.model_dump(),
        )
    except Exception as exc:
        _agent_unavailable = True
        logger.warning("spell_check agent unavailable: %s", exc)
        return None


def correct_user_text(text: str) -> SpellCorrectResult:
    """对用户自由文本纠错；失败时原样返回。"""
    raw = (text or "").strip()
    if not raw:
        empty_detail = _correction_detail(
            confidence=1.0,
            explanation="空文本",
        )
        return SpellCorrectResult(
            original_text="",
            corrected_text="",
            changed=False,
            source="skip",
            explanation="空文本",
            detail=empty_detail,
        )

    cfg = get_b2b_api_config()
    if not cfg["spell_check_enabled"]:
        detail = _correction_detail(
            confidence=1.0,
            explanation="纠错已关闭",
        )
        return SpellCorrectResult(
            original_text=raw,
            corrected_text=raw,
            changed=False,
            source="disabled",
            explanation="纠错已关闭",
            detail=detail,
        )

    if cfg["spell_check_use_llm_agent"]:
        agent_result = _agent_correct(
            raw,
            require_high_confidence=bool(cfg["spell_check_require_high_confidence"]),
        )
        if agent_result is not None:
            return agent_result

    if cfg["spell_check_use_bert_fallback"]:
        bert_result = _bert_correct(raw)
        if bert_result is not None:
            return bert_result

    detail = _correction_detail(
        confidence=1.0,
        explanation="纠错不可用，保留原文",
    )
    return SpellCorrectResult(
        original_text=raw,
        corrected_text=raw,
        changed=False,
        source="passthrough",
        explanation="纠错不可用，保留原文",
        detail=detail,
    )


def apply_spell_correction_to_state(state: dict[str, Any]) -> dict[str, Any]:
    """将 state 中的 query 纠错后写回，保留 raw_query 与完整 detail。"""
    from algorithm.text_input import extract_query

    raw = extract_query(state)
    if not raw:
        return {}

    return build_spell_correction_state_update(correct_user_text(raw))
