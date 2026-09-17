"""B2B API 接入层与入口通道配置（读主项目 config.yaml + .env）。"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from configs.paths import ensure_uvgraph_root

__all__ = ["get_b2b_api_config"]


def _section_dict(data: dict[str, Any], section: str) -> dict[str, Any]:
    section_data = data.get(section)
    if not isinstance(section_data, dict):
        return {}
    return {k: v for k, v in section_data.items() if v is not None}


@lru_cache
def get_b2b_api_config() -> dict[str, Any]:
    """返回 B2B API 鉴权、限流与输入管道配置。"""
    ensure_uvgraph_root()
    from src.configs.config_loader import get_yaml_config

    cfg = _section_dict(get_yaml_config(), "b2b_api")
    rate = cfg.get("rate_limit") if isinstance(cfg.get("rate_limit"), dict) else {}
    input_cfg = cfg.get("input") if isinstance(cfg.get("input"), dict) else {}
    asr_cfg = cfg.get("asr_normalize") if isinstance(cfg.get("asr_normalize"), dict) else {}
    spell_cfg = cfg.get("spell_check") if isinstance(cfg.get("spell_check"), dict) else {}
    intent_cfg = cfg.get("intent_classify") if isinstance(cfg.get("intent_classify"), dict) else {}
    uie_cfg = cfg.get("uie") if isinstance(cfg.get("uie"), dict) else {}
    reply_cfg = cfg.get("reply_agent") if isinstance(cfg.get("reply_agent"), dict) else {}
    qwen_cfg = cfg.get("qwen_fallback") if isinstance(cfg.get("qwen_fallback"), dict) else {}
    rerank_cfg = cfg.get("align_rerank") if isinstance(cfg.get("align_rerank"), dict) else {}

    default_channel = str(input_cfg.get("default_channel", "text")).strip().lower()
    if default_channel not in ("text", "asr"):
        default_channel = "text"

    return {
        "api_key_required": bool(cfg.get("api_key_required", False)),
        "api_key": os.getenv("B2B_API_KEY", "").strip(),
        "rate_limit_enabled": bool(rate.get("enabled", True)),
        "rate_limit_max_requests": int(rate.get("max_requests", 60)),
        "rate_limit_window_seconds": int(rate.get("window_seconds", 60)),
        "max_text_length": int(input_cfg.get("max_text_length", 2000)),
        "default_input_channel": default_channel,
        "asr_collapse_spaces": bool(asr_cfg.get("collapse_spaces", True)),
        "asr_strip_punctuation_noise": bool(asr_cfg.get("strip_punctuation_noise", True)),
        "spell_check_enabled": bool(spell_cfg.get("enabled", True)),
        "spell_check_use_llm_agent": bool(spell_cfg.get("use_llm_agent", True)),
        "spell_check_use_bert_fallback": bool(spell_cfg.get("use_bert_fallback", True)),
        "spell_check_require_high_confidence": bool(spell_cfg.get("require_high_confidence", True)),
        "intent_classify_model_dir": str(intent_cfg.get("model_dir", "checkpoint/intent_classify/best_model")),
        "intent_classify_confidence_threshold": float(intent_cfg.get("confidence_threshold", 0.7)),
        "intent_classify_max_length": int(intent_cfg.get("max_length", 128)),
        "uie_enabled": bool(uie_cfg.get("enabled", True)),
        "uie_model": str(uie_cfg.get("model", "uie-base")),
        "uie_task_path": str(uie_cfg.get("task_path") or ""),
        "uie_position_prob": float(uie_cfg.get("position_prob", 0.5)),
        "uie_max_seq_len": int(uie_cfg.get("max_seq_len", 512)),
        "reply_agent_enabled": bool(reply_cfg.get("enabled", True)),
        "qwen_fallback_enabled": bool(qwen_cfg.get("enabled", True)),
        "qwen_fallback_model": str(qwen_cfg.get("model", "qwen-turbo")),
        "qwen_fallback_base_url": str(
            qwen_cfg.get("base_url") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        "align_rerank_enabled": bool(rerank_cfg.get("enabled", True)),
        "align_rerank_recall_k": int(rerank_cfg.get("recall_k", 10)),
    }
