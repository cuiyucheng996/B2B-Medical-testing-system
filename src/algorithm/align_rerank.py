"""Milvus 粗召回后的 DeepSeek 精排：从候选标准名里选一个或空。"""

from __future__ import annotations

import json
import logging
import re
from typing import Literal

from algorithm.input_parse import is_systemic_unlocalized_text
from configs.b2b_api import get_b2b_api_config
from configs.paths import ensure_uvgraph_root

logger = logging.getLogger(__name__)

__all__ = ["rerank_align_candidates"]

EntityType = Literal["body_part", "disease", "symptom"]

_JSON_RE = re.compile(r"\{[\s\S]*\}")

_TYPE_HINT = {
    "body_part": "部位（头面部/颈部/胸胁部/胃腹部/二阴部/四肢躯干等）",
    "disease": "疾病名（如头痛、咳嗽、水肿），不是乏力、没劲、使不上劲这类症状描述",
    "symptom": "症状名（如乏力、头痛、刺痛），不是疾病诊断名",
}


def rerank_align_candidates(
    mention: str,
    candidates: list[str],
    *,
    entity_type: EntityType,
    utterance: str | None = None,
) -> str | None:
    """只允许返回候选列表中的标准名；不像该类型则返回 None。"""
    query = (mention or "").strip()
    names = list(dict.fromkeys(str(item).strip() for item in candidates if str(item).strip()))
    if not query or not names:
        return None
    # 粗召回可以有「浑身」；精排必须刷掉，不能锁成四肢躯干等单一标准部位
    if entity_type == "body_part" and is_systemic_unlocalized_text(query):
        logger.warning("align rerank drop systemic body mention=%s recall=%s", query, names[:8])
        return None
    cfg = get_b2b_api_config()
    if not cfg.get("align_rerank_enabled", True):
        return names[0]

    ensure_uvgraph_root()
    try:
        from src.configs import get_app_settings
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
    except Exception:
        logger.warning("align rerank imports failed")
        return None

    settings = get_app_settings()
    api_key = (getattr(settings, "deepseek_api_key", None) or "").strip()
    if not api_key:
        logger.warning("align rerank skipped: DEEPSEEK_API_KEY empty")
        return None

    numbered = "\n".join(f"{idx}. {name}" for idx, name in enumerate(names, start=1))
    source = (utterance or "").strip()
    user = f"口语片段：{query}\n"
    if source and source != query:
        user += f"原句：{source}\n"
    user += f"候选：\n{numbered}"
    system = (
        "你是中医问诊对齐精排。根据用户口语片段，从候选标准名中选一个最贴切的。"
        f"当前要对齐的类型是：{_TYPE_HINT.get(entity_type, entity_type)}。"
        "如果口语明显不是这个类型、或列表里没有足够贴切的项，name 必须输出空字符串。"
        "若原句用没有/不是/不疼等否定了该片段，name 必须空，不能锁成标准部位或病名。"
        "部位对齐时：浑身、全身、周身、满身这类无法落到头面/颈/胸胁/胃腹/二阴/四肢之一的，必须空，禁止猜四肢躯干。"
        "只能选列表里出现过的原名，禁止自造。"
        '只输出一个 JSON：{"name":"标准名或空","reason":"一句理由"}'
    )
    try:
        llm = ChatOpenAI(
            model=str(getattr(settings, "deepseek_model", None) or "deepseek-chat"),
            api_key=api_key,
            base_url=str(getattr(settings, "deepseek_base_url", None) or "https://api.deepseek.com/v1"),
            temperature=0,
            max_tokens=200,
        )
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        content = str(getattr(resp, "content", "") or "")
    except Exception:
        logger.warning("align rerank deepseek call failed")
        return None

    match = _JSON_RE.search(content)
    if not match:
        logger.warning("align rerank json parse failed")
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        logger.warning("align rerank json parse failed")
        return None
    if not isinstance(data, dict):
        return None
    picked = str(data.get("name") or "").strip()
    if not picked or picked not in names:
        logger.warning("align rerank empty mention=%s type=%s", query, entity_type)
        return None
    logger.warning(
        "align rerank picked mention=%s type=%s -> %s",
        query,
        entity_type,
        picked,
    )
    return picked
