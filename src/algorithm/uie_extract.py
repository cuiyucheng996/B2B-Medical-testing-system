"""UIE 实体抽取：对齐老师 ChatService.extract_entity（set_schema → 抽 span → {类型: [文本]}）。"""

from __future__ import annotations

import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from configs.b2b_api import get_b2b_api_config
from configs.paths import UVGRAPH_ROOT, ensure_uvgraph_root
from schema.slot_schema import UIE_SCHEMA

logger = logging.getLogger(__name__)

__all__ = [
    "extract_attr",
    "extract_entity",
    "get_uie_predictor",
]


def _uie_code_root() -> Path:
    return UVGRAPH_ROOT / "External" / "uie_pytorch"


def _resolve_task_path() -> Path:
    cfg = get_b2b_api_config()
    raw = str(cfg.get("uie_task_path") or "").strip()
    if raw:
        path = Path(raw)
        if not path.is_absolute():
            path = UVGRAPH_ROOT / path
        return path
    b2c = UVGRAPH_ROOT / "B2C business" / "pretrained" / "uie_base_pytorch"
    if (b2c / "pytorch_model.bin").is_file():
        return b2c
    ensure_uvgraph_root()
    from src.uie.model_cache import ensure_uie_model

    model_name = str(cfg.get("uie_model", "uie-base"))
    return ensure_uie_model(model_name)


@lru_cache(maxsize=1)
def get_uie_predictor():
    cfg = get_b2b_api_config()
    if not cfg.get("uie_enabled", True):
        return None
    code_root = str(_uie_code_root())
    if code_root not in sys.path:
        sys.path.insert(0, code_root)
    try:
        from uie_predictor import UIEPredictor
    except ImportError:
        logger.warning("未找到 uie_predictor，跳过 UIE 抽取")
        return None

    task_path = _resolve_task_path()
    if not task_path.exists():
        logger.warning("UIE 权重目录不存在: %s", task_path)
        return None

    device = "gpu" if _cuda_available() else "cpu"
    try:
        predictor = UIEPredictor(
            model=str(cfg.get("uie_model", "uie-base")),
            schema=list(UIE_SCHEMA),
            task_path=str(task_path),
            schema_lang="zh",
            engine="pytorch",
            device=device,
            position_prob=float(cfg.get("uie_position_prob", 0.5)),
            max_seq_len=int(cfg.get("uie_max_seq_len", 512)),
        )
        logger.info("UIE 已加载: %s device=%s", task_path, device)
        return predictor
    except Exception as exc:
        logger.warning("加载 UIE 失败: %s", exc)
        return None


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _flatten_uie_block(block: dict[str, Any]) -> dict[str, list[str]]:
    """老师：result[key] = [item['text'] for item in result[key]]。"""
    flattened: dict[str, list[str]] = {}
    for key, value in (block or {}).items():
        if not isinstance(value, list):
            continue
        texts: list[str] = []
        for item in value:
            if isinstance(item, dict) and item.get("text"):
                texts.append(str(item["text"]).strip())
            elif isinstance(item, str) and item.strip():
                texts.append(item.strip())
        flattened[key] = list(dict.fromkeys(t for t in texts if t))
    return flattened


def extract_entity(question: str, schema: list[str] | None = None) -> dict[str, list[str]]:
    """按当前意图 schema 抽实体。schema 为空则不抽。"""
    labels = [item for item in (schema or []) if item]
    if not (question or "").strip() or not labels:
        return {key: [] for key in labels}

    predictor = get_uie_predictor()
    if predictor is None:
        return {key: [] for key in labels}

    predictor.set_schema(labels)
    try:
        raw = predictor(question)
    except Exception as exc:
        logger.warning("UIE 抽取失败: %s", exc)
        return {key: [] for key in labels}

    block = raw[0] if isinstance(raw, list) and raw else (raw if isinstance(raw, dict) else {})
    flattened = _flatten_uie_block(block if isinstance(block, dict) else {})
    return {key: list(flattened.get(key) or []) for key in labels}


def extract_attr(question: str, catalog: list[str] | None = None) -> str:
    """老师 extract_attr：在全局 schema 上抽到的类型里取最长匹配。"""
    found = {key for key, spans in extract_entity(question, UIE_SCHEMA).items() if spans}
    if catalog:
        found &= set(catalog)
    return max(found, key=len, default="")
