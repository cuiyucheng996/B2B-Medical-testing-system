"""纠错后句子意图：规则优先，BERT 兜底（对齐老师 IntentClassifyBertPredictor）。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from algorithm.intent_rule import match_intent
from configs.b2b_api import get_b2b_api_config
from configs.paths import B2B_ROOT, UVGRAPH_ROOT
from schema.b2b import TextIntent

logger = logging.getLogger(__name__)

__all__ = [
    "IntentClassifyBertPredictor",
    "IntentClassifyResult",
    "classify_sentence_intent",
    "get_intent_model_dir",
]


@dataclass(frozen=True)
class IntentClassifyResult:
    intent: TextIntent
    confidence: float
    reason: str
    source: str
    recommend_action: str = "正常处理"
    is_other: bool = False


def get_intent_model_dir() -> Path:
    cfg = get_b2b_api_config()
    raw = str(cfg.get("intent_classify_model_dir", "")).strip()
    if raw:
        path = Path(raw)
        if not path.is_absolute():
            path = B2B_ROOT / path
        return path
    return B2B_ROOT / "checkpoint" / "intent_classify" / "best_model"


def _normalize_label(label: str) -> TextIntent:
    if label in ("complaint", "chitchat", "vague"):
        return label
    return "vague"


class IntentClassifyBertPredictor:
    def __init__(self, model_path: Path | str | None = None) -> None:
        import torch
        from transformers import BertForSequenceClassification, BertTokenizer

        model_dir = Path(model_path) if model_path else get_intent_model_dir()
        mapping_path = model_dir.parent / "label_mapping.json"
        self.model = BertForSequenceClassification.from_pretrained(str(model_dir))
        self.tokenizer = BertTokenizer.from_pretrained(str(model_dir))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        with mapping_path.open(encoding="utf-8") as f:
            raw_mapping = json.load(f)
        self.label_mapping = {str(k): str(v) for k, v in raw_mapping.items()}

    def predict_intent(
        self,
        text: str,
        *,
        max_length: int | None = None,
        threshold: float | None = None,
    ) -> IntentClassifyResult:
        import torch

        cfg = get_b2b_api_config()
        max_len = int(max_length or cfg.get("intent_classify_max_length", 128))
        cutoff = float(threshold if threshold is not None else cfg.get("intent_classify_confidence_threshold", 0.7))

        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            pred_id = int(torch.argmax(probs, dim=-1).item())
            confidence = float(probs[pred_id].item())

        raw_label = self.label_mapping.get(str(pred_id), "unknown")
        is_other = raw_label in ("other", "unknown", "未知意图")
        if is_other:
            return IntentClassifyResult(
                intent="vague",
                confidence=confidence,
                reason=f"BERT 预测为 {raw_label}",
                source="bert",
                recommend_action="转人工或触发兜底回复",
                is_other=True,
            )
        if confidence < cutoff:
            return IntentClassifyResult(
                intent="vague",
                confidence=confidence,
                reason=f"BERT 低置信({confidence:.2f})，预测为 {raw_label}",
                source="bert",
                recommend_action="建议人工复核",
            )
        intent = _normalize_label(raw_label)
        return IntentClassifyResult(
            intent=intent,
            confidence=confidence,
            reason=f"BERT 预测为 {intent}",
            source="bert",
            recommend_action="正常处理",
        )


@lru_cache(maxsize=1)
def _get_predictor() -> IntentClassifyBertPredictor | None:
    model_dir = get_intent_model_dir()
    mapping_path = model_dir.parent / "label_mapping.json"
    if not model_dir.is_dir() or not mapping_path.is_file():
        logger.info("意图 BERT 模型未就绪，仅用规则: %s", model_dir)
        return None
    try:
        return IntentClassifyBertPredictor(model_dir)
    except Exception as exc:
        logger.warning("加载意图 BERT 失败，仅用规则: %s", exc)
        return None


def classify_sentence_intent(text: str) -> IntentClassifyResult:
    """规则命中即返回；否则 BERT；无模型则 vague。"""
    normalized = (text or "").strip()
    if not normalized:
        return IntentClassifyResult("vague", 1.0, "输入为空", "rule", "触发兜底回复")

    ruled = match_intent(normalized)
    if ruled is not None:
        return IntentClassifyResult(ruled, 0.9, "规则命中", "rule")

    predictor = _get_predictor()
    if predictor is not None:
        return predictor.predict_intent(normalized)
    return IntentClassifyResult("vague", 0.4, "规则未命中且无 BERT 模型", "fallback", "触发兜底回复")


def default_pretrained_dir() -> Path:
    return UVGRAPH_ROOT / "B2C business" / "pretrained" / "bert-base-chinese"
