"""用户口语实体 → Neo4j 入口节点。同义词表 → Milvus 向量检索。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from knowledge.milvus_align import get_milvus_align_repository
from schema.entity_synonyms import lookup_synonym

logger = logging.getLogger(__name__)

__all__ = ["AlignHit", "align_symptoms_to_catalog", "entity_align", "EMBED_ACCEPT_DISTANCE"]

EntityType = Literal["body_part", "disease", "symptom"]
# Milvus 只做粗召回；是否锁名由 DeepSeek 精排决定
EMBED_ACCEPT_DISTANCE = 0.5


@dataclass(frozen=True)
class AlignHit:
    mention: str
    standard_name: str | None
    node_id: int | None
    node_label: str | None
    method: str
    score: float
    needs_review: bool = False

    @property
    def ok(self) -> bool:
        return bool(self.standard_name) and not self.needs_review


def _name_of(row: dict[str, Any], name_key: str) -> str:
    return str(row.get(name_key) or "").strip()


def _row_id(row: dict[str, Any], name_key: str) -> str:
    if row.get("id") is not None:
        return str(row["id"])
    return _name_of(row, name_key)


def _find_row(catalog: list[dict[str, Any]], name_key: str, name: str) -> dict[str, Any] | None:
    target = (name or "").strip()
    if not target:
        return None
    for row in catalog:
        if _name_of(row, name_key) == target:
            return row
    for row in catalog:
        std = _name_of(row, name_key)
        if std and (target in std or std in target):
            return row
    return None


def _find_row_by_id(catalog: list[dict[str, Any]], name_key: str, node_id: str | None) -> dict[str, Any] | None:
    if not node_id:
        return None
    for row in catalog:
        if _row_id(row, name_key) == str(node_id):
            return row
    return None


def _parse_node_id(raw: str | None) -> int | None:
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _allowed_names(catalog: list[dict[str, Any]], name_key: str) -> set[str]:
    return {_name_of(row, name_key) for row in catalog if _name_of(row, name_key)}


def entity_align(
    entity_type: EntityType,
    mention: str,
    catalog: list[dict[str, Any]],
    *,
    name_key: str,
    node_label: str,
    for_span: bool = False,
    utterance: str | None = None,
) -> AlignHit:
    """入槽后对齐：同义词表 → 标准名 → Milvus 粗召回 + DeepSeek 精排。"""
    text = (mention or "").strip()
    if not text:
        return AlignHit(text, None, None, node_label, "empty", 0.0, True)

    alias = lookup_synonym(entity_type, text)
    if alias:
        row = _find_row(catalog, name_key, alias)
        if row:
            logger.info("同义词表: %s -> %s", text, _name_of(row, name_key))
            return AlignHit(
                mention=text,
                standard_name=_name_of(row, name_key),
                node_id=row.get("id") if isinstance(row.get("id"), int) else _parse_node_id(str(row.get("id"))),
                node_label=node_label,
                method="synonym",
                score=1.0,
            )

    exact = _find_row(catalog, name_key, text)
    if exact:
        return AlignHit(
            text,
            _name_of(exact, name_key),
            exact.get("id") if isinstance(exact.get("id"), int) else _parse_node_id(_row_id(exact, name_key)),
            node_label,
            "exact",
            1.0,
        )

    # 整句不打 Milvus；已抽好的短 span（for_span）只拦超长
    too_long = len(text) > 24 if for_span else (len(text) > 12 or any(mark in text for mark in "。！？，、呢吧啊呀"))
    if too_long:
        return AlignHit(text, None, None, node_label, "skip_milvus", 0.0, True)

    allowed = _allowed_names(catalog, name_key) or None
    try:
        std_name, distance, raw_id = get_milvus_align_repository().get_standard_word_by_synonym(
            entity_type,
            text,
            allowed_names=allowed,
            max_distance=EMBED_ACCEPT_DISTANCE,
            utterance=utterance,
        )
    except Exception as exc:
        logger.warning("Milvus 对齐失败: %s", exc)
        return AlignHit(text, None, None, node_label, "miss", 0.0, True)

    row = _find_row_by_id(catalog, name_key, raw_id) or _find_row(catalog, name_key, std_name or "")
    node_id = row.get("id") if row and isinstance(row.get("id"), int) else _parse_node_id(raw_id)
    standard = _name_of(row, name_key) if row else std_name
    score = 0.9 if standard else 0.0

    if standard:
        logger.info("Milvus+精排对齐: %s -> %s (distance=%s)", text, standard, distance)
        return AlignHit(
            mention=text,
            standard_name=standard,
            node_id=node_id,
            node_label=node_label,
            method="milvus_rerank",
            score=score,
        )

    logger.info("精排未锁定: %s (distance=%s)", text, distance)
    return AlignHit(
        mention=text,
        standard_name=standard,
        node_id=node_id,
        node_label=node_label,
        method="milvus",
        score=score,
        needs_review=True,
    )


def align_symptoms_to_catalog(
    mentions: list[str],
    catalog: list[str],
    *,
    drop_equal_to: str | None = None,
) -> tuple[list[str], list[AlignHit]]:
    """已入槽的症状 span → 本轮候选项标准名（同义词 / 精确 / Milvus）。

    对齐后的标准名若与 drop_equal_to（已锁疾病标准名）完全相同，丢弃该条。
    """
    names = [str(item).strip() for item in catalog if str(item).strip()]
    rows = [{"id": None, "symptom_name": name} for name in names]
    matched: list[str] = []
    hits: list[AlignHit] = []
    if not rows:
        return [], hits
    skip = (drop_equal_to or "").strip()
    seen: set[str] = set()
    for mention in mentions:
        text = str(mention or "").strip()
        if not text:
            continue
        hit = entity_align(
            "symptom",
            text,
            rows,
            name_key="symptom_name",
            node_label="Symptom",
            for_span=True,
        )
        std = (hit.standard_name or "").strip()
        if skip and std == skip:
            continue
        hits.append(hit)
        if hit.ok and std and std in names and std not in seen:
            seen.add(std)
            matched.append(std)
    return matched, hits
