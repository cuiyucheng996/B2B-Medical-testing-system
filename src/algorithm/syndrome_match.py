"""证型缩窄与症状选项算法。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from algorithm.entity_align import AlignHit, align_symptoms_to_catalog
from knowledge.neo4j_consult import (
    filter_syndromes_by_auxiliary_symptoms_graph,
    filter_syndromes_by_main_symptoms_graph,
    list_all_syndrome_main_symptoms,
)


RollbackReply = Literal["yes", "no", "unclear"]

_ROLLBACK_YES = ("回退", "重选", "重新选", "换病", "换一个", "换疾病", "回到第二轮")
_ROLLBACK_NO = ("不", "否", "不用", "别", "继续", "就这个", "不用换", "先按")
_STAY_AFFIRM = {"是", "是的", "对", "对的", "好", "好的", "嗯"}


@dataclass
class GlobalMainSymptomMatch:
    current_hits: list[str] = field(default_factory=list)
    other_by_disease: dict[str, list[str]] = field(default_factory=dict)
    alignments: list[AlignHit] = field(default_factory=list)

    @property
    def other_diseases(self) -> list[str]:
        return list(self.other_by_disease.keys())


def build_global_main_symptom_catalog() -> tuple[list[str], dict[str, list[str]]]:
    """全库证型主症标准名，以及标准名 → 疾病名列表。"""
    names: list[str] = []
    diseases_of: dict[str, list[str]] = {}
    seen: set[str] = set()
    for row in list_all_syndrome_main_symptoms():
        name = str(row.get("symptom_name") or "").strip()
        disease = str(row.get("disease_name") or "").strip()
        if not name or not disease:
            continue
        if name not in seen:
            seen.add(name)
            names.append(name)
        bucket = diseases_of.setdefault(name, [])
        if disease not in bucket:
            bucket.append(disease)
    return names, diseases_of


def ok_aligned_symptom_names(alignments: list[Any] | None) -> list[str]:
    """slots.alignments 里已成功锁定的症状标准名（部位/病对齐不算）。"""
    names: list[str] = []
    seen: set[str] = set()
    for row in alignments or []:
        if isinstance(row, AlignHit):
            if not row.ok or row.needs_review:
                continue
            label = row.node_label
            std = (row.standard_name or "").strip()
        elif isinstance(row, dict):
            if row.get("needs_review"):
                continue
            std = str(row.get("standard_name") or "").strip()
            label = row.get("node_label")
        else:
            continue
        if not std or std in seen:
            continue
        if label in ("ConsultBodyPart", "ConsultDisease"):
            continue
        seen.add(std)
        names.append(std)
    return names


def names_in_main_symptom_catalog(
    names: list[str],
    catalog: list[str],
    *,
    drop_equal_to: str | None = None,
) -> list[str]:
    """已有标准名与当前病主症表求交，不再走向量对齐。"""
    allowed = {str(item).strip() for item in catalog if str(item).strip()}
    skip = (drop_equal_to or "").strip()
    matched: list[str] = []
    seen: set[str] = set()
    for name in names:
        text = str(name or "").strip()
        if not text or text == skip or text not in allowed or text in seen:
            continue
        seen.add(text)
        matched.append(text)
    return matched


def _split_matched_by_disease(
    matched: list[str],
    *,
    current: str,
    diseases_of: dict[str, list[str]],
) -> tuple[list[str], dict[str, list[str]]]:
    current_hits: list[str] = []
    other_by_disease: dict[str, list[str]] = {}
    for name in matched:
        owners = diseases_of.get(name) or []
        if current and current in owners:
            current_hits.append(name)
            continue
        for disease in owners:
            if disease == current:
                continue
            bucket = other_by_disease.setdefault(disease, [])
            if name not in bucket:
                bucket.append(name)
    return list(dict.fromkeys(current_hits)), other_by_disease


def match_mentions_to_global_main_symptoms(
    mentions: list[str],
    *,
    current_disease: str | None,
    already_aligned: list[str] | None = None,
) -> GlobalMainSymptomMatch:
    """已对齐标准名只查表；未对齐口语才走 Milvus。"""
    catalog, diseases_of = build_global_main_symptom_catalog()
    current = (current_disease or "").strip()
    reused = names_in_main_symptom_catalog(
        list(already_aligned or []),
        catalog,
        drop_equal_to=current,
    )
    reused_set = set(reused)
    pending: list[str] = []
    seen_pending: set[str] = set()
    for mention in mentions:
        text = str(mention or "").strip()
        if not text or text == current or text in reused_set or text in seen_pending:
            continue
        seen_pending.add(text)
        pending.append(text)
    extra: list[str] = []
    hits: list[AlignHit] = []
    if pending:
        extra, hits = align_symptoms_to_catalog(
            pending,
            catalog,
            drop_equal_to=current,
        )
    matched = list(dict.fromkeys([*reused, *extra]))
    current_hits, other_by_disease = _split_matched_by_disease(
        matched,
        current=current,
        diseases_of=diseases_of,
    )
    return GlobalMainSymptomMatch(
        current_hits=current_hits,
        other_by_disease=other_by_disease,
        alignments=hits,
    )


def classify_rollback_reply(text: str) -> RollbackReply:
    raw = (text or "").strip()
    if not raw:
        return "unclear"
    if raw in {"1", "１"}:
        return "yes"
    if raw in {"2", "２"}:
        return "no"
    # 「是的」表示确认当前病、继续第三轮，不是「同意换病」
    if raw in _STAY_AFFIRM:
        return "no"
    if any(token in raw for token in ("不用", "不要", "不必", "不换", "不用换", "否", "继续", "就这个", "先按")):
        return "no"
    if any(token in raw for token in _ROLLBACK_YES):
        return "yes"
    if any(token in raw for token in _ROLLBACK_NO):
        return "no"
    return "unclear"


def union_symptoms(syndromes: list[dict[str, Any]], *, field: str) -> list[str]:
    seen: set[str] = set()
    items: list[str] = []
    for syndrome in syndromes:
        for name in syndrome.get(field) or []:
            text = str(name).strip()
            if text and text not in seen:
                seen.add(text)
                items.append(text)
    return items


def build_auxiliary_symptom_catalog(
    syndromes: list[dict[str, Any]],
    *,
    exclude: set[str] | None = None,
) -> list[str]:
    """剩余候选证型在打分表中的辅症并集（内部对齐用）。"""
    from knowledge.neo4j_consult import list_dialectic_symptoms_for_syndromes

    skip = {str(item).strip() for item in (exclude or set()) if str(item).strip()}
    syndrome_ids = [int(item["id"]) for item in syndromes if item.get("id") is not None]
    symptoms = list_dialectic_symptoms_for_syndromes(syndrome_ids)
    return [item for item in symptoms if item not in skip]


def filter_syndromes_by_main_symptoms(
    syndromes: list[dict[str, Any]],
    selected: list[str],
) -> list[dict[str, Any]]:
    if not selected:
        return syndromes
    try:
        graph_hits = filter_syndromes_by_main_symptoms_graph(syndromes, selected)
    except Exception:
        graph_hits = []
    if graph_hits:
        return graph_hits
    picked = {str(item).strip() for item in selected if str(item).strip()}
    scored: list[tuple[int, dict[str, Any]]] = []
    for syndrome in syndromes:
        main_set = {str(name).strip() for name in (syndrome.get("main_symptoms") or []) if str(name).strip()}
        overlap = len(picked & main_set)
        if overlap:
            scored.append((overlap, syndrome))
    if not scored:
        return []
    best = max(score for score, _ in scored)
    return [item for score, item in scored if score == best]


def filter_syndromes_by_auxiliary_symptoms(
    syndromes: list[dict[str, Any]],
    selected: list[str],
) -> list[dict[str, Any]]:
    if not syndromes:
        return []
    if not selected:
        return syndromes
    return filter_syndromes_by_auxiliary_symptoms_graph(syndromes, selected)


def auxiliary_options(
    syndromes: list[dict[str, Any]],
    *,
    exclude: set[str] | None = None,
) -> list[str]:
    return build_auxiliary_symptom_catalog(syndromes, exclude=exclude)
