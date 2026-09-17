"""B2B 用户文本输入：复用主项目分发与相关性算法，结果对齐 Neo4j 问诊图。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from algorithm.input_parse import latest_human_text, parse_symptom_selections, resolve_symptom_catalog_picks
from algorithm.spell_correct import SpellCorrectResult, build_spell_correction_state_update, correct_user_text
from configs.paths import ensure_uvgraph_root
from knowledge.neo4j_consult import list_body_parts, list_diseases_for_part
from schema.b2b import OTHER_OPTION, SYMPTOM_OTHER_OPTION

__all__ = [
    "BodyTextResult",
    "DiseaseTextResult",
    "SymptomTextResult",
    "analyze_body_text",
    "analyze_disease_text",
    "correct_query_in_state",
    "dispatch_body_input",
    "dispatch_disease_input",
    "dispatch_auxiliary_symptom_input",
    "dispatch_main_symptom_input",
    "dispatch_symptom_input",
    "extract_query",
    "resolve_body_part_name",
]


def is_other_option(value: str | None) -> bool:
    return (value or "").strip() == OTHER_OPTION


_HANDLER_FROM_MAIN: dict[str, str] = {
    "option_agent": "option_input",
    "text_agent": "text_input",
    "vague_agent": "vague_input",
}


def _normalize_active_handler(payload: dict[str, Any]) -> dict[str, Any]:
    handler = payload.get("active_handler")
    if isinstance(handler, str) and handler in _HANDLER_FROM_MAIN:
        payload["active_handler"] = _HANDLER_FROM_MAIN[handler]
    return payload


def is_valid_body_selection(value: str | None, *, allowed: set[str] | None = None) -> bool:
    text = (value or "").strip()
    if not text or is_other_option(text):
        return False
    if allowed is None:
        return True
    return text in allowed


def is_valid_disease_selection(value: str | None, *, allowed: set[str] | None = None) -> bool:
    return is_valid_body_selection(value, allowed=allowed)


def extract_query(state: Mapping[str, Any]) -> str:
    query = state.get("query", "")
    if isinstance(query, str) and query.strip():
        return query.strip()
    return latest_human_text(state.get("messages"))


def correct_query_in_state(state: Mapping[str, Any]) -> tuple[dict[str, Any], SpellCorrectResult | None]:
    """thinking_relevance / 症状文字链前置：先纠错再业务推断。"""
    raw = extract_query(state)
    if not raw:
        return {}, None
    result = correct_user_text(raw)
    return build_spell_correction_state_update(result), result


def _state_with_spell_correction(state: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(state)
    correction_update, _ = correct_query_in_state(state)
    merged.update(correction_update)
    return merged


@dataclass(frozen=True)
class BodyTextResult:
    is_relevant: bool
    body_part: str | None
    body_part_id: int | None
    reason: str
    needs_clarification: bool = False
    disease_suggested: list[str] | None = None


@dataclass(frozen=True)
class DiseaseTextResult:
    is_relevant: bool
    disease_name: str | None
    disease_id: int | None
    reason: str
    needs_clarification: bool = False
    disease_suggested: list[str] | None = None


@dataclass(frozen=True)
class SymptomTextResult:
    selected_symptoms: list[str]
    needs_clarification: bool
    reason: str = ""
    slots: dict[str, Any] | None = None
    text_intent: str | None = None
    text_intent_reason: str | None = None


def _refill_empty_symptom_slots_with_qwen(
    text: str,
    *,
    slots: Any,
    major_round: int,
    body_part: Any,
    body_part_id: Any,
    catalog: list[str],
) -> tuple[Any, str | None]:
    """UIE 症状 span 为空时，用 qwen-turbo 抽原文片段再填槽。"""
    from algorithm.intent_slots import fill_consult_slots, slots_have_mentions
    from algorithm.qwen_consult_fallback import run_qwen_consult_fallback

    if slots_have_mentions(slots.to_state().get("slots"), ("症状",)):
        return slots, None
    fallback = run_qwen_consult_fallback(text, major_round=major_round)
    if fallback is None:
        return slots, "qwen-turbo 降级未返回（无 key / 调用失败 / 未启用）"
    if fallback.intent != "complaint" or not (fallback.entities.get("症状") or []):
        return slots, f"qwen-turbo 降级：{fallback.reason}"
    filled = fill_consult_slots(
        text,
        intent="complaint",
        major_round=major_round,
        body_part=body_part,
        body_part_id=body_part_id,
        symptom_catalog=catalog,
        seed_entities=fallback.entities,
        fill_reason=f"qwen-turbo 降级：{fallback.reason}",
    )
    return filled, f"qwen-turbo 降级：{fallback.reason}"


def resolve_body_part_name(candidate: str | None, parts: list[dict[str, Any]]) -> tuple[str | None, int | None]:
    if not candidate:
        return None, None
    text = candidate.strip()
    by_name = {row.get("part_name"): row for row in parts if row.get("part_name")}
    if text in by_name:
        row = by_name[text]
        return text, row.get("id")
    for name, row in by_name.items():
        if text in name or name in text:
            return name, row.get("id")
    return None, None


def _main_round1_algorithms():
    ensure_uvgraph_root()
    from src.algorithm.round1 import build_text_body_input, dispatch_body_input as main_dispatch
    from src.text import needs_clarification_for_text, normalize_user_text

    return main_dispatch, build_text_body_input, needs_clarification_for_text, normalize_user_text


def dispatch_body_input(
    *,
    selected_body_part: str | None,
    query: str,
    allowed_parts: set[str] | None = None,
) -> dict[str, Any]:
    """选项 / 文字 / 模糊 三分发（主项目 round1 算法）。"""
    main_dispatch, _, _, _ = _main_round1_algorithms()
    result = main_dispatch(selected_body_part=selected_body_part, query=query)
    if (
        result.get("active_handler") in ("option_agent", "option_input")
        and allowed_parts is not None
        and not is_valid_body_selection(result.get("selected_body_part"), allowed=allowed_parts)
    ):
        return {
            "query": query.strip(),
            "selected_body_part": None,
            "active_handler": "vague_input",
            "user_input_mode": "text",
            "needs_text_input": False,
            "needs_clarification": True,
        }
    return _normalize_active_handler(dict(result))


def dispatch_disease_input(
    *,
    selected_disease: str | None,
    query: str,
    allowed_diseases: set[str] | None = None,
) -> dict[str, Any]:
    """疾病轮：选项优先，否则走文字链。"""
    _, build_text_body_input, _, normalize_user_text = _main_round1_algorithms()
    if selected_disease and is_valid_disease_selection(selected_disease, allowed=allowed_diseases):
        return {
            "query": "",
            "selected_disease": selected_disease,
            "active_handler": "option_input",
            "user_input_mode": "option",
            "needs_text_input": False,
            "needs_clarification": False,
        }
    if is_other_option(selected_disease):
        return _normalize_active_handler(dict(build_text_body_input(query)))
    if normalize_user_text(query):
        payload = dict(build_text_body_input(query))
        payload["selected_disease"] = None
        return _normalize_active_handler(payload)
    return {
        "query": "",
        "selected_disease": None,
        "active_handler": "vague_input",
        "user_input_mode": "text",
        "needs_text_input": True,
        "needs_clarification": True,
    }


def dispatch_main_symptom_input(state: Mapping[str, Any]) -> SymptomTextResult:
    """第三轮主症：参考列表点选 / 文字描述 → UIE + Milvus 对齐标准主症名。"""
    from algorithm.intent_slots import fill_consult_slots

    _, _, needs_clarification_for_text, normalize_user_text = _main_round1_algorithms()
    catalog = [
        str(item).strip()
        for item in (state.get("symptom_options") or [])
        if str(item).strip() and str(item).strip() not in {SYMPTOM_OTHER_OPTION, OTHER_OPTION}
    ]
    allowed = set(catalog)

    selected = parse_symptom_selections(state)
    picked = resolve_symptom_catalog_picks(selected, catalog)
    if picked:
        return SymptomTextResult(selected_symptoms=picked, needs_clarification=False)

    query = extract_query(state)
    text = normalize_user_text(query)
    if not text and selected:
        text = "、".join(selected)
    if not text:
        return SymptomTextResult(
            selected_symptoms=[],
            needs_clarification=True,
            reason="请描述主要不适症状",
        )
    query_picks = resolve_symptom_catalog_picks(
        [part.strip() for part in text.replace("、", "|").replace("，", "|").replace(",", "|").split("|") if part.strip()],
        catalog,
    )
    if query_picks:
        return SymptomTextResult(selected_symptoms=query_picks, needs_clarification=False)
    if needs_clarification_for_text(text):
        return SymptomTextResult(
            selected_symptoms=[],
            needs_clarification=True,
            reason="症状描述过于模糊",
        )

    slots, qwen_reason = _refill_empty_symptom_slots_with_qwen(
        text,
        slots=fill_consult_slots(
            text,
            intent="complaint",
            major_round=3,
            body_part=state.get("body_part"),
            body_part_id=state.get("body_part_id"),
            symptom_catalog=catalog,
        ),
        major_round=3,
        body_part=state.get("body_part"),
        body_part_id=state.get("body_part_id"),
        catalog=catalog,
    )
    slot_state = slots.to_state().get("slots")
    matched = [
        str(name).strip()
        for name in slots.symptoms
        if str(name).strip() and (not allowed or str(name).strip() in allowed)
    ]
    if not matched and slots.alignments:
        for row in slots.alignments:
            name = str((row or {}).get("standard_name") or "").strip()
            if not name or (allowed and name not in allowed):
                continue
            if (row or {}).get("needs_review"):
                continue
            matched.append(name)
    matched = list(dict.fromkeys(matched))
    if not matched:
        return SymptomTextResult(
            selected_symptoms=[],
            needs_clarification=True,
            reason="未能从描述中识别出可匹配的主症，请换种说法或参考常见主症",
            slots=slot_state,
            text_intent="complaint",
            text_intent_reason=qwen_reason,
        )
    return SymptomTextResult(
        selected_symptoms=matched,
        needs_clarification=False,
        slots=slot_state,
        text_intent="complaint",
        text_intent_reason=qwen_reason,
    )


def _auxiliary_symptom_catalog(state: Mapping[str, Any]) -> list[str]:
    from algorithm.syndrome_match import build_auxiliary_symptom_catalog

    catalog = [
        str(item).strip()
        for item in (state.get("auxiliary_symptom_catalog") or [])
        if str(item).strip()
    ]
    if catalog:
        return catalog
    syndromes = list(state.get("candidate_syndromes") or [])
    exclude = {str(item).strip() for item in (state.get("confirmed_main_symptoms") or []) if str(item).strip()}
    return build_auxiliary_symptom_catalog(syndromes, exclude=exclude)


def dispatch_auxiliary_symptom_input(state: Mapping[str, Any]) -> SymptomTextResult:
    """第四轮辅症：自由描述 → UIE + Milvus 对齐剩余证型打分表辅症。"""
    from algorithm.intent_slots import fill_consult_slots

    _, _, needs_clarification_for_text, normalize_user_text = _main_round1_algorithms()
    catalog = _auxiliary_symptom_catalog(state)
    allowed = set(catalog)

    selected = parse_symptom_selections(state)
    picked = resolve_symptom_catalog_picks(selected, catalog)
    if picked:
        return SymptomTextResult(selected_symptoms=picked, needs_clarification=False)

    query = extract_query(state)
    text = normalize_user_text(query)
    if not text and selected:
        text = "、".join(selected)
    if not text:
        return SymptomTextResult(
            selected_symptoms=[],
            needs_clarification=True,
            reason="请补充其他不适表现",
        )
    query_picks = resolve_symptom_catalog_picks(
        [part.strip() for part in text.replace("、", "|").replace("，", "|").replace(",", "|").split("|") if part.strip()],
        catalog,
    )
    if query_picks:
        return SymptomTextResult(selected_symptoms=query_picks, needs_clarification=False)
    if needs_clarification_for_text(text):
        return SymptomTextResult(
            selected_symptoms=[],
            needs_clarification=True,
            reason="症状描述过于模糊",
        )

    slots, qwen_reason = _refill_empty_symptom_slots_with_qwen(
        text,
        slots=fill_consult_slots(
            text,
            intent="complaint",
            major_round=4,
            body_part=state.get("body_part"),
            body_part_id=state.get("body_part_id"),
            symptom_catalog=catalog,
        ),
        major_round=4,
        body_part=state.get("body_part"),
        body_part_id=state.get("body_part_id"),
        catalog=catalog,
    )
    symptoms = [
        str(name).strip()
        for name in slots.symptoms
        if str(name).strip() and (not allowed or str(name).strip() in allowed)
    ]
    if not symptoms and slots.alignments:
        for row in slots.alignments:
            name = str((row or {}).get("standard_name") or "").strip()
            if not name or (allowed and name not in allowed):
                continue
            if (row or {}).get("needs_review"):
                continue
            symptoms.append(name)
    symptoms = list(dict.fromkeys(symptoms))
    if not symptoms:
        return SymptomTextResult(
            selected_symptoms=[],
            needs_clarification=True,
            reason="未能从描述中识别出可匹配的辅助症状，请换种说法补充",
            slots=slots.to_state().get("slots"),
            text_intent="complaint",
            text_intent_reason=qwen_reason,
        )
    return SymptomTextResult(
        selected_symptoms=symptoms,
        needs_clarification=False,
        slots=slots.to_state().get("slots"),
        text_intent="complaint",
        text_intent_reason=qwen_reason,
    )


def dispatch_symptom_input(state: Mapping[str, Any]) -> SymptomTextResult:
    """症状轮：多选优先，否则解析用户文字（先纠错）。"""
    _, _, needs_clarification_for_text, normalize_user_text = _main_round1_algorithms()
    selected = parse_symptom_selections(state)
    if selected:
        return SymptomTextResult(selected_symptoms=selected, needs_clarification=False)

    query = extract_query(state)
    text = normalize_user_text(query)
    if not text:
        return SymptomTextResult(
            selected_symptoms=[],
            needs_clarification=True,
            reason="未选择症状",
        )
    if needs_clarification_for_text(text):
        return SymptomTextResult(
            selected_symptoms=[],
            needs_clarification=True,
            reason="症状描述过于模糊",
        )
    return SymptomTextResult(
        selected_symptoms=parse_symptom_selections({"query": text}),
        needs_clarification=False,
    )


def analyze_body_text(state: Mapping[str, Any]) -> tuple[BodyTextResult, dict[str, Any]]:
    """文字描述 → 先纠错 → 相关性判定 + 部位映射到 Neo4j。"""
    ensure_uvgraph_root()
    from src.algorithm.relevance import check_relevance_from_state

    correction_update, _ = correct_query_in_state(state)
    working_state = {**dict(state), **correction_update}

    parts = list_body_parts()
    allowed = {row["part_name"] for row in parts if row.get("part_name")}
    result = check_relevance_from_state(working_state)

    if not result.is_relevant:
        return (
            BodyTextResult(
                is_relevant=False,
                body_part=None,
                body_part_id=None,
                reason=result.reason or "未识别为身体不适描述",
            ),
            correction_update,
        )

    body_name, body_id = resolve_body_part_name(result.body_part, parts)
    if not body_name:
        return (
            BodyTextResult(
                is_relevant=True,
                body_part=None,
                body_part_id=None,
                reason="已识别为问诊内容，但未能匹配到可选部位",
                needs_clarification=True,
                disease_suggested=list(result.disease_suggested or []),
            ),
            correction_update,
        )

    if allowed and body_name not in allowed:
        return (
            BodyTextResult(
                is_relevant=True,
                body_part=None,
                body_part_id=None,
                reason="推断部位不在当前机构可选列表中",
                needs_clarification=True,
            ),
            correction_update,
        )

    return (
        BodyTextResult(
            is_relevant=True,
            body_part=body_name,
            body_part_id=body_id,
            reason=result.reason or "已识别身体不适描述",
            disease_suggested=list(result.disease_suggested or []),
        ),
        correction_update,
    )


def analyze_disease_text(
    state: Mapping[str, Any],
    *,
    body_part: str | None,
    body_part_id: int | None,
) -> tuple[DiseaseTextResult, dict[str, Any]]:
    """文字描述 → 先纠错 → Milvus 疾病匹配。"""
    ensure_uvgraph_root()
    from src.algorithm.match_disease import match_diseases
    from src.text import needs_clarification_for_text, normalize_user_text

    correction_update, _ = correct_query_in_state(state)
    working_state = {**dict(state), **correction_update}

    diseases = list_diseases_for_part(part_id=body_part_id, part_name=body_part)
    allowed = {row["disease_name"] for row in diseases if row.get("disease_name")}
    by_name = {row["disease_name"]: row for row in diseases if row.get("disease_name")}

    query = extract_query(working_state)
    text = normalize_user_text(query)
    if not text:
        return (
            DiseaseTextResult(
                is_relevant=False,
                disease_name=None,
                disease_id=None,
                reason="输入为空",
                needs_clarification=True,
            ),
            correction_update,
        )
    if needs_clarification_for_text(text):
        return (
            DiseaseTextResult(
                is_relevant=False,
                disease_name=None,
                disease_id=None,
                reason="描述过于模糊",
                needs_clarification=True,
            ),
            correction_update,
        )

    matched_name: str | None = None
    for name in allowed:
        if name in text:
            matched_name = name
            break

    suggested: list[str] = []
    slots = state.get("slots") if isinstance(state.get("slots"), dict) else {}
    from algorithm.intent_slots import slots_have_mentions

    # 没有疾病 span 时不拿整句打 Milvus
    if not matched_name and slots_have_mentions(slots, ("疾病",)):
        match_result = match_diseases(text, existing_body_part=body_part)
        suggested = [item.disease_name for item in match_result.candidates if item.disease_name in allowed]
        matched_name = match_result.top_disease if match_result.top_disease in allowed else None
        if not matched_name and suggested:
            matched_name = suggested[0]

    if not matched_name:
        return (
            DiseaseTextResult(
                is_relevant=True,
                disease_name=None,
                disease_id=None,
                reason="未能匹配到当前部位下的疾病",
                needs_clarification=True,
                disease_suggested=suggested[:3],
            ),
            correction_update,
        )

    row = by_name.get(matched_name)
    return (
        DiseaseTextResult(
            is_relevant=True,
            disease_name=matched_name,
            disease_id=row.get("id") if row else None,
            reason="已匹配疾病",
            disease_suggested=suggested[:3] or [matched_name],
        ),
        correction_update,
    )
