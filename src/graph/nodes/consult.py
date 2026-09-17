"""B2B 四轮辨证节点。"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from algorithm.syndrome_match import (
    classify_rollback_reply,
    filter_syndromes_by_auxiliary_symptoms,
    filter_syndromes_by_main_symptoms,
    names_in_main_symptom_catalog,
    ok_aligned_symptom_names,
    union_symptoms,
    build_global_main_symptom_catalog,
)
from algorithm import (
    build_auxiliary_symptom_catalog,
    is_valid_single_choice,
    parse_symptom_selections,
)
from algorithm.input_parse import (
    is_localized_body_part,
    is_systemic_unlocalized_text,
    match_name_in_catalog,
    resolve_symptom_catalog_picks,
)
from schema.entity_synonyms import lookup_synonym
from algorithm.disease_explain import compose_disease_plain_list
from algorithm.reply_agent import compose_user_reply
from algorithm.intent_classify_bert import classify_sentence_intent
from algorithm.intent_slots import fill_consult_slots, slots_have_mentions
from algorithm.qwen_consult_fallback import run_qwen_consult_fallback
from algorithm.text_input import (
    analyze_body_text,
    analyze_disease_text,
    correct_query_in_state,
    dispatch_body_input,
    dispatch_disease_input,
    dispatch_auxiliary_symptom_input,
    dispatch_main_symptom_input,
    dispatch_symptom_input,
    extract_query,
)
from knowledge.neo4j_consult import (
    get_body_part_by_id,
    get_disease_by_id,
    get_disease_by_name,
    list_body_parts,
    list_diseases_for_part,
    list_selectable_disease_names,
    list_syndromes_for_disease,
)
from schema.b2b import MAX_AUXILIARY_ROUNDS, OTHER_OPTION

from ..state import B2BRouterState

_ROUND_MARKERS = {
    1: "【第一轮】",
    2: "【第二轮】",
    3: "【第三轮】",
    4: "【第四轮】",
}


def _round_marker(state: B2BRouterState | dict, *, round_no: int | None = None) -> str:
    n = int(round_no if round_no is not None else (state.get("major_round") or 1))
    if n not in _ROUND_MARKERS:
        n = 1 if n < 2 else 4
    return _ROUND_MARKERS[n]


def _ensure_round_marker(text: str, state: B2BRouterState | dict, *, round_no: int | None = None) -> str:
    """用户可见回复必须以【第N轮】开头，标明当前大轮。"""
    body = (text or "").strip()
    if not body:
        return body
    locked = state.get("body_part") or state.get("selected_body_part")
    n = round_no
    if n is None:
        n = int(state.get("major_round") or 1)
    # 标准部位未锁定时，话术即使还在问部位也不能标第二轮
    if not is_localized_body_part(locked):
        n = 1
    marker = _round_marker(state, round_no=n)
    for item in _ROUND_MARKERS.values():
        if body.startswith(item):
            if body.startswith(marker):
                return body
            return marker + body[len(item) :].lstrip()
    if body.startswith("【第四轮"):
        rest = body.split("】", 1)
        tail = rest[1].lstrip() if len(rest) > 1 else body
        return marker + tail
    return f"{marker}{body}"


CONSULT_ENDED_LINE = "本次问诊系统已结束，如需再次辨证请新开一场会话。"


def _with_consult_ended(text: str) -> str:
    body = (text or "").rstrip()
    if CONSULT_ENDED_LINE in body:
        return body
    return f"{body}\n\n{CONSULT_ENDED_LINE}"


def _merge_collected_symptoms(state: B2BRouterState | dict, extra: list[str] | None) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for item in list(state.get("collected_symptoms") or []) + list(extra or []):
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        merged.append(text)
    return merged


def _symptom_mentions_from_state(state: B2BRouterState | dict) -> list[str]:
    mentions: list[str] = []
    slots = state.get("slots") if isinstance(state.get("slots"), dict) else {}
    entities = slots.get("entities") if isinstance(slots.get("entities"), dict) else {}
    for item in (
        list(state.get("collected_symptoms") or [])
        + list(slots.get("symptoms") or [])
        + list(entities.get("症状") or [])
    ):
        text = str(item).strip()
        if text:
            mentions.append(text)
    return list(dict.fromkeys(mentions))


def _align_hits_to_dicts(hits: list) -> list[dict]:
    rows: list[dict] = []
    for hit in hits:
        rows.append(
            {
                "mention": hit.mention,
                "standard_name": hit.standard_name,
                "node_id": hit.node_id,
                "node_label": hit.node_label,
                "method": hit.method,
                "score": hit.score,
                "needs_review": hit.needs_review,
            }
        )
    return rows


def _align_collected_main_symptoms(state: B2BRouterState | dict, syndromes: list) -> tuple[list[str], list[str], list[dict]]:
    catalog = union_symptoms(syndromes, field="main_symptoms")
    disease = str(state.get("disease_name") or state.get("selected_disease") or "")
    slots = state.get("slots") if isinstance(state.get("slots"), dict) else {}
    names = ok_aligned_symptom_names(slots.get("alignments") if isinstance(slots, dict) else None)
    names.extend(_symptom_mentions_from_state(state))
    matched = names_in_main_symptom_catalog(names, catalog, drop_equal_to=disease)
    return catalog, matched, []


def _format_options(title: str, options: list[str]) -> str:
    lines = [title, ""]
    for idx, item in enumerate(options, start=1):
        lines.append(f"{idx}. {item}")
    return "\n".join(lines)


def _user_reply(state: B2BRouterState, *, task: str, fallback: str) -> str:
    """意图/抽槽之后的人话：单 Agent 组句，失败用规则稿。"""
    return _ensure_round_marker(
        compose_user_reply(state, task=task, fallback=fallback),
        state,
    )


def _round2_disease_names(state: B2BRouterState) -> list[str]:
    options = [str(item).strip() for item in (state.get("disease_options") or []) if str(item).strip()]
    if options:
        return options
    diseases = list_diseases_for_part(
        part_id=state.get("body_part_id"),
        part_name=state.get("body_part"),
    )
    return [row["disease_name"] for row in diseases if row.get("disease_name")]


def _round2_unfilled_reply(state: B2BRouterState, *, fallback: str, task: str) -> str:
    """部位已定、疾病轮已问过且槽仍空：主诉或 vague 都展示该部位疾病白话列表。寒暄不走这条。"""
    intent = state.get("text_intent")
    if (
        intent != "chitchat"
        and state.get("disease_asked")
        and state.get("body_part")
        and not (state.get("disease_name") or state.get("selected_disease"))
    ):
        return _ensure_round_marker(
            compose_disease_plain_list(
                body_part=state.get("body_part"),
                body_part_id=state.get("body_part_id"),
                disease_names=_round2_disease_names(state),
            ),
            state,
            round_no=2,
        )
    return _user_reply(state, task=task, fallback=fallback)


def opening_round1(state: B2BRouterState) -> B2BRouterState:
    parts = list_body_parts()
    options = [row["part_name"] for row in parts if row.get("part_name")]
    return {
        "major_round": 1,
        "session_mode": "round1_body",
        "body_part_options": options,
        "next_action": "await_user",
        "messages": [
            AIMessage(
                content=_format_options(
                    "【第一轮】请选择不适部位，或直接描述症状：",
                    options,
                )
            )
        ],
    }


def parse_round1_input(state: B2BRouterState) -> B2BRouterState:
    """解析第一轮输入：选项 / 文字 / 模糊（主项目 dispatch 算法）。"""
    parts = list_body_parts()
    allowed = {row["part_name"] for row in parts if row.get("part_name")}
    payload = dispatch_body_input(
        selected_body_part=state.get("selected_body_part"),
        query=extract_query(state),
        allowed_parts=allowed,
    )
    return {
        **payload,
        "needs_clarification": bool(payload.get("needs_clarification")),
    }


def _classify_text_intent(state: B2BRouterState) -> B2BRouterState:
    """纠错后：标准名 → BERT/UIE；非 complaint 或抽空则 qwen-turbo 为准。"""
    correction_update, _ = correct_query_in_state(state)
    working = {**dict(state), **correction_update}
    query = extract_query(working)
    payload: B2BRouterState = dict(correction_update)
    round_no = int(state.get("major_round") or 1)
    catalog_hit = _match_standard_name_after_correct(working, query)
    if catalog_hit:
        payload.update(catalog_hit)

    if not catalog_hit:
        result = classify_sentence_intent(query)
        payload.update(
            {
                "text_intent": result.intent,
                "text_intent_confidence": result.confidence,
                "text_intent_reason": result.reason,
            }
        )
        intent_source = result.source
    else:
        intent_source = "rule"

    payload["text_intent_source"] = intent_source

    slots = None
    if catalog_hit or payload.get("text_intent") == "complaint":
        slots = fill_consult_slots(
            query,
            intent="complaint",
            major_round=round_no,
            body_part=payload.get("body_part") or state.get("body_part"),
            body_part_id=payload.get("body_part_id") or state.get("body_part_id"),
            symptom_catalog=list(state.get("symptom_options") or []),
        )

    need_qwen = (not catalog_hit) and (
        payload.get("text_intent") != "complaint"
        or slots is None
        or not slots_have_mentions(slots.to_state().get("slots"))
    )
    if need_qwen:
        fallback = run_qwen_consult_fallback(query, major_round=round_no)
        if fallback is not None:
            payload["text_intent"] = fallback.intent
            payload["text_intent_confidence"] = 0.85
            payload["text_intent_reason"] = f"qwen-turbo 降级：{fallback.reason}"
            intent_source = "qwen-turbo"
            payload["text_intent_source"] = intent_source
            if fallback.intent == "complaint":
                slots = fill_consult_slots(
                    query,
                    intent="complaint",
                    major_round=round_no,
                    body_part=payload.get("body_part") or state.get("body_part"),
                    body_part_id=payload.get("body_part_id") or state.get("body_part_id"),
                    symptom_catalog=list(state.get("symptom_options") or []),
                    seed_entities=fallback.entities,
                    fill_reason=f"qwen-turbo 降级：{fallback.reason}",
                )

    if payload.get("text_intent") != "complaint":
        return payload
    if slots is None:
        return payload
    payload.update(slots.to_state())
    if catalog_hit:
        if catalog_hit.get("body_part"):
            payload["body_part"] = catalog_hit.get("body_part")
            payload["selected_body_part"] = catalog_hit.get("selected_body_part") or catalog_hit.get("body_part")
            payload["body_part_id"] = catalog_hit.get("body_part_id")
        if round_no >= 2 and catalog_hit.get("disease_name"):
            payload["disease_name"] = catalog_hit.get("disease_name")
            payload["selected_disease"] = catalog_hit.get("selected_disease") or catalog_hit.get("disease_name")
            payload["disease_id"] = catalog_hit.get("disease_id")
    else:
        if slots.body_part:
            payload["body_part"] = slots.body_part
            payload["selected_body_part"] = slots.body_part
            payload["body_part_id"] = slots.body_part_id
        if round_no >= 2 and slots.disease_name:
            payload["disease_name"] = slots.disease_name
            payload["selected_disease"] = slots.disease_name
            payload["disease_id"] = slots.disease_id
    if round_no <= 1:
        payload["disease_name"] = None
        payload["selected_disease"] = None
        payload["disease_id"] = None
        slot_state = payload.get("slots")
        if isinstance(slot_state, dict):
            slot_state["disease_name"] = None
            slot_state["disease_id"] = None
            nodes = dict(slot_state.get("entry_nodes") or {})
            nodes.pop("disease", None)
            slot_state["entry_nodes"] = nodes
            slot_state["entities"] = {
                key: value for key, value in (slot_state.get("entities") or {}).items() if key != "疾病"
            }
            slot_state["mentions"] = {
                key: value for key, value in (slot_state.get("mentions") or {}).items() if key != "疾病"
            }
        guessed = payload.get("body_part") or payload.get("selected_body_part")
        if (not is_localized_body_part(guessed)) or is_systemic_unlocalized_text(query):
            payload["body_part"] = None
            payload["selected_body_part"] = None
            payload["body_part_id"] = None
            if isinstance(slot_state, dict):
                slot_state["body_part"] = None
                slot_state["body_part_id"] = None
                nodes = dict(slot_state.get("entry_nodes") or {})
                nodes.pop("body_part", None)
                slot_state["entry_nodes"] = nodes
    extra = list(slots.symptoms or [])
    extra.extend(str(item).strip() for item in (slots.entities.get("症状") or []) if str(item).strip())
    payload["collected_symptoms"] = _merge_collected_symptoms(working, extra)
    if (
        not catalog_hit
        and intent_source not in ("rule", "qwen-turbo")
        and not slots.disease_name
        and not slots_have_mentions(slots.to_state().get("slots"))
    ):
        payload["text_intent"] = "vague"
        payload["text_intent_reason"] = f"{payload.get('text_intent_reason') or ''}；无问诊实体，改 vague"
    return payload


def _match_standard_name_after_correct(state: B2BRouterState, query: str) -> B2BRouterState:
    """纠错后的句子直接对部位/疾病标准名，命中则不再走 BERT。"""
    if not (query or "").strip():
        return {}
    round_no = int(state.get("major_round") or 1)
    alias = lookup_synonym("disease", query)
    if round_no >= 2 and (round_no == 2 or alias):
        allowed = {str(item).strip() for item in (state.get("disease_options") or []) if str(item).strip()}
        named = match_name_in_catalog(query, allowed)
        if not named:
            try:
                named = match_name_in_catalog(query, set(list_selectable_disease_names()))
            except Exception:
                named = None
        if not named and alias:
            named = alias
        if named:
            row = get_disease_by_name(named) or {}
            hit: B2BRouterState = {
                "text_intent": "complaint",
                "text_intent_reason": "纠错后命中标准病名或口语别名",
                "selected_disease": named,
                "disease_name": named,
                "disease_id": row.get("id"),
                "needs_clarification": False,
            }
            if row.get("body_part"):
                hit["body_part"] = row.get("body_part")
                hit["selected_body_part"] = row.get("body_part")
                hit["body_part_id"] = row.get("body_part_id")
            return hit
    if round_no == 1:
        allowed = {str(item).strip() for item in (state.get("body_part_options") or []) if str(item).strip()}
        if not allowed:
            try:
                allowed = {row["part_name"] for row in list_body_parts() if row.get("part_name")}
            except Exception:
                allowed = set()
        named = match_name_in_catalog(query, allowed)
        if not named:
            return {}
        parts = {row.get("part_name"): row for row in list_body_parts() if row.get("part_name")}
        row = parts.get(named) or {}
        return {
            "text_intent": "complaint",
            "text_intent_reason": "纠错后命中标准部位",
            "selected_body_part": named,
            "body_part": named,
            "body_part_id": row.get("id"),
            "needs_clarification": False,
        }
    return {}


def classify_text_intent_round1(state: B2BRouterState) -> B2BRouterState:
    return _classify_text_intent(state)


def classify_text_intent_round2(state: B2BRouterState) -> B2BRouterState:
    return _classify_text_intent(state)


def _respond_text_intent_message(state: B2BRouterState, *, round_no: int) -> str:
    intent = state.get("text_intent") or "vague"
    slots = state.get("slots") or {}
    missing = "、".join(slots.get("missing") or [])
    reason = (state.get("relevance_reason") or state.get("text_intent_reason") or "").strip()
    hint = f"（{reason}）" if reason else ""

    if round_no == 1:
        if intent == "chitchat":
            return f"我们先聚焦一下您的身体状况，请说说最近哪里不舒服，或选择对应部位。{hint}"
        if intent == "complaint":
            need = f"还缺：{missing}。" if missing else ""
            return f"未能从描述中确定部位。{need}请从部位列表中选择，或补充更具体的症状。{hint}"
        return f"请再具体说说，最近哪里不太舒服？也可以直接从部位列表中选择。{hint}"

    if intent == "chitchat":
        return f"请结合已选部位，说说具体不适或从疾病列表中选择。{hint}"
    if intent == "complaint":
        need = f"还缺：{missing}。" if missing else ""
        return f"未能匹配到疾病。{need}请从疾病列表中选择，或补充更具体的症状描述。{hint}"
    return f"请再具体描述一下不适表现，或从疾病列表中选择一项。{hint}"


def respond_text_intent_round1(state: B2BRouterState) -> B2BRouterState:
    fallback = _respond_text_intent_message(state, round_no=1)
    reply = _user_reply(
        {**dict(state), "major_round": 1},
        task="第一轮：根据意图和槽位回复。寒暄则拉回问诊；主诉但部位未齐则追问或请点选部位列表。全身发冷等未锁定部位时仍标【第一轮】，继续问部位。",
        fallback=fallback,
    )
    return {
        "needs_clarification": True,
        "next_action": "await_user",
        "major_round": 1,
        "session_mode": "round1_body",
        "messages": [AIMessage(content=reply)],
    }


def respond_text_intent_round2(state: B2BRouterState) -> B2BRouterState:
    fallback = _respond_text_intent_message(state, round_no=2)
    reply = _round2_unfilled_reply(
        state,
        fallback=fallback,
        task=(
            "第二轮：根据意图、实体和疾病候选项回复。"
            "若抽到的是症状而疾病槽为空，承认症状并请用户从疾病列表中选择最贴近的一项，或补充更具体描述；不要把症状说成已确诊病名。"
        ),
    )
    return {
        "needs_clarification": True,
        "next_action": "await_user",
        "messages": [AIMessage(content=reply)],
    }


def thinking_relevance_round1(state: B2BRouterState) -> B2BRouterState:
    """complaint：槽位已对齐部位则直接用，否则再推断。"""
    query = extract_query(state)
    slots = state.get("slots") or {}
    selected = state.get("body_part") or state.get("selected_body_part")
    if is_systemic_unlocalized_text(query) or not is_localized_body_part(selected):
        selected = None
    if selected or (slots.get("filled") and is_localized_body_part(slots.get("body_part")) and not is_systemic_unlocalized_text(query)):
        name = selected or slots.get("body_part")
        if is_localized_body_part(name) and not is_systemic_unlocalized_text(query):
            return {
                "is_relevant": True,
                "needs_clarification": False,
                "body_part": name,
                "selected_body_part": name,
                "body_part_id": state.get("body_part_id") or slots.get("body_part_id"),
                "relevance_reason": "纠错后命中标准部位" if selected else "槽位已对齐部位",
                "major_round": 2,
                "session_mode": "round2_disease",
                "query": "",
            }
    from algorithm.intent_slots import slots_have_mentions

    if not slots_have_mentions(slots, ("部位", "症状")):
        return {
            "needs_clarification": True,
            "relevance_reason": "未抽到部位或症状实体，跳过向量检索",
            "next_action": "await_user",
        }
    result, correction_update = analyze_body_text(state)
    base: B2BRouterState = dict(correction_update)
    if result.body_part and is_localized_body_part(result.body_part) and not is_systemic_unlocalized_text(query):
        return {
            **base,
            "is_relevant": True,
            "needs_clarification": False,
            "body_part": result.body_part,
            "selected_body_part": result.body_part,
            "body_part_id": result.body_part_id,
            "relevance_reason": result.reason,
            "disease_suggested": list(result.disease_suggested or []),
            "major_round": 2,
            "session_mode": "round2_disease",
            "query": "",
        }
    payload: B2BRouterState = {
        **base,
        "needs_clarification": True,
        "relevance_reason": result.reason if not result.body_part else "全身性不适，部位未锁定",
        "next_action": "await_user",
        "major_round": 1,
        "session_mode": "round1_body",
        "body_part": None,
        "selected_body_part": None,
    }
    if result.disease_suggested:
        payload["disease_suggested"] = list(result.disease_suggested)
    return payload


def apply_body_part(state: B2BRouterState) -> B2BRouterState:
    selected = state.get("selected_body_part") or state.get("body_part")
    if not is_valid_single_choice(selected) or not is_localized_body_part(selected):
        return {"needs_clarification": True, "major_round": 1, "session_mode": "round1_body"}
    parts = list_body_parts()
    part_row = next((row for row in parts if row.get("part_name") == selected), None)
    return {
        "needs_clarification": False,
        "is_relevant": None,
        "active_route": None,
        "body_part": selected,
        "selected_body_part": selected,
        "body_part_id": part_row.get("id") if part_row else None,
        "major_round": 2,
        "session_mode": "round2_disease",
    }


def clarify_round1(state: B2BRouterState) -> B2BRouterState:
    fallback = f"请从部位列表中选择一项，或输入具体部位名称（不要选「{OTHER_OPTION}」）。"
    reply = _user_reply(
        state,
        task="第一轮澄清：请用户从部位列表点选或说更具体的部位/症状。",
        fallback=fallback,
    )
    return {
        "next_action": "await_user",
        "messages": [AIMessage(content=reply)],
    }


def parse_round2_input(state: B2BRouterState) -> B2BRouterState:
    """解析第二轮输入：选项 / 文字 / 模糊。"""
    diseases = list_diseases_for_part(
        part_id=state.get("body_part_id"),
        part_name=state.get("body_part"),
    )
    allowed = {row["disease_name"] for row in diseases if row.get("disease_name")}
    payload = dispatch_disease_input(
        selected_disease=state.get("selected_disease"),
        query=extract_query(state),
        allowed_diseases=allowed,
    )
    return {
        **payload,
        "needs_clarification": bool(payload.get("needs_clarification")),
    }


def thinking_relevance_round2(state: B2BRouterState) -> B2BRouterState:
    """complaint：槽位已对齐疾病则直接用，否则再匹配。"""
    slots = state.get("slots") or {}
    selected = state.get("disease_name") or state.get("selected_disease")
    if selected or (slots.get("filled") and slots.get("disease_name")):
        name = selected or slots.get("disease_name")
        return {
            "is_relevant": True,
            "needs_clarification": False,
            "selected_disease": name,
            "disease_name": name,
            "disease_id": state.get("disease_id") or slots.get("disease_id"),
            "relevance_reason": "纠错后命中标准病名" if selected else "槽位已对齐疾病",
            "query": "",
        }
    from algorithm.intent_slots import slots_have_mentions

    if not slots_have_mentions(slots, ("疾病",)):
        return {
            "needs_clarification": True,
            "relevance_reason": "未抽到疾病实体，跳过向量检索",
            "next_action": "await_user",
        }
    result, correction_update = analyze_disease_text(
        state,
        body_part=state.get("body_part"),
        body_part_id=state.get("body_part_id"),
    )
    base: B2BRouterState = dict(correction_update)
    if result.disease_name:
        return {
            **base,
            "is_relevant": True,
            "needs_clarification": False,
            "selected_disease": result.disease_name,
            "disease_name": result.disease_name,
            "disease_id": result.disease_id,
            "relevance_reason": result.reason,
            "disease_suggested": list(result.disease_suggested or []),
            "query": "",
        }
    payload: B2BRouterState = {
        **base,
        "needs_clarification": True,
        "relevance_reason": result.reason,
        "next_action": "await_user",
    }
    if result.disease_suggested:
        payload["disease_suggested"] = list(result.disease_suggested)
        payload["disease_options"] = list(result.disease_suggested)
    return payload


def offer_disease_selection(state: B2BRouterState) -> B2BRouterState:
    diseases = list_diseases_for_part(
        part_id=state.get("body_part_id"),
        part_name=state.get("body_part"),
    )
    options = [row["disease_name"] for row in diseases if row.get("disease_name")]
    suggested = list(state.get("disease_suggested") or [])
    if suggested:
        options = list(dict.fromkeys([*suggested, *options]))
    fallback = _format_options(
        f"【第二轮】已选部位：{state.get('body_part')}。请选择疾病：",
        options,
    )
    reply = _user_reply(
        {
            **dict(state),
            "disease_options": options,
            "major_round": 2,
            "symptom_options": [],
            "disease_name": state.get("disease_name"),
        },
        task="第二轮展示疾病候选项，邀请点选或继续用文字描述；编号列表必须是疾病名，不要列出主症或伴随症状。",
        fallback=fallback,
    )
    return {
        "disease_options": options,
        "disease_asked": True,
        "next_action": "await_user",
        "messages": [AIMessage(content=reply)],
    }


def apply_disease(state: B2BRouterState) -> B2BRouterState:
    selected = state.get("selected_disease") or state.get("disease_name")
    if not is_valid_single_choice(selected):
        return {"needs_clarification": True}
    diseases = list_diseases_for_part(
        part_id=state.get("body_part_id"),
        part_name=state.get("body_part"),
    )
    disease_row = next((row for row in diseases if row.get("disease_name") == selected), None)
    if disease_row is None:
        disease_row = get_disease_by_name(selected)
    syndromes = list_syndromes_for_disease(
        disease_id=disease_row.get("id") if disease_row else None,
        disease_name=selected,
    )
    payload: B2BRouterState = {
        "needs_clarification": False,
        "is_relevant": None,
        "active_route": None,
        "disease_name": selected,
        "selected_disease": selected,
        "disease_id": disease_row.get("id") if disease_row else None,
        "candidate_syndromes": syndromes,
        "major_round": 3,
        "session_mode": "round3_main_symptom",
        "selected_symptoms": [],
        "symptom_options": union_symptoms(syndromes, field="main_symptoms"),
        "rollback_decision": None,
    }
    if disease_row and disease_row.get("body_part") and not state.get("body_part"):
        payload["body_part"] = disease_row.get("body_part")
        payload["body_part_id"] = disease_row.get("body_part_id")
    return payload


def retrieve_collected_main_symptoms(state: B2BRouterState) -> B2BRouterState:
    """前两轮已对齐症状名 ∩ 当前病主症表；不再向量对齐。"""
    empty: B2BRouterState = {
        "selected_symptoms": [],
        "pending_current_main_hits": [],
        "rollback_candidate_diseases": [],
        "rollback_decision": None,
    }
    syndromes = state.get("candidate_syndromes") or []
    disease = str(state.get("disease_name") or state.get("selected_disease") or "").strip()
    current_catalog = union_symptoms(syndromes, field="main_symptoms")
    slots_now = state.get("slots") if isinstance(state.get("slots"), dict) else {}
    names = ok_aligned_symptom_names(slots_now.get("alignments") if isinstance(slots_now, dict) else None)
    names.extend(_symptom_mentions_from_state(state))
    if not names:
        return empty
    current_hits = names_in_main_symptom_catalog(names, current_catalog, drop_equal_to=disease)

    payload: B2BRouterState = {
        "pending_current_main_hits": current_hits,
        "rollback_candidate_diseases": [],
        "rollback_decision": None,
        "selected_symptoms": [],
    }
    global_catalog, diseases_of = build_global_main_symptom_catalog()
    current_set = set(current_catalog)
    other_diseases: list[str] = []
    for name in names_in_main_symptom_catalog(names, global_catalog, drop_equal_to=disease):
        if name in current_set:
            continue
        for owner in diseases_of.get(name) or []:
            if owner and owner != disease and owner not in other_diseases:
                other_diseases.append(owner)
    if other_diseases and not current_hits:
        payload["rollback_candidate_diseases"] = other_diseases
        return payload
    if current_hits:
        payload["selected_symptoms"] = current_hits
        payload["collected_symptoms"] = _merge_collected_symptoms(state, current_hits)
    return payload


def confirm_disease_rollback(state: B2BRouterState) -> B2BRouterState:
    others = [str(item).strip() for item in (state.get("rollback_candidate_diseases") or []) if str(item).strip()]
    current = state.get("disease_name") or state.get("selected_disease") or "当前疾病"
    other_txt = "、".join(others) or "其他疾病"
    fallback = (
        f"【第三轮】根据前面已收集的不适，在全部疾病证型的主症中检索到更接近：{other_txt}，"
        f"与当前锁定的「{current}」不一致。\n"
        "是否回到第二轮重新选择疾病？\n\n"
        "1. 是，回到第二轮换疾病\n"
        "2. 否，仍按当前疾病继续"
    )
    reply = _ensure_round_marker(fallback, {**dict(state), "major_round": 3}, round_no=3)
    return {
        "next_action": "await_user",
        "messages": [AIMessage(content=reply)],
        "major_round": 3,
        "session_mode": "round3_main_symptom",
    }


def parse_rollback_decision(state: B2BRouterState) -> B2BRouterState:
    decision = classify_rollback_reply(extract_query(state))
    if decision == "unclear":
        return {"rollback_decision": "unclear", "needs_clarification": True}
    if decision == "yes":
        suggested = [str(item).strip() for item in (state.get("rollback_candidate_diseases") or []) if str(item).strip()]
        return {
            "rollback_decision": "yes",
            "needs_clarification": False,
            "disease_name": None,
            "selected_disease": None,
            "disease_id": None,
            "candidate_syndromes": [],
            "selected_symptoms": [],
            "disease_suggested": suggested,
            "major_round": 2,
            "session_mode": "round2_disease",
            "query": "",
            "rollback_candidate_diseases": [],
            "pending_current_main_hits": [],
            "symptom_options": [],
        }
    hits = [str(item).strip() for item in (state.get("pending_current_main_hits") or []) if str(item).strip()]
    return {
        "rollback_decision": "no",
        "needs_clarification": False,
        "selected_symptoms": hits,
        "rollback_candidate_diseases": [],
        "query": "",
        "major_round": 3,
        "session_mode": "round3_main_symptom",
    }


def clarify_round2(state: B2BRouterState) -> B2BRouterState:
    fallback = "请从疾病列表中选择一项，或输入更具体的症状描述。"
    reply = _round2_unfilled_reply(
        state,
        fallback=fallback,
        task="第二轮澄清：疾病未锁定。结合已抽症状引导点选疾病列表或补充描述。",
    )
    return {
        "next_action": "await_user",
        "messages": [AIMessage(content=reply)],
    }


def parse_symptom_input(state: B2BRouterState) -> B2BRouterState:
    """解析症状输入：第三/四轮走 UIE 对齐链，其余沿用分词。"""
    correction_update, _ = correct_query_in_state(state)
    working_state = {**dict(state), **correction_update}
    major_round = int(state.get("major_round") or 1)
    if major_round == 3:
        parsed = dispatch_main_symptom_input(working_state)
    elif major_round == 4:
        parsed = dispatch_auxiliary_symptom_input(working_state)
    else:
        parsed = dispatch_symptom_input(working_state)
    if parsed.needs_clarification:
        payload: B2BRouterState = {
            **correction_update,
            "needs_clarification": True,
            "selected_symptoms": [],
            "relevance_reason": parsed.reason,
        }
        extra: list[str] = []
        if parsed.slots:
            payload["slots"] = parsed.slots
            extra.extend(str(item).strip() for item in (parsed.slots.get("symptoms") or []) if str(item).strip())
            extra.extend(str(item).strip() for item in ((parsed.slots.get("entities") or {}).get("症状") or []) if str(item).strip())
        payload["collected_symptoms"] = _merge_collected_symptoms(working_state, extra)
        if parsed.text_intent:
            payload["text_intent"] = parsed.text_intent
        if parsed.text_intent_reason:
            payload["text_intent_reason"] = parsed.text_intent_reason
        return payload
    payload = {
        **correction_update,
        "needs_clarification": False,
        "selected_symptoms": parsed.selected_symptoms,
        "query": "",
    }
    if parsed.slots:
        payload["slots"] = parsed.slots
        extra = list(parsed.selected_symptoms or [])
        extra.extend(str(item).strip() for item in (parsed.slots.get("symptoms") or []) if str(item).strip())
        extra.extend(str(item).strip() for item in ((parsed.slots.get("entities") or {}).get("症状") or []) if str(item).strip())
        payload["collected_symptoms"] = _merge_collected_symptoms(working_state, extra)
    else:
        payload["collected_symptoms"] = _merge_collected_symptoms(working_state, parsed.selected_symptoms)
    if parsed.text_intent:
        payload["text_intent"] = parsed.text_intent
    if parsed.text_intent_reason:
        payload["text_intent_reason"] = parsed.text_intent_reason
    return payload


def offer_main_symptoms(state: B2BRouterState) -> B2BRouterState:
    syndromes = state.get("candidate_syndromes") or []
    options, matched, align_rows = _align_collected_main_symptoms(state, syndromes)
    if not options:
        options = union_symptoms(syndromes, field="main_symptoms")
    intro = (
        f"【第三轮】疾病：{state.get('disease_name')}。\n"
        "请描述您目前的主要不适症状（可以说多个，如咳嗽、头痛、发热等）。\n"
        "常见主症参考："
    )
    fallback = _format_options(intro, options)
    reply_state = {**dict(state), "symptom_options": options, "major_round": 3}
    if matched:
        reply_state["selected_symptoms"] = matched
        fallback = (
            f"【第三轮】已根据前面提到的不适对齐到：{'、'.join(matched)}。\n"
            "如还有其他主要不适，请补充；也可直接从下面点选。\n"
            + _format_options("常见主症参考：", options)
        )
    reply = _user_reply(
        reply_state,
        task="第三轮邀请描述主症；若已有对齐到的标准主症请点明，并附常见主症编号列表。",
        fallback=fallback,
    )
    payload: B2BRouterState = {
        "symptom_options": options,
        "major_round": 3,
        "user_input_mode": "text",
        "next_action": "await_user",
        "messages": [AIMessage(content=reply)],
        "collected_symptoms": _merge_collected_symptoms(state, matched),
        "selected_symptoms": [],
    }
    if align_rows:
        slots = dict(state.get("slots") or {}) if isinstance(state.get("slots"), dict) else {}
        slots["alignments"] = list(slots.get("alignments") or []) + align_rows
        payload["slots"] = slots
    return payload


def apply_main_symptoms(state: B2BRouterState) -> B2BRouterState:
    options = [
        str(item).strip()
        for item in (state.get("symptom_options") or [])
        if str(item).strip()
    ]
    selected = resolve_symptom_catalog_picks(
        list(state.get("selected_symptoms") or []) or parse_symptom_selections(state),
        options,
    )
    if not selected:
        selected = list(state.get("selected_symptoms") or []) or parse_symptom_selections(state)
    disease = str(state.get("disease_name") or state.get("selected_disease") or "").strip()
    if disease:
        selected = [item for item in selected if str(item).strip() != disease]
    syndromes = state.get("candidate_syndromes") or []
    if not selected:
        return {"needs_clarification": True}

    filtered = filter_syndromes_by_main_symptoms(syndromes, selected)
    from_option_list = bool(resolve_symptom_catalog_picks(selected, options))
    if not filtered and from_option_list:
        filtered = list(syndromes)
    if not filtered:
        return {
            "needs_clarification": True,
            "messages": [
                AIMessage(
                    content=_user_reply(
                        state,
                        task="第三轮：用户描述未能匹配当前疾病主症，请请用户补充或换种说法。",
                        fallback="描述的症状未能匹配到当前疾病下的证型主症，请补充或换种说法。",
                    )
                )
            ],
        }

    return {
        "needs_clarification": False,
        "confirmed_main_symptoms": selected,
        "candidate_syndromes": filtered,
        "selected_symptoms": [],
        "auxiliary_round": 0,
        "major_round": 4,
        "session_mode": "round4_auxiliary",
    }


def clarify_main_symptoms(state: B2BRouterState) -> B2BRouterState:
    reason = (state.get("relevance_reason") or "").strip()
    hint = f"（{reason}）" if reason else ""
    fallback = f"请具体描述您的主要不适症状，可参考上方常见主症。{hint}"
    reply = _user_reply(
        state,
        task="第三轮澄清：请用户具体描述主要不适症状。",
        fallback=fallback,
    )
    return {
        "next_action": "await_user",
        "messages": [AIMessage(content=reply)],
    }


def offer_auxiliary_symptoms(state: B2BRouterState) -> B2BRouterState:
    syndromes = state.get("candidate_syndromes") or []
    round_no = int(state.get("auxiliary_round") or 0) + 1
    if round_no > MAX_AUXILIARY_ROUNDS:
        round_no = MAX_AUXILIARY_ROUNDS
    syndrome_names = [item.get("syndrome_name", "") for item in syndromes]
    exclude = {str(item).strip() for item in (state.get("confirmed_main_symptoms") or []) if str(item).strip()}
    catalog = build_auxiliary_symptom_catalog(syndromes, exclude=exclude)
    intro = (
        f"【第四轮】当前剩余 {len(syndromes)} 个潜在证型："
        f"{'、'.join(name for name in syndrome_names if name)}。\n"
        f"请从下列辅助症状中选择符合的（可多选），或用文字补充（第 {round_no}/{MAX_AUXILIARY_ROUNDS} 次）。\n"
    )
    fallback = _format_options(intro + "剩余证型相关辅症：", catalog) if catalog else intro + "暂无结构化辅症列表，请用文字补充其他不适。"
    reply = _user_reply(
        {**dict(state), "major_round": 4, "symptom_options": catalog},
        task="第四轮：根据当前剩余证型列出其辅助症状编号，请用户点选或补充；不要编造证型名。",
        fallback=fallback,
    )
    return {
        "auxiliary_symptom_catalog": catalog,
        "symptom_options": catalog,
        "major_round": 4,
        "auxiliary_round": round_no,
        "user_input_mode": "multi_option" if catalog else "text",
        "next_action": "await_user",
        "selected_symptoms": [],
        "messages": [AIMessage(content=reply)],
    }


def apply_auxiliary_symptoms(state: B2BRouterState) -> B2BRouterState:
    catalog = [
        str(item).strip()
        for item in (state.get("symptom_options") or state.get("auxiliary_symptom_catalog") or [])
        if str(item).strip()
    ]
    selected = resolve_symptom_catalog_picks(
        list(state.get("selected_symptoms") or []) or parse_symptom_selections(state),
        catalog,
    )
    if not selected:
        selected = list(state.get("selected_symptoms") or []) or parse_symptom_selections(state)
    syndromes = state.get("candidate_syndromes") or []
    current_round = int(state.get("auxiliary_round") or 0)

    if not selected:
        if current_round >= MAX_AUXILIARY_ROUNDS:
            return {"needs_clarification": False, "auxiliary_round": current_round}
        return {
            "needs_clarification": True,
            "auxiliary_round": current_round,
        }

    filtered = filter_syndromes_by_auxiliary_symptoms(syndromes, selected)
    if not filtered:
        if current_round >= MAX_AUXILIARY_ROUNDS:
            return {"needs_clarification": False, "auxiliary_round": current_round}
        return {
            "needs_clarification": True,
            "auxiliary_round": current_round,
            "messages": [
                AIMessage(
                    content=_user_reply(
                        state,
                        task="第四轮：补充症状未能区分证型，请用户再描述其他不适。",
                        fallback="补充的症状未能帮助区分证型，请再描述其他不适表现。",
                    )
                )
            ],
        }
    return {
        "needs_clarification": False,
        "confirmed_auxiliary_symptoms": selected,
        "candidate_syndromes": filtered,
        "selected_symptoms": [],
        "auxiliary_round": current_round,
    }


def clarify_auxiliary_symptoms(state: B2BRouterState) -> B2BRouterState:
    reason = (state.get("relevance_reason") or "").strip()
    hint = f"（{reason}）" if reason else ""
    fallback = f"请补充其他不适表现，帮助进一步辨别证型。{hint}"
    reply = _user_reply(
        state,
        task="第四轮澄清：请用户补充其他不适表现。",
        fallback=fallback,
    )
    return {
        "next_action": "await_user",
        "messages": [AIMessage(content=reply)],
    }


def emit_unique_syndrome(state: B2BRouterState) -> B2BRouterState:
    syndromes = state.get("candidate_syndromes") or []
    matched = syndromes[0] if syndromes else None
    name = (matched or {}).get("syndrome_name", "未知")
    fallback = (
        f"辨证完成。\n"
        f"部位：{state.get('body_part')}\n"
        f"疾病：{state.get('disease_name')}\n"
        f"证型：{name}"
    )
    reply = _with_consult_ended(
        _user_reply(
            state,
            task="辨证已唯一确定证型，用口语总结部位、疾病、证型；不要加处方，不要邀请继续补充症状。末尾将由系统提示问诊结束。",
            fallback=fallback,
        )
    )
    return {
        "matched_syndrome": matched,
        "final_syndromes": syndromes,
        "session_mode": "resolved",
        "should_end": True,
        "next_action": "end",
        "messages": [AIMessage(content=reply)],
    }


def emit_ambiguous_syndromes(state: B2BRouterState) -> B2BRouterState:
    syndromes = state.get("candidate_syndromes") or []
    names = "、".join(item.get("syndrome_name", "") for item in syndromes) or "无"
    fallback = (
        f"已完成 {MAX_AUXILIARY_ROUNDS} 轮辅助辩证，仍无法唯一确定证型。\n"
        f"部位：{state.get('body_part')}\n"
        f"疾病：{state.get('disease_name')}\n"
        f"潜在证型：{names}"
    )
    reply = _with_consult_ended(
        _user_reply(
            state,
            task="辅助轮次已满仍无法唯一确定证型，如实告知全部剩余潜在证型，不要编造唯一结论，不要邀请继续问诊。",
            fallback=fallback,
        )
    )
    return {
        "matched_syndrome": None,
        "final_syndromes": syndromes,
        "session_mode": "ambiguous",
        "should_end": True,
        "next_action": "end",
        "messages": [AIMessage(content=reply)],
    }


CONSULT_NODES = {
    "opening_round1": opening_round1,
    "parse_round1_input": parse_round1_input,
    "classify_text_intent_round1": classify_text_intent_round1,
    "respond_text_intent_round1": respond_text_intent_round1,
    "thinking_relevance_round1": thinking_relevance_round1,
    "apply_body_part": apply_body_part,
    "clarify_round1": clarify_round1,
    "offer_disease_selection": offer_disease_selection,
    "parse_round2_input": parse_round2_input,
    "classify_text_intent_round2": classify_text_intent_round2,
    "respond_text_intent_round2": respond_text_intent_round2,
    "thinking_relevance_round2": thinking_relevance_round2,
    "apply_disease": apply_disease,
    "retrieve_collected_main_symptoms": retrieve_collected_main_symptoms,
    "confirm_disease_rollback": confirm_disease_rollback,
    "parse_rollback_decision": parse_rollback_decision,
    "clarify_round2": clarify_round2,
    "offer_main_symptoms": offer_main_symptoms,
    "parse_symptom_input": parse_symptom_input,
    "apply_main_symptoms": apply_main_symptoms,
    "clarify_main_symptoms": clarify_main_symptoms,
    "offer_auxiliary_symptoms": offer_auxiliary_symptoms,
    "apply_auxiliary_symptoms": apply_auxiliary_symptoms,
    "clarify_auxiliary_symptoms": clarify_auxiliary_symptoms,
    "emit_unique_syndrome": emit_unique_syndrome,
    "emit_ambiguous_syndromes": emit_ambiguous_syndromes,
}

__all__ = ["CONSULT_NODES"]
