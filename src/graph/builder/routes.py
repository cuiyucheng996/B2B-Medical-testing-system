"""B2B 条件边路由。"""

from __future__ import annotations

from typing import Literal

from schema.b2b import MAX_AUXILIARY_ROUNDS
from algorithm.input_parse import is_localized_body_part, is_systemic_unlocalized_text
from algorithm.intent_slots import slots_have_mentions
from algorithm.text_input import extract_query

from ..state import B2BRouterState

__all__ = [
    "route_after_apply_auxiliary",
    "route_after_apply_body_part",
    "route_after_apply_disease",
    "route_after_apply_main_symptoms",
    "route_after_retrieve_collected_main_symptoms",
    "route_after_parse_rollback_decision",
    "route_after_classify_text_intent_round1",
    "route_after_classify_text_intent_round2",
    "route_after_parse_round1_input",
    "route_after_parse_round2_input",
    "route_after_parse_symptom_input",
    "route_after_thinking_relevance_round1",
    "route_after_thinking_relevance_round2",
]

ClarifyRoute = Literal["clarify", "continue", "has_disease"]
DiseaseApplyRoute = Literal["clarify", "continue"]
RetrieveMainRoute = Literal["offer", "apply", "confirm"]
RollbackParseRoute = Literal["rollback", "apply", "offer", "reconfirm"]
MainResultRoute = Literal["unique", "need_aux", "clarify"]
AuxResultRoute = Literal["unique", "retry_aux", "ambiguous", "clarify"]
ParseSymptomRoute = Literal["clarify", "continue", "clarify_aux", "apply_aux", "ambiguous"]
ParseInputRoute = Literal["option", "classify"]
TextIntentRoute = Literal["complaint", "respond"]
ThinkingBodyRoute = Literal["respond", "continue"]
ThinkingDiseaseRoute = Literal["respond", "continue"]

_HANDLER_TO_PARSE_ROUTE: dict[str, ParseInputRoute] = {
    "option_input": "option",
    "text_input": "classify",
    "vague_input": "classify",
    "option_agent": "option",
    "text_agent": "classify",
    "vague_agent": "classify",
}


def route_after_parse_round1_input(state: B2BRouterState) -> ParseInputRoute:
    handler = state.get("active_handler")
    if handler and handler in _HANDLER_TO_PARSE_ROUTE:
        return _HANDLER_TO_PARSE_ROUTE[handler]
    mode = state.get("user_input_mode", "option")
    if mode == "text":
        return "classify"
    if mode == "option":
        return "option"
    return "classify"


def route_after_classify_text_intent_round1(state: B2BRouterState) -> TextIntentRoute:
    query = extract_query(state)
    locked = state.get("body_part") or state.get("selected_body_part")
    if is_systemic_unlocalized_text(query) or not is_localized_body_part(locked):
        locked = None
    if locked:
        return "complaint"
    if state.get("text_intent") != "complaint":
        return "respond"
    slots = state.get("slots") if isinstance(state.get("slots"), dict) else {}
    slot_part = slots.get("body_part") if slots.get("filled") else None
    if is_localized_body_part(slot_part) and not is_systemic_unlocalized_text(query):
        return "complaint"
    # 抽到了口语部位但对不齐六个标准部位（对齐失败/阈值/未召回）→ 继续第一轮，不要整句打 Milvus
    if "部位" in (slots.get("missing") or []):
        return "respond"
    # 没有抽到 span 时不要进 thinking（会拿整句打 Milvus）
    if not slots_have_mentions(slots):
        return "respond"
    return "complaint"


def route_after_thinking_relevance_round1(state: B2BRouterState) -> ThinkingBodyRoute:
    locked = state.get("body_part") or state.get("selected_body_part")
    if is_localized_body_part(locked) and not is_systemic_unlocalized_text(extract_query(state)):
        return "continue"
    return "respond"


def route_after_classify_text_intent_round2(state: B2BRouterState) -> TextIntentRoute:
    if state.get("disease_name") or state.get("selected_disease"):
        return "complaint"
    if state.get("text_intent") != "complaint":
        return "respond"
    slots = state.get("slots") if isinstance(state.get("slots"), dict) else {}
    if slots.get("filled") and slots.get("disease_name"):
        return "complaint"
    if not slots_have_mentions(slots, ("疾病",)):
        return "respond"
    return "complaint"


def route_after_parse_round2_input(state: B2BRouterState) -> ParseInputRoute:
    return route_after_parse_round1_input(state)


def route_after_thinking_relevance_round2(state: B2BRouterState) -> ThinkingDiseaseRoute:
    if state.get("disease_name") or state.get("selected_disease"):
        return "continue"
    return "respond"


def route_after_parse_symptom_input(state: B2BRouterState) -> ParseSymptomRoute:
    round_no = int(state.get("major_round") or 1)
    aux_round = int(state.get("auxiliary_round") or 0)
    if round_no >= 4:
        if state.get("needs_clarification"):
            if aux_round >= MAX_AUXILIARY_ROUNDS:
                return "ambiguous"
            return "clarify_aux"
        return "apply_aux"
    if state.get("needs_clarification"):
        return "clarify"
    return "continue"


def route_after_apply_body_part(state: B2BRouterState) -> ClarifyRoute:
    if state.get("needs_clarification"):
        return "clarify"
    if state.get("disease_name") or state.get("selected_disease"):
        return "has_disease"
    return "continue"


def route_after_apply_disease(state: B2BRouterState) -> DiseaseApplyRoute:
    if state.get("needs_clarification"):
        return "clarify"
    return "continue"


def route_after_retrieve_collected_main_symptoms(state: B2BRouterState) -> RetrieveMainRoute:
    if state.get("rollback_candidate_diseases"):
        return "confirm"
    if state.get("selected_symptoms"):
        return "apply"
    return "offer"


def route_after_parse_rollback_decision(state: B2BRouterState) -> RollbackParseRoute:
    if state.get("rollback_decision") == "yes":
        return "rollback"
    if state.get("rollback_decision") == "unclear" or state.get("needs_clarification"):
        return "reconfirm"
    if state.get("selected_symptoms"):
        return "apply"
    return "offer"


def route_after_apply_main_symptoms(state: B2BRouterState) -> MainResultRoute:
    if state.get("needs_clarification"):
        return "clarify"
    return "need_aux"


def route_after_apply_auxiliary(state: B2BRouterState) -> AuxResultRoute:
    syndromes = state.get("candidate_syndromes") or []
    aux_round = int(state.get("auxiliary_round") or 0)
    if not state.get("needs_clarification") and len(syndromes) == 1:
        return "unique"
    if aux_round >= MAX_AUXILIARY_ROUNDS:
        return "ambiguous"
    if state.get("needs_clarification"):
        return "clarify"
    return "retry_aux"
