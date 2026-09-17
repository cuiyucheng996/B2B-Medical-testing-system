"""注册 B2B 辨证图节点与边。"""

from __future__ import annotations

from langgraph.graph import END, START
from langgraph.graph.state import StateGraph

from ..nodes import CONSULT_NODES
from .routes import (
    route_after_apply_auxiliary,
    route_after_apply_body_part,
    route_after_apply_disease,
    route_after_apply_main_symptoms,
    route_after_retrieve_collected_main_symptoms,
    route_after_parse_rollback_decision,
    route_after_classify_text_intent_round1,
    route_after_classify_text_intent_round2,
    route_after_parse_round1_input,
    route_after_parse_round2_input,
    route_after_parse_symptom_input,
    route_after_thinking_relevance_round1,
    route_after_thinking_relevance_round2,
)

__all__ = [
    "INTERRUPT_AFTER",
    "add_consult_edges",
    "add_consult_nodes",
]

INTERRUPT_AFTER: tuple[str, ...] = (
    "opening_round1",
    "respond_text_intent_round1",
    "clarify_round1",
    "offer_disease_selection",
    "respond_text_intent_round2",
    "clarify_round2",
    "offer_main_symptoms",
    "clarify_main_symptoms",
    "confirm_disease_rollback",
    "offer_auxiliary_symptoms",
    "clarify_auxiliary_symptoms",
)


def add_consult_nodes(builder: StateGraph) -> None:
    for name, fn in CONSULT_NODES.items():
        builder.add_node(name, fn)


def add_consult_edges(builder: StateGraph) -> None:
    builder.add_edge(START, "opening_round1")
    builder.add_edge("opening_round1", "parse_round1_input")

    builder.add_conditional_edges(
        "parse_round1_input",
        route_after_parse_round1_input,
        {
            "option": "apply_body_part",
            "classify": "classify_text_intent_round1",
        },
    )
    builder.add_conditional_edges(
        "classify_text_intent_round1",
        route_after_classify_text_intent_round1,
        {
            "complaint": "thinking_relevance_round1",
            "respond": "respond_text_intent_round1",
        },
    )
    builder.add_conditional_edges(
        "thinking_relevance_round1",
        route_after_thinking_relevance_round1,
        {
            "respond": "respond_text_intent_round1",
            "continue": "apply_body_part",
        },
    )
    builder.add_edge("respond_text_intent_round1", "parse_round1_input")

    builder.add_conditional_edges(
        "apply_body_part",
        route_after_apply_body_part,
        {"clarify": "clarify_round1", "continue": "offer_disease_selection", "has_disease": "apply_disease"},
    )
    builder.add_edge("clarify_round1", "parse_round1_input")

    builder.add_edge("offer_disease_selection", "parse_round2_input")
    builder.add_conditional_edges(
        "parse_round2_input",
        route_after_parse_round2_input,
        {
            "option": "apply_disease",
            "classify": "classify_text_intent_round2",
        },
    )
    builder.add_conditional_edges(
        "classify_text_intent_round2",
        route_after_classify_text_intent_round2,
        {
            "complaint": "thinking_relevance_round2",
            "respond": "respond_text_intent_round2",
        },
    )
    builder.add_conditional_edges(
        "thinking_relevance_round2",
        route_after_thinking_relevance_round2,
        {
            "respond": "respond_text_intent_round2",
            "continue": "apply_disease",
        },
    )
    builder.add_edge("respond_text_intent_round2", "parse_round2_input")

    builder.add_conditional_edges(
        "apply_disease",
        route_after_apply_disease,
        {"clarify": "clarify_round2", "continue": "retrieve_collected_main_symptoms"},
    )
    builder.add_edge("clarify_round2", "parse_round2_input")
    builder.add_conditional_edges(
        "retrieve_collected_main_symptoms",
        route_after_retrieve_collected_main_symptoms,
        {
            "offer": "offer_main_symptoms",
            "apply": "apply_main_symptoms",
            "confirm": "confirm_disease_rollback",
        },
    )
    builder.add_edge("confirm_disease_rollback", "parse_rollback_decision")
    builder.add_conditional_edges(
        "parse_rollback_decision",
        route_after_parse_rollback_decision,
        {
            "rollback": "offer_disease_selection",
            "apply": "apply_main_symptoms",
            "offer": "offer_main_symptoms",
            "reconfirm": "confirm_disease_rollback",
        },
    )

    builder.add_edge("offer_main_symptoms", "parse_symptom_input")
    builder.add_conditional_edges(
        "parse_symptom_input",
        route_after_parse_symptom_input,
        {
            "clarify": "clarify_main_symptoms",
            "continue": "apply_main_symptoms",
            "clarify_aux": "clarify_auxiliary_symptoms",
            "apply_aux": "apply_auxiliary_symptoms",
            "ambiguous": "emit_ambiguous_syndromes",
        },
    )
    builder.add_conditional_edges(
        "apply_main_symptoms",
        route_after_apply_main_symptoms,
        {
            "clarify": "clarify_main_symptoms",
            "unique": "emit_unique_syndrome",
            "need_aux": "offer_auxiliary_symptoms",
        },
    )
    builder.add_edge("clarify_main_symptoms", "parse_symptom_input")

    builder.add_edge("offer_auxiliary_symptoms", "parse_symptom_input")
    builder.add_conditional_edges(
        "apply_auxiliary_symptoms",
        route_after_apply_auxiliary,
        {
            "unique": "emit_unique_syndrome",
            "retry_aux": "offer_auxiliary_symptoms",
            "ambiguous": "emit_ambiguous_syndromes",
            "clarify": "clarify_auxiliary_symptoms",
        },
    )
    builder.add_edge("clarify_auxiliary_symptoms", "parse_symptom_input")

    builder.add_edge("emit_unique_syndrome", END)
    builder.add_edge("emit_ambiguous_syndromes", END)
