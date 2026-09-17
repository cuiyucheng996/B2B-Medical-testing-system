"""B2B 图级共享 state 切片。"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from schema.b2b import (
    InputChannel,
    InputHandler,
    MajorRound,
    NextAction,
    SessionMode,
    TextIntent,
    UserInputMode,
)

__all__ = [
    "ConsultState",
    "CoreDialogueState",
    "TextInputState",
]


class TextInputState(TypedDict, total=False):
    active_handler: InputHandler | None
    is_relevant: bool | None
    relevance_reason: str | None
    active_route: str | None
    disease_suggested: list[str]
    needs_text_input: bool
    spell_correction: dict[str, Any]
    spell_correction_detail: dict[str, Any]
    text_intent: TextIntent | None
    text_intent_confidence: float | None
    text_intent_reason: str | None
    text_intent_source: str | None
    slots: dict[str, Any]


class CoreDialogueState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    session_mode: SessionMode
    major_round: MajorRound
    user_input_mode: UserInputMode
    input_channel: InputChannel
    raw_query: str
    query: str
    next_action: NextAction | None
    should_end: bool
    user_id: str | None
    org_id: str | None


class ConsultState(TypedDict, total=False):
    body_part: str | None
    body_part_id: int | None
    selected_body_part: str | None
    disease_name: str | None
    disease_id: int | None
    selected_disease: str | None
    body_part_options: list[str]
    disease_options: list[str]
    disease_asked: bool
    candidate_syndromes: list[dict[str, Any]]
    confirmed_main_symptoms: Annotated[list[str], operator.add]
    confirmed_auxiliary_symptoms: Annotated[list[str], operator.add]
    auxiliary_round: int
    symptom_options: list[str]
    auxiliary_symptom_catalog: list[str]
    selected_symptoms: list[str]
    collected_symptoms: list[str]
    pending_current_main_hits: list[str]
    rollback_candidate_diseases: list[str]
    rollback_decision: str | None
    matched_syndrome: dict[str, Any] | None
    final_syndromes: list[dict[str, Any]]
    needs_clarification: bool
