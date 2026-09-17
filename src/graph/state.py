"""B2B LangGraph 总状态。"""

from __future__ import annotations

from .states import ConsultState, CoreDialogueState, TextInputState

__all__ = ["B2BRouterState", "initial_b2b_state"]


class B2BRouterState(CoreDialogueState, ConsultState, TextInputState, total=False):
    """B2B 辨证路由状态。"""


def initial_b2b_state() -> B2BRouterState:
    return B2BRouterState(
        messages=[],
        session_mode="idle",
        major_round=1,
        user_input_mode="option",
        input_channel="text",
        raw_query="",
        query="",
        user_id=None,
        org_id=None,
        body_part=None,
        body_part_id=None,
        selected_body_part=None,
        disease_name=None,
        disease_id=None,
        selected_disease=None,
        body_part_options=[],
        disease_options=[],
        disease_asked=False,
        candidate_syndromes=[],
        confirmed_main_symptoms=[],
        confirmed_auxiliary_symptoms=[],
        auxiliary_round=0,
        symptom_options=[],
        auxiliary_symptom_catalog=[],
        selected_symptoms=[],
        collected_symptoms=[],
        pending_current_main_hits=[],
        rollback_candidate_diseases=[],
        rollback_decision=None,
        matched_syndrome=None,
        final_syndromes=[],
        needs_clarification=False,
        next_action="await_user",
        should_end=False,
    )
