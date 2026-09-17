"""B2B 图节点进度：astream(updates) → 前端 progress 事件。"""

from __future__ import annotations

from typing import Any

__all__ = ["NODE_PROGRESS", "progress_events_from_update"]

NODE_PROGRESS: dict[str, str] = {
    "opening_round1": "已发出第一轮开场",
    "parse_round1_input": "正在解析部位输入",
    "classify_text_intent_round1": "正在识别意图并抽槽",
    "respond_text_intent_round1": "正在组织第一轮回复",
    "thinking_relevance_round1": "正在匹配部位",
    "apply_body_part": "正在锁定部位",
    "clarify_round1": "部位未齐，正在追问",
    "offer_disease_selection": "正在给出疾病候选项",
    "parse_round2_input": "正在解析疾病输入",
    "classify_text_intent_round2": "正在识别第二轮意图",
    "respond_text_intent_round2": "正在组织第二轮回复",
    "thinking_relevance_round2": "正在匹配疾病",
    "apply_disease": "正在锁定疾病",
    "retrieve_collected_main_symptoms": "正在用已收集症状检索全库主症",
    "confirm_disease_rollback": "主症指向其他疾病，正在确认是否回退",
    "parse_rollback_decision": "正在解析是否回退第二轮",
    "clarify_round2": "疾病未齐，正在追问",
    "offer_main_symptoms": "正在邀请描述主症",
    "parse_symptom_input": "正在解析症状",
    "apply_main_symptoms": "正在用主症缩窄证型",
    "clarify_main_symptoms": "主症未齐，正在追问",
    "offer_auxiliary_symptoms": "正在邀请补充其他不适",
    "apply_auxiliary_symptoms": "正在用辅症辨别证型",
    "clarify_auxiliary_symptoms": "辅症未齐，正在追问",
    "emit_unique_syndrome": "证型已唯一确定",
    "emit_ambiguous_syndromes": "仍有多个潜在证型，正在收口",
}


def progress_events_from_update(chunk: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if "__interrupt__" in chunk:
        events.append(
            {
                "event": "graph_interrupt",
                "message": "图已暂停，等待用户输入",
            }
        )
    for node_name, delta in chunk.items():
        if node_name == "__interrupt__":
            continue
        message = NODE_PROGRESS.get(node_name) or f"已完成：{node_name}"
        events.append(
            {
                "event": "progress",
                "node": node_name,
                "message": message,
                "has_delta": delta is not None,
            }
        )
    return events
