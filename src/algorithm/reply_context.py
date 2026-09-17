"""答复 Agent 上下文与 prompt 渲染（对齐主项目 main_agent 模板填槽）。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from langchain_core.messages import BaseMessage, HumanMessage

__all__ = [
    "ReplyAgentRunContext",
    "build_reply_agent_context",
    "build_reply_agent_messages",
    "build_reply_agent_packet",
    "render_reply_agent_system_prompt",
]

_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompt" / "reply_agent_system_prompt.txt"

_PHASE_TITLES = {
    1: "第一轮 · 锁定不适部位",
    2: "第二轮 · 锁定疾病",
    3: "第三轮 · 主症缩窄证型",
    4: "第四轮 · 辅助症状辨别证型",
}

_OPERATION_RULES = {
    1: """\
1. 寒暄/无关 → 礼貌拉回，请用户描述哪里不舒服或点选部位列表。回复以【第一轮】开头。
2. 主诉但部位未齐：结合已抽实体追问，或请从部位候选项中点选。
3. 不得把症状口语说成已锁定部位以外的新部位名。
4. 列出部位时必须用数据包中的编号候选项。""",
    2: """\
1. 目标是锁定当前部位下的疾病候选项，不是确诊。回复以【第二轮】开头。
2. 第一次展示：编号列出疾病原名，邀请点选或文字描述。
3. 已问过一次仍未填疾病槽：不要只空喊点选；上游会另路输出「编号 + 病名 + 约15字白话」。
4. 不要把「胸闷」「上不来气」等口语直接说成已确诊病名。
5. 列出疾病时必须用数据包中的编号候选项原名。""",
    3: """\
1. 疾病已锁定，请用户描述主要不适；可附常见主症编号列表作参考。回复以【第三轮】开头。
2. 未能匹配主症时，请换种说法或补充，不要编造证型。
3. 若命中其他疾病主症、正在确认是否回退第二轮：只根据数据包里的候选疾病名提问，必须保留 1/2。
4. 不要提前给出证型结论。""",
    4: """\
1. 用补充不适帮助区分潜在证型；证型名只用上下文中已有的。回复以【第四轮】开头。
2. 必须用数据包中的辅助症状编号列表（来自当前剩余证型），不要说暂不展示。
3. 最多询问 2 次；仍不唯一时如实列出全部剩余潜在证型，不要编造唯一结论。
4. 已唯一确定：口语总结部位、疾病、证型，不加处方。
5. 收口时不要邀请用户继续补充症状或继续问诊。""",
}

_SUB_AGENTS = "（本环节不委派；纠错 / 意图 / 抽取已由上游算法完成。）"

_RED_LINES = """\
- 只根据数据包里的文本、意图、实体、槽位、候选项说话。
- 禁止编造未出现的部位、疾病、证型、处方、用药。
- 禁止输出诊断结论口吻（「您得了××」）。
- 寒暄拉回问诊；模糊则追问数据包中的 missing 槽。
- 敏感内容礼貌拒绝并引导回辨证。"""


@dataclass(slots=True)
class ReplyAgentRunContext:
    major_round: int
    query: str
    task: str
    body_part: str | None
    disease_name: str | None
    recent_dialogue: str
    phase_title: str
    phase_context: str
    packet: str


def _join(items: list[Any] | None, *, limit: int = 20) -> str:
    values = [str(item).strip() for item in (items or []) if str(item).strip()]
    if not values:
        return "（无）"
    return "、".join(values[:limit])


def _is_user_turn(item: Any) -> bool:
    role = getattr(item, "type", None) or (item.get("role") if isinstance(item, dict) else "")
    return str(role or "").lower() in ("human", "user")


def _strip_trailing_user_turns(messages: list[Any]) -> list[Any]:
    """checkpoint 回退时去掉末尾未完成的本轮用户话。"""
    items = list(messages or [])
    while items and _is_user_turn(items[-1]):
        items.pop()
    return items


def _recent_dialogue(messages: list[Any], *, limit: int | None = None) -> str:
    from algorithm.short_memory import load_b2b_short_messages

    cap = 40 if limit is None else limit

    rows = load_b2b_short_messages()
    if rows:
        lines: list[str] = []
        for item in rows:
            role = "用户" if item.get("role") == "user" else "助手"
            text = str(item.get("content") or "").strip()
            if text:
                lines.append(f"{role}：{text}")
        if lines:
            return "\n".join(lines)

    lines = []
    for item in _strip_trailing_user_turns(messages)[-cap:]:
        role = getattr(item, "type", None) or (item.get("role") if isinstance(item, dict) else "ai")
        content = getattr(item, "content", None)
        if content is None and isinstance(item, dict):
            content = item.get("content")
        text = str(content or "").strip()
        if not text:
            continue
        label = "用户" if role in ("human", "user") else "助手"
        lines.append(f"{label}：{text}")
    return "\n".join(lines) if lines else "（无）"


def build_reply_agent_packet(state: Mapping[str, Any], *, task: str) -> str:
    """意图与抽槽之后的完整包，作为单 Agent 的 human 消息。"""
    slots = state.get("slots") if isinstance(state.get("slots"), dict) else {}
    spell = state.get("spell_correction_detail") if isinstance(state.get("spell_correction_detail"), dict) else {}
    syndromes = state.get("candidate_syndromes") or []
    syndrome_names = [
        str(item.get("syndrome_name") or "").strip()
        for item in syndromes
        if isinstance(item, dict) and str(item.get("syndrome_name") or "").strip()
    ]
    entities = slots.get("entities") or {}
    packet = {
        "task": task,
        "processed_text": {
            "raw_query": state.get("raw_query") or "",
            "query": state.get("query") or "",
            "spell_correction": {
                "explanation": spell.get("explanation") or "",
            },
        },
        "intent": {
            "text_intent": state.get("text_intent"),
            "confidence": state.get("text_intent_confidence"),
            "reason": state.get("text_intent_reason") or "",
            "relevance_reason": state.get("relevance_reason") or "",
        },
        "entities": entities,
        "slots": {
            "missing": slots.get("missing") or [],
            "filled": slots.get("filled"),
            "body_part": slots.get("body_part") or state.get("body_part"),
            "disease_name": slots.get("disease_name") or state.get("disease_name"),
            "symptoms": slots.get("symptoms") or [],
            "collected_symptoms": state.get("collected_symptoms") or [],
            "alignments": slots.get("alignments") or [],
        },
        "consult": {
            "major_round": state.get("major_round"),
            "session_mode": state.get("session_mode"),
            "body_part": state.get("body_part"),
            "disease_name": state.get("disease_name"),
            "body_part_options": state.get("body_part_options") or [],
            "disease_options": state.get("disease_options") or [],
            "symptom_options": state.get("symptom_options") or [],
            "candidate_syndromes": syndrome_names,
            "needs_clarification": state.get("needs_clarification"),
        },
        "recent_dialogue": _recent_dialogue(list(state.get("messages") or [])),
        "note": "recent_dialogue 为历史窗口；本轮用户话只在 processed_text",
    }
    return json.dumps(packet, ensure_ascii=False, indent=2)


def _phase_context(state: Mapping[str, Any], *, round_no: int) -> str:
    body = state.get("body_part") or "（尚未选定）"
    disease = state.get("disease_name") or "（尚未选定）"
    slots = state.get("slots") if isinstance(state.get("slots"), dict) else {}
    missing = "、".join(str(item) for item in (slots.get("missing") or []) if item) or "（无）"
    syndromes = [
        str(item.get("syndrome_name") or "").strip()
        for item in (state.get("candidate_syndromes") or [])
        if isinstance(item, dict) and str(item.get("syndrome_name") or "").strip()
    ]

    if round_no <= 1:
        return (
            f"已选部位：{body}\n"
            f"部位候选项：{_join(state.get('body_part_options'))}\n"
            f"缺槽：{missing}"
        )
    if round_no == 2:
        return (
            f"已选部位：{body}\n"
            f"已锁定疾病：{disease}\n"
            f"疾病候选项：{_join(state.get('disease_options'))}\n"
            f"缺槽：{missing}"
        )
    if round_no == 3:
        return (
            f"已选部位：{body}\n"
            f"已锁定疾病：{disease}\n"
            f"主症参考：{_join(state.get('symptom_options'))}\n"
            f"潜在证型：{_join(syndromes)}"
        )
    return (
        f"已选部位：{body}\n"
        f"已锁定疾病：{disease}\n"
        f"潜在证型：{_join(syndromes)}\n"
        f"辅助轮次：{state.get('auxiliary_round') or 0}\n"
        f"会话模式：{state.get('session_mode') or '（无）'}"
    )


def build_reply_agent_context(state: Mapping[str, Any], *, task: str) -> ReplyAgentRunContext:
    round_no = int(state.get("major_round") or 1)
    if round_no not in _OPERATION_RULES:
        round_no = 1 if round_no < 2 else 4
    return ReplyAgentRunContext(
        major_round=round_no,
        query=str(state.get("query") or state.get("raw_query") or "").strip(),
        task=task,
        body_part=state.get("body_part"),
        disease_name=state.get("disease_name"),
        recent_dialogue=_recent_dialogue(list(state.get("messages") or [])),
        phase_title=_PHASE_TITLES.get(round_no, _PHASE_TITLES[1]),
        phase_context=_phase_context(state, round_no=round_no),
        packet=build_reply_agent_packet(state, task=task),
    )


def render_reply_agent_system_prompt(ctx: ReplyAgentRunContext) -> str:
    template = _PROMPT_PATH.read_text(encoding="utf-8")
    return template.format(
        phase_title=ctx.phase_title,
        phase_context=ctx.phase_context,
        sub_agents=_SUB_AGENTS,
        operation_rules=_OPERATION_RULES[ctx.major_round],
        red_lines=_RED_LINES,
    )


def build_reply_agent_messages(ctx: ReplyAgentRunContext) -> list[BaseMessage]:
    content = (
        f"【近期对话】\n{ctx.recent_dialogue or '（无）'}\n\n"
        f"【本轮用户输入】\n{ctx.query or '（空）'}\n\n"
        f"【本节点任务】\n{ctx.task}\n\n"
        f"【上游结果】\n{ctx.packet}\n\n"
        "【候选项摘要已包含在上游结果 consult 字段中】"
    )
    return [HumanMessage(content=content)]
