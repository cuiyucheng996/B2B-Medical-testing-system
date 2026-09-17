"""B2B 槽位 schema：对齐老师 config.SCHEMA + 按意图 set_schema。"""

from __future__ import annotations

from schema.b2b import MajorRound, TextIntent

__all__ = [
    "INTENT_SLOT_SCHEMA",
    "REQUIRED_SLOTS",
    "UIE_SCHEMA",
    "required_slots_for",
    "schema_for_intent",
]

# 全局可抽实体类型（对应老师 config.SCHEMA；问诊侧与 Doccano span_labels 一致）
UIE_SCHEMA: list[str] = ["部位", "疾病", "症状"]

# 意图 + 轮次 → 本次 UIE set_schema 列表（对应老师 match intent 后 schema = ["商品"]）
INTENT_SLOT_SCHEMA: dict[tuple[str, int], list[str]] = {
    ("complaint", 1): ["部位", "症状"],
    ("complaint", 2): ["疾病", "症状"],
    ("complaint", 3): ["症状"],
    ("complaint", 4): ["症状"],
}

# 进图谱前必须齐的槽；未列出的抽取项仅作补充
REQUIRED_SLOTS: dict[tuple[str, int], list[str]] = {
    ("complaint", 1): ["部位"],
    ("complaint", 2): ["疾病"],
    ("complaint", 3): ["症状"],
    ("complaint", 4): ["症状"],
}


def schema_for_intent(
    intent: TextIntent | str | None,
    *,
    major_round: MajorRound | int = 1,
) -> list[str]:
    if intent != "complaint":
        return []
    round_no = int(major_round or 1)
    return list(INTENT_SLOT_SCHEMA.get(("complaint", round_no), ["部位"]))


def required_slots_for(
    intent: TextIntent | str | None,
    *,
    major_round: MajorRound | int = 1,
) -> list[str]:
    if intent != "complaint":
        return []
    round_no = int(major_round or 1)
    return list(REQUIRED_SLOTS.get(("complaint", round_no), ["部位"]))
