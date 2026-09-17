"""关键词意图识别：命中返回对应意图，未命中返回 None（规则优先于 BERT）。"""

from __future__ import annotations

from schema.b2b import TextIntent

__all__ = ["INTENT_RULES", "match_intent"]

# 含症状时优先 complaint，避免「您好医生我头疼」被寒暄抢走
INTENT_RULES: list[tuple[TextIntent, tuple[str, ...]]] = [
    (
        "complaint",
        (
            "痛", "疼", "闷", "胀", "咳", "热", "乏", "晕", "吐", "泻", "秘",
            "失眠", "心悸", "恶心", "呕吐", "腹泻", "便秘", "头痛", "胃痛",
            "腰痛", "胸闷", "腹胀", "腰酸", "口干", "发热",
        ),
    ),
    (
        "chitchat",
        (
            "你好", "您好", "在吗", "谢谢", "再见", "拜拜", "早上好",
            "辛苦了", "打扰", "挂号", "医保", "几点上班", "天气", "哈哈",
            "出去玩", "出去耍", "出去浪", "去玩", "想玩", "逛街", "旅游",
            "看电影", "吃饭去",
        ),
    ),
    (
        "vague",
        ("不舒服", "难受", "说不清", "不知道", "随便", "还行", "有点问题", "说不上来"),
    ),
]


def match_intent(question: str) -> TextIntent | None:
    text = (question or "").replace(" ", "").strip()
    if not text:
        return None
    for intent, keywords in INTENT_RULES:
        if any(keyword in text for keyword in keywords):
            return intent
    return None
