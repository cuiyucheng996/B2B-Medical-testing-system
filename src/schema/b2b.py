"""B2B 辨证问诊字面量。"""

from __future__ import annotations

from typing import Literal

MajorRound = Literal[1, 2, 3, 4]
"""第一轮部位 → 第二轮疾病 → 第三轮主症 → 第四轮辅助症状。"""

SessionMode = Literal[
    "idle",
    "round1_body",
    "round2_disease",
    "round3_main_symptom",
    "round4_auxiliary",
    "resolved",
    "ambiguous",
]

UserInputMode = Literal["option", "text", "multi_option"]
InputHandler = Literal["option_input", "text_input", "vague_input"]
InputChannel = Literal["text", "asr"]
DEFAULT_INPUT_CHANNEL: InputChannel = "text"

OTHER_OPTION = "其他"
SYMPTOM_OTHER_OPTION = "以上都没有"
LOCALIZED_BODY_PARTS = frozenset(
    {"头面部", "颈部", "胸胁部", "胃腹部", "二阴部", "四肢躯干"}
)

NextAction = Literal["await_user", "continue", "end"]

TextIntent = Literal["complaint", "chitchat", "vague"]
"""纠错后句子意图：主诉 / 闲话 / 模糊。"""

MAX_AUXILIARY_ROUNDS = 2

__all__ = [
    "DEFAULT_INPUT_CHANNEL",
    "MAX_AUXILIARY_ROUNDS",
    "MajorRound",
    "NextAction",
    "OTHER_OPTION",
    "LOCALIZED_BODY_PARTS",
    "SYMPTOM_OTHER_OPTION",
    "TextIntent",
    "SessionMode",
    "UserInputMode",
    "InputHandler",
    "InputChannel",
]
