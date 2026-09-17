"""B2B 答复 Agent 结构化输出（单 Agent，不委派子 Agent）。"""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["DiseasePlainExplainItem", "DiseasePlainExplainOutput", "ReplyComposeOutput"]


class ReplyComposeOutput(BaseModel):
    """面向用户的人话答复。"""

    assistant_reply: str = Field(
        description="给用户的专业、亲和回复；可列出选项编号；不暴露内部字段名",
    )


class DiseasePlainExplainItem(BaseModel):
    disease_name: str = Field(description="必须与给定病名原字完全一致")
    plain_explain: str = Field(description="约 15 个汉字的白话说明，不确诊、不开药")


class DiseasePlainExplainOutput(BaseModel):
    """部位下全部疾病的白话短解释。"""

    items: list[DiseasePlainExplainItem] = Field(default_factory=list)
