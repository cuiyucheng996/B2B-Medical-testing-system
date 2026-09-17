"""否定标记抽取，并按与部位/疾病/症状 span 的位置判定是否入槽。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "NegationSpan",
    "build_negation_trace",
    "drop_negated_mentions",
    "extract_negation_spans",
    "is_mention_negated",
]

# 长的优先，避免「没有」被切成「没」
_NEG_MARKERS = (
    "并没有",
    "并不是",
    "没有",
    "并非",
    "并未",
    "并不",
    "不是",
    "没",
)

# 否定词 + 能力缺失，本身是症状，不当否认
_KEEP_PHRASES = (
    "没有力气",
    "没力气",
    "没有精神",
    "没精神",
    "没有胃口",
    "没胃口",
    "睡不着",
    "吃不下",
    "喝不下",
    "尿不出",
    "拉不出",
    "喘不上气",
    "喘不上",
    "使不上劲",
)

_DEGREE_PREFIXES = ("不是很", "不是太", "不是特别", "不太", "不怎么")

# 「不是说 A 是 B」：A 是被纠正的说法，B 才是主诉
_CORRECTION_OPENERS = ("并不是说", "我不是说", "不是说", "不是讲")

_CLAUSE_BREAKS = ("。", "！", "？", "；", ";", "，", ",", "、", "但是", "但", "就是", "可是", "不过")

_LOCAL_NEG_SUFFIX = ("不疼", "不痛", "不胀", "不咳", "不晕", "不痒", "不闷")


@dataclass(frozen=True)
class NegationSpan:
    text: str
    start: int
    end: int
    scope_end: int


def _next_clause_end(text: str, start: int) -> int:
    nearest = len(text)
    for brk in _CLAUSE_BREAKS:
        pos = text.find(brk, start)
        if pos >= 0:
            nearest = min(nearest, pos)
    return nearest


def _overlaps_keep_phrase(text: str, start: int, end: int) -> bool:
    for keep in _KEEP_PHRASES:
        pos = 0
        while True:
            found = text.find(keep, pos)
            if found < 0:
                break
            keep_end = found + len(keep)
            if not (end <= found or start >= keep_end):
                return True
            pos = found + 1
    return False


def extract_negation_spans(text: str) -> list[NegationSpan]:
    """从原句抽出否定标记，作用范围到下一个分句标点。"""
    raw = text or ""
    if not raw:
        return []
    occupied = [False] * len(raw)
    spans: list[NegationSpan] = []
    for marker in _NEG_MARKERS:
        start = 0
        while True:
            pos = raw.find(marker, start)
            if pos < 0:
                break
            end = pos + len(marker)
            start = pos + 1
            if any(occupied[pos:end]):
                continue
            window = raw[pos : pos + 8]
            if any(window.startswith(deg) for deg in _DEGREE_PREFIXES):
                continue
            if any(window.startswith(opener) for opener in _CORRECTION_OPENERS):
                continue
            if _overlaps_keep_phrase(raw, pos, end):
                continue
            for i in range(pos, end):
                occupied[i] = True
            spans.append(
                NegationSpan(
                    text=marker,
                    start=pos,
                    end=end,
                    scope_end=_next_clause_end(raw, end),
                )
            )
    spans.sort(key=lambda item: item.start)
    return spans


def _mention_positions(text: str, mention: str) -> list[tuple[int, int]]:
    found: list[tuple[int, int]] = []
    start = 0
    while True:
        pos = text.find(mention, start)
        if pos < 0:
            return found
        found.append((pos, pos + len(mention)))
        start = pos + 1


def correction_assert_range(text: str) -> tuple[int, int] | None:
    """不是说…是… 时，返回纠正后半句（是 之后）的起止。"""
    raw = text or ""
    for opener in _CORRECTION_OPENERS:
        pos = raw.find(opener)
        if pos < 0:
            continue
        shi = raw.find("是", pos + len(opener))
        if shi < 0:
            return None
        return shi + 1, len(raw)
    return None


def _occurrence_negated(raw: str, start: int, end: int, marks: list[NegationSpan]) -> bool:
    tail = raw[end : end + 4]
    if any(tail.startswith(suf) for suf in _LOCAL_NEG_SUFFIX):
        return True
    for mark in marks:
        if end == mark.start or (end < mark.start and not raw[end:mark.start].strip("的了还就")):
            if 0 <= mark.start - end <= 2:
                return True
        if mark.end <= start < mark.scope_end:
            return True
    return False


def is_mention_negated(text: str, mention: str, negations: list[NegationSpan] | None = None) -> bool:
    raw = text or ""
    span = (mention or "").strip()
    if not raw or not span or span not in raw:
        return False
    for keep in _KEEP_PHRASES:
        if keep in raw and (span in keep or keep in span):
            return False
    marks = negations if negations is not None else extract_negation_spans(raw)
    positions = _mention_positions(raw, span)
    asserted = correction_assert_range(raw)
    if asserted:
        a0, a1 = asserted
        in_assert = [pos for pos in positions if pos[0] >= a0 and pos[1] <= a1]
        if not in_assert:
            return True
        return all(_occurrence_negated(raw, start, end, marks) for start, end in in_assert)
    return all(_occurrence_negated(raw, start, end, marks) for start, end in positions)


def drop_negated_mentions(text: str, mentions: list[str]) -> list[str]:
    """保留未被否定罩住的部位/疾病/症状 span。"""
    marks = extract_negation_spans(text)
    kept: list[str] = []
    for mention in mentions:
        item = str(mention or "").strip()
        if not item:
            continue
        if is_mention_negated(text, item, marks):
            continue
        if item not in kept:
            kept.append(item)
    return kept


def build_negation_trace(text: str, extracted: dict[str, list[str]]) -> dict[str, Any]:
    """抽槽后先问 qwen 每个 span 是否否定；失败再用规则。否定则不入槽。"""
    marks = extract_negation_spans(text)
    asserted = correction_assert_range(text)
    from algorithm.qwen_consult_fallback import judge_extracted_negation

    judged = judge_extracted_negation(text, extracted)
    judged_map: dict[tuple[str, str], dict[str, Any]] = {}
    source = "rule"
    qwen_items: list[dict[str, Any]] = []
    qwen_model = ""
    if judged:
        from configs.b2b_api import get_b2b_api_config

        source = "qwen-turbo"
        qwen_model = str(get_b2b_api_config().get("qwen_fallback_model") or "qwen-turbo")
        qwen_items = list(judged)
        judged_map = {(str(row.get("type")), str(row.get("span"))): row for row in judged}

    dropped: list[dict[str, Any]] = []
    kept: dict[str, list[str]] = {}
    for key, mentions in (extracted or {}).items():
        kept_items: list[str] = []
        for mention in mentions or []:
            item = str(mention or "").strip()
            if not item:
                continue
            row = judged_map.get((key, item))
            if row is not None:
                negated = bool(row.get("negated"))
                reason = str(row.get("reason") or "qwen 判定")
                judge = "qwen-turbo"
            else:
                negated = is_mention_negated(text, item, marks)
                reason = "规则：否定谓词罩住或不在纠偏后半句" if negated else "规则：未否定"
                judge = "rule"
            if negated:
                dropped.append(
                    {
                        "type": key,
                        "span": item,
                        "入槽": False,
                        "judge": judge,
                        "reason": reason,
                    }
                )
                continue
            if item not in kept_items:
                kept_items.append(item)
        kept[key] = kept_items
    return {
        "markers": [
            {"text": mark.text, "start": mark.start, "end": mark.end, "scope_end": mark.scope_end}
            for mark in marks
        ],
        "dropped": dropped,
        "kept": kept,
        "judge": source,
        "qwen_model": qwen_model,
        "qwen_items": qwen_items,
        "入槽策略": "抽槽后由 qwen 判断是否否定，否定不入槽；qwen 失败回退规则",
        "纠偏": None
        if asserted is None
        else {"assert_start": asserted[0], "asserted": (text or "")[asserted[0] :]},
    }
