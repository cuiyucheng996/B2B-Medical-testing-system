"""用户输入解析。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.messages import HumanMessage

from schema.b2b import LOCALIZED_BODY_PARTS, OTHER_OPTION, SYMPTOM_OTHER_OPTION


def latest_human_text(messages: list[Any] | None) -> str:
    if not messages:
        return ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return str(msg.content or "").strip()
        role = getattr(msg, "type", None) or (msg.get("role") if isinstance(msg, dict) else None)
        if role in ("human", "user"):
            content = msg.content if hasattr(msg, "content") else msg.get("content", "")
            return str(content or "").strip()
    return ""


def normalize_selection(value: str | None) -> str | None:
    text = (value or "").strip()
    return text or None


def resolve_symptom_catalog_picks(items: list[str], catalog: list[str]) -> list[str]:
    """点选名或编号 → 候选项原名；有一条点中即算有效主症选择。"""
    names = [str(item).strip() for item in catalog if str(item).strip()]
    if not names:
        return []
    by_index = {str(idx): name for idx, name in enumerate(names, start=1)}
    picked: list[str] = []
    seen: set[str] = set()
    for raw in items:
        text = str(raw or "").strip()
        if not text or text in {SYMPTOM_OTHER_OPTION, OTHER_OPTION}:
            continue
        mapped = text if text in names else by_index.get(text)
        if not mapped and text.isdigit():
            mapped = by_index.get(str(int(text)))
        if mapped and mapped not in seen:
            seen.add(mapped)
            picked.append(mapped)
    return picked


def parse_symptom_selections(
    state: Mapping[str, Any],
) -> list[str]:
    selected = state.get("selected_symptoms") or []
    if isinstance(selected, str):
        selected = [selected]
    items = [str(item).strip() for item in selected if str(item).strip()]
    items = [item for item in items if item not in {SYMPTOM_OTHER_OPTION}]
    if items:
        return items

    raw = (state.get("raw_query") or state.get("query") or latest_human_text(state.get("messages")) or "").strip()
    if not raw:
        return []
    if raw in {SYMPTOM_OTHER_OPTION, OTHER_OPTION}:
        return []
    for sep in ("、", "，", ",", ";", "；"):
        raw = raw.replace(sep, "|")
    return [part.strip() for part in raw.split("|") if part.strip()]


def is_localized_body_part(value: str | None) -> bool:
    return (value or "").strip() in LOCALIZED_BODY_PARTS


def is_systemic_unlocalized_text(text: str | None) -> bool:
    """浑身/全身发冷这类，没有点明头面四肢等具体部位。"""
    compact = (text or "").replace(" ", "").replace("　", "").strip()
    if not compact:
        return False
    if any(part in compact for part in LOCALIZED_BODY_PARTS):
        return False
    local_hints = (
        "头", "脸", "颈", "脖子", "咽", "喉", "胸", "肋", "腹", "胃",
        "腰", "背", "腿", "脚", "手", "胳膊", "四肢", "二阴",
    )
    if any(hint in compact for hint in local_hints):
        return False
    markers = ("浑身", "全身", "满身", "通身", "周身", "到处")
    return any(mark in compact for mark in markers)


def is_valid_single_choice(value: str | None) -> bool:
    return bool(normalize_selection(value)) and normalize_selection(value) != OTHER_OPTION


def match_names_in_text(query: str, catalog: set[str] | list[str] | None) -> list[str]:
    """句中命中多个候选项原名，长名优先并消费已匹配片段。"""
    text = (query or "").replace(" ", "").replace("　", "").strip()
    names = [str(item).strip() for item in (catalog or []) if str(item).strip()]
    if not text or not names:
        return []
    names.sort(key=lambda item: len(item.replace(" ", "").replace("　", "")), reverse=True)
    remaining = text
    found: list[str] = []
    for name in names:
        key = name.replace(" ", "").replace("　", "")
        if len(key) < 2 or key not in remaining:
            continue
        found.append(name)
        remaining = remaining.replace(key, "\0" * len(key), 1)
    return list(dict.fromkeys(found))


def match_name_in_catalog(query: str, allowed: set[str] | None) -> str | None:
    """整句或句中唯一命中候选项标准名（如「水肿」「我水肿了」）。"""
    text = (query or "").replace(" ", "").replace("　", "").strip()
    if not text or not allowed:
        return None
    compact = {str(name).replace(" ", "").replace("　", "").strip(): str(name).strip() for name in allowed if str(name).strip()}
    if text in compact:
        return compact[text]
    hits = [name for key, name in compact.items() if key and key in text]
    if not hits:
        return None
    hits.sort(key=lambda n: len(n.replace(" ", "")), reverse=True)
    best = hits[0]
    best_key = best.replace(" ", "").replace("　", "")
    if any(n.replace(" ", "").replace("　", "") not in best_key for n in hits[1:]):
        return None
    return best
