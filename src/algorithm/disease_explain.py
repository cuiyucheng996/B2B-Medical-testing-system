"""第二轮：疾病槽未填时，用 Neo4j ConsultDisease.description 列在病名后。"""

from __future__ import annotations

from knowledge.neo4j_consult import list_diseases_for_part

__all__ = ["compose_disease_plain_list"]


def _clean(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def compose_disease_plain_list(
    *,
    body_part: str | None,
    body_part_id: int | None = None,
    disease_names: list[str] | None = None,
) -> str:
    rows = list_diseases_for_part(part_id=body_part_id, part_name=body_part)
    by_name = {
        str(row.get("disease_name") or "").strip(): _clean(row.get("description") or "")
        for row in rows
        if str(row.get("disease_name") or "").strip()
    }
    names = [str(item).strip() for item in (disease_names or []) if str(item).strip()]
    if not names:
        names = list(by_name.keys())
    if not names:
        return "当前部位下暂无疾病候选项，请用文字再描述一下不适。"

    part = body_part or "该部位"
    lines = [
        f"还没对上具体病名。下面是「{part}」常见几类（图谱说明），您对照选一项或再补充更贴近的感觉：",
        "",
    ]
    for idx, name in enumerate(names, start=1):
        desc = by_name.get(name) or ""
        lines.append(f"{idx}. {name} {desc}".rstrip())
    return "\n".join(lines)
