"""从主项目 Neo4j 问诊知识图读取部位/疾病/证型/症状。"""

from __future__ import annotations

import json
from typing import Any

from configs.paths import ensure_uvgraph_root

ensure_uvgraph_root()

from src.RAGgraph.driver.neo4j_driver import get_neo4j_driver
from src.configs import NEO4J_CONFIG


def _driver():
    return get_neo4j_driver()


def _db() -> str:
    return str(NEO4J_CONFIG.get("database") or "neo4j")


def _parse_json_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [text]
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    return []


def list_body_parts() -> list[dict[str, Any]]:
    result = _driver().execute_query(
        """
        MATCH (p:ConsultBodyPart)
        WHERE coalesce(p.status, 1) = 1
        RETURN p.id AS id, p.part_code AS part_code, p.part_name AS part_name
        ORDER BY coalesce(p.sort_order, 0), p.part_name
        """,
        database_=_db(),
    )
    return [row.data() for row in result.records]


def get_body_part_by_id(part_id: int | None) -> dict[str, Any] | None:
    if part_id is None:
        return None
    result = _driver().execute_query(
        """
        MATCH (p:ConsultBodyPart {id: $part_id})
        WHERE coalesce(p.status, 1) = 1
        RETURN p.id AS id, p.part_code AS part_code, p.part_name AS part_name
        """,
        parameters_={"part_id": part_id},
        database_=_db(),
    )
    return result.records[0].data() if result.records else None


def get_disease_by_id(disease_id: int | None) -> dict[str, Any] | None:
    if disease_id is None:
        return None
    result = _driver().execute_query(
        """
        MATCH (d:ConsultDisease {id: $disease_id})
        WHERE coalesce(d.status, 1) = 1
        RETURN d.id AS id, d.disease_code AS disease_code, d.disease_name AS disease_name
        """,
        parameters_={"disease_id": disease_id},
        database_=_db(),
    )
    return result.records[0].data() if result.records else None


def get_disease_by_name(disease_name: str | None) -> dict[str, Any] | None:
    name = (disease_name or "").strip()
    if not name:
        return None
    result = _driver().execute_query(
        """
        MATCH (d:ConsultDisease)
        WHERE d.disease_name = $name AND coalesce(d.status, 1) = 1
        OPTIONAL MATCH (d)-[:BELONG]->(p:ConsultBodyPart)
        RETURN d.id AS id, d.disease_code AS disease_code, d.disease_name AS disease_name,
               p.id AS body_part_id, p.part_name AS body_part
        LIMIT 1
        """,
        parameters_={"name": name},
        database_=_db(),
    )
    return result.records[0].data() if result.records else None


def list_selectable_disease_names() -> list[str]:
    result = _driver().execute_query(
        """
        MATCH (d:ConsultDisease)
        WHERE coalesce(d.status, 1) = 1 AND coalesce(d.selectable, 1) = 1
        RETURN d.disease_name AS disease_name
        ORDER BY d.disease_name
        """,
        database_=_db(),
    )
    return [str(row["disease_name"]).strip() for row in result.records if row.get("disease_name")]


def list_diseases_for_part(*, part_id: int | None = None, part_name: str | None = None) -> list[dict[str, Any]]:
    if part_id is not None:
        result = _driver().execute_query(
            """
            MATCH (d:ConsultDisease)-[:BELONG]->(p:ConsultBodyPart {id: $part_id})
            WHERE coalesce(d.status, 1) = 1 AND coalesce(d.selectable, 1) = 1
            RETURN d.id AS id, d.disease_code AS disease_code, d.disease_name AS disease_name,
                   coalesce(d.description, '') AS description
            ORDER BY coalesce(d.sort_order, 0), d.disease_name
            """,
            parameters_={"part_id": part_id},
            database_=_db(),
        )
    elif part_name:
        result = _driver().execute_query(
            """
            MATCH (d:ConsultDisease)-[:BELONG]->(p:ConsultBodyPart)
            WHERE p.part_name = $part_name
              AND coalesce(d.status, 1) = 1 AND coalesce(d.selectable, 1) = 1
            RETURN d.id AS id, d.disease_code AS disease_code, d.disease_name AS disease_name,
                   coalesce(d.description, '') AS description
            ORDER BY coalesce(d.sort_order, 0), d.disease_name
            """,
            parameters_={"part_name": part_name},
            database_=_db(),
        )
    else:
        return []
    return [row.data() for row in result.records]


def list_syndromes_for_disease(*, disease_id: int | None = None, disease_name: str | None = None) -> list[dict[str, Any]]:
    if disease_id is not None:
        query = """
            MATCH (st:SyndromeType)-[:BELONG]->(d:ConsultDisease {id: $disease_id})
            WHERE coalesce(st.status, 1) = 1
            RETURN st.id AS id, st.syndrome_name AS syndrome_name, st.disease_name AS disease_name,
                   st.main_symptoms AS main_symptoms, st.supplementary_symptoms AS supplementary_symptoms
            ORDER BY st.syndrome_name
        """
        params = {"disease_id": disease_id}
    elif disease_name:
        query = """
            MATCH (st:SyndromeType)-[:BELONG]->(d:ConsultDisease)
            WHERE d.disease_name = $disease_name AND coalesce(st.status, 1) = 1
            RETURN st.id AS id, st.syndrome_name AS syndrome_name, st.disease_name AS disease_name,
                   st.main_symptoms AS main_symptoms, st.supplementary_symptoms AS supplementary_symptoms
            ORDER BY st.syndrome_name
        """
        params = {"disease_name": disease_name}
    else:
        return []

    rows = _driver().execute_query(query, parameters_=params, database_=_db()).records
    syndromes: list[dict[str, Any]] = []
    for row in rows:
        data = row.data()
        syndrome_id = data["id"]
        main_symptoms = _parse_json_list(data.get("main_symptoms"))
        supplementary = _parse_json_list(data.get("supplementary_symptoms"))
        dialectic = list_auxiliary_symptoms_for_syndrome(syndrome_id)
        merged_aux = list(dict.fromkeys([*supplementary, *dialectic]))
        syndromes.append(
            {
                "id": syndrome_id,
                "syndrome_name": data.get("syndrome_name") or "",
                "disease_name": data.get("disease_name") or "",
                "main_symptoms": main_symptoms,
                "supplementary_symptoms": merged_aux,
            }
        )
    return syndromes


def list_all_syndrome_main_symptoms() -> list[dict[str, Any]]:
    """全部证型主症：标准名 → 所属疾病（供第三轮全局向量检索）。"""
    result = _driver().execute_query(
        """
        MATCH (st:SyndromeType)-[:BELONG]->(d:ConsultDisease)
        WHERE coalesce(st.status, 1) = 1 AND coalesce(d.status, 1) = 1
        RETURN coalesce(d.disease_name, st.disease_name) AS disease_name,
               d.id AS disease_id,
               st.id AS syndrome_id, st.syndrome_name AS syndrome_name,
               st.main_symptoms AS main_symptoms
        """,
        database_=_db(),
    )
    rows: list[dict[str, Any]] = []
    for record in result.records:
        data = record.data()
        disease = str(data.get("disease_name") or "").strip()
        if not disease:
            continue
        for name in _parse_json_list(data.get("main_symptoms")):
            rows.append(
                {
                    "symptom_name": name,
                    "disease_id": data.get("disease_id"),
                    "disease_name": disease,
                    "syndrome_id": data.get("syndrome_id"),
                    "syndrome_name": data.get("syndrome_name") or "",
                }
            )
    return rows


def _expand_dialectic_symptom_text(text: str) -> list[str]:
    """打分表 symptom_text 可能是「A|B|C」或顿号分隔。"""
    raw = (text or "").replace("|", "、").replace("，", "、").replace(",", "、")
    return [part.strip() for part in raw.split("、") if part.strip()]


def list_dialectic_symptoms_for_syndromes(syndrome_ids: list[int]) -> list[str]:
    """剩余候选证型在 SyndromeDialecticScore 中的标准辅症名（去重）。"""
    if not syndrome_ids:
        return []
    result = _driver().execute_query(
        """
        MATCH (sc:SyndromeDialecticScore)-[:BELONG]->(st:SyndromeType)
        WHERE st.id IN $syndrome_ids AND coalesce(sc.status, 1) = 1
        RETURN DISTINCT sc.symptom_text AS symptom_text
        ORDER BY symptom_text
        """,
        parameters_={"syndrome_ids": syndrome_ids},
        database_=_db(),
    )
    seen: set[str] = set()
    items: list[str] = []
    for row in result.records:
        for name in _expand_dialectic_symptom_text(str(row.get("symptom_text") or "")):
            if name not in seen:
                seen.add(name)
                items.append(name)
    return items


def list_auxiliary_symptoms_for_syndrome(syndrome_id: int) -> list[str]:
    return list_dialectic_symptoms_for_syndromes([syndrome_id])


def score_syndromes_by_auxiliary(
  syndrome_ids: list[int],
  selected_symptoms: list[str],
) -> dict[int, int]:
    if not syndrome_ids or not selected_symptoms:
        return {}
    picked = {str(item).strip() for item in selected_symptoms if str(item).strip()}
    result = _driver().execute_query(
        """
        MATCH (sc:SyndromeDialecticScore)-[:BELONG]->(st:SyndromeType)
        WHERE st.id IN $syndrome_ids AND coalesce(sc.status, 1) = 1
        RETURN st.id AS syndrome_id, sc.symptom_text AS symptom_text, sc.score AS score
        """,
        parameters_={"syndrome_ids": syndrome_ids},
        database_=_db(),
    )
    totals: dict[int, int] = {}
    for row in result.records:
        sid = int(row["syndrome_id"])
        expanded = set(_expand_dialectic_symptom_text(str(row.get("symptom_text") or "")))
        if not (picked & expanded):
            continue
        totals[sid] = totals.get(sid, 0) + int(row.get("score") or 0)
    return totals


def filter_syndromes_by_main_symptoms_graph(
    syndromes: list[dict[str, Any]],
    selected_symptoms: list[str],
) -> list[dict[str, Any]]:
    """Cypher 加载证型主症，刷掉与用户症状无交集的候选，保留命中数最多者。"""
    syndrome_ids = [int(item["id"]) for item in syndromes if item.get("id") is not None]
    picked = {str(item).strip() for item in selected_symptoms if str(item).strip()}
    if not syndrome_ids or not picked:
        return []

    result = _driver().execute_query(
        """
        MATCH (st:SyndromeType)
        WHERE st.id IN $syndrome_ids AND coalesce(st.status, 1) = 1
        RETURN st.id AS id, st.syndrome_name AS syndrome_name, st.disease_name AS disease_name,
               st.main_symptoms AS main_symptoms, st.supplementary_symptoms AS supplementary_symptoms
        """,
        parameters_={"syndrome_ids": syndrome_ids},
        database_=_db(),
    )
    lookup = {int(item["id"]): item for item in syndromes if item.get("id") is not None}
    matched: list[tuple[int, dict[str, Any]]] = []
    for row in result.records:
        data = row.data()
        sid = int(data["id"])
        main_set = {str(name).strip() for name in _parse_json_list(data.get("main_symptoms"))}
        overlap = len(picked & main_set)
        if overlap <= 0:
            continue
        base = dict(lookup.get(sid, data))
        if data.get("main_symptoms") is not None:
            base["main_symptoms"] = _parse_json_list(data.get("main_symptoms"))
        if data.get("supplementary_symptoms") is not None:
            supplementary = _parse_json_list(data.get("supplementary_symptoms"))
            dialectic = list_auxiliary_symptoms_for_syndrome(sid)
            base["supplementary_symptoms"] = list(dict.fromkeys([*supplementary, *dialectic]))
        matched.append((overlap, base))

    if not matched:
        return []
    max_overlap = max(score for score, _ in matched)
    return [item for score, item in matched if score == max_overlap]


def filter_syndromes_by_auxiliary_symptoms_graph(
    syndromes: list[dict[str, Any]],
    selected_symptoms: list[str],
) -> list[dict[str, Any]]:
    """仅在剩余证型上查 SyndromeDialecticScore，按命中辅症数 + score 刷证型。"""
    syndrome_ids = [int(item["id"]) for item in syndromes if item.get("id") is not None]
    picked = {str(item).strip() for item in selected_symptoms if str(item).strip()}
    if not syndrome_ids or not picked:
        return []

    result = _driver().execute_query(
        """
        MATCH (sc:SyndromeDialecticScore)-[:BELONG]->(st:SyndromeType)
        WHERE st.id IN $syndrome_ids
          AND coalesce(sc.status, 1) = 1
          AND coalesce(st.status, 1) = 1
        RETURN st.id AS id, sc.symptom_text AS symptom_text, sc.score AS score
        """,
        parameters_={"syndrome_ids": syndrome_ids},
        database_=_db(),
    )
    lookup = {int(item["id"]): item for item in syndromes if item.get("id") is not None}
    hit_counts: dict[int, int] = {}
    score_sums: dict[int, int] = {}
    for row in result.records:
        sid = int(row["id"])
        expanded = set(_expand_dialectic_symptom_text(str(row.get("symptom_text") or "")))
        overlap = picked & expanded
        if not overlap:
            continue
        hit_counts[sid] = hit_counts.get(sid, 0) + len(overlap)
        score_sums[sid] = score_sums.get(sid, 0) + int(row.get("score") or 0)

    if not hit_counts:
        return []

    matched: list[tuple[int, dict[str, Any]]] = []
    for sid, hits in hit_counts.items():
        total = hits * 10 + score_sums.get(sid, 0)
        base = lookup.get(sid)
        if base is not None:
            matched.append((total, base))

    if not matched:
        return []
    max_score = max(score for score, _ in matched)
    return [item for score, item in matched if score == max_score]
