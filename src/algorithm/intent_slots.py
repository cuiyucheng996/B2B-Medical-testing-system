"""槽位填充：UIE 抽出口语实体 → entity_align 落到 Neo4j 入口节点。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from algorithm.input_parse import is_systemic_unlocalized_text, match_name_in_catalog
from algorithm.entity_align import AlignHit, entity_align
from algorithm.negation import build_negation_trace
from algorithm.uie_extract import extract_entity
from knowledge.neo4j_consult import (
    get_body_part_by_id,
    get_disease_by_id,
    list_body_parts,
    list_diseases_for_part,
)
from schema.b2b import MajorRound, TextIntent
from schema.entity_synonyms import body_from_symptom, collect_synonym_spans
from schema.slot_schema import required_slots_for, schema_for_intent

__all__ = [
    "ConsultSlots",
    "fill_consult_slots",
    "slots_have_mentions",
]


def slots_have_mentions(slots: dict[str, Any] | None, keys: tuple[str, ...] = ("部位", "疾病", "症状")) -> bool:
    """UIE 是否抽到过可对齐的 span；没有则不应再拿整句打 Milvus。"""
    entities = (slots or {}).get("entities") or {}
    mentions = (slots or {}).get("mentions") or {}
    for key in keys:
        for item in list(entities.get(key) or []) + list(mentions.get(key) or []):
            if str(item).strip():
                return True
    return False


def _hit_dict(hit: AlignHit | None) -> dict[str, Any] | None:
    if hit is None:
        return None
    return {
        "mention": hit.mention,
        "standard_name": hit.standard_name,
        "node_id": hit.node_id,
        "node_label": hit.node_label,
        "method": hit.method,
        "score": hit.score,
        "needs_review": hit.needs_review,
    }


@dataclass
class ConsultSlots:
    schema: list[str] = field(default_factory=list)
    entities: dict[str, list[str]] = field(default_factory=dict)
    mentions: dict[str, list[str]] = field(default_factory=dict)
    body_part: str | None = None
    body_part_id: int | None = None
    disease_name: str | None = None
    disease_id: int | None = None
    symptoms: list[str] = field(default_factory=list)
    entry_nodes: dict[str, Any] = field(default_factory=dict)
    alignments: list[dict[str, Any]] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    extracted: dict[str, list[str]] = field(default_factory=dict)
    negation: dict[str, Any] = field(default_factory=dict)
    fill_source: str = "uie"
    fill_model: str = ""
    fill_reason: str = ""

    @property
    def filled(self) -> bool:
        return not self.missing

    def to_state(self) -> dict[str, Any]:
        return {
            "slots": {
                "schema": list(self.schema),
                "extracted": {k: list(v) for k, v in self.extracted.items()},
                "entities": {k: list(v) for k, v in self.entities.items()},
                "mentions": {k: list(v) for k, v in self.mentions.items()},
                "body_part": self.body_part,
                "body_part_id": self.body_part_id,
                "disease_name": self.disease_name,
                "disease_id": self.disease_id,
                "symptoms": list(self.symptoms),
                "entry_nodes": dict(self.entry_nodes),
                "alignments": list(self.alignments),
                "missing": list(self.missing),
                "filled": self.filled,
                "negation": dict(self.negation),
                "fill_source": self.fill_source,
                "fill_model": self.fill_model,
                "fill_reason": self.fill_reason,
            }
        }


def _trusted_part_mentions(part_mentions: list[str], symptom_mentions: list[str]) -> list[str]:
    """UIE 常把「脑袋疼」拆成部位「脑袋」+ 症状「脑袋疼」，部位子串不可信。"""
    trusted: list[str] = []
    for part in part_mentions:
        text = (part or "").strip()
        if not text:
            continue
        if any(text in symptom and text != symptom for symptom in symptom_mentions):
            continue
        trusted.append(text)
    return trusted


def _align_first(
    mentions: list[str],
    *,
    entity_type: str,
    catalog: list[dict[str, Any]],
    name_key: str,
    node_label: str,
    utterance: str | None = None,
) -> AlignHit | None:
    best: AlignHit | None = None
    for mention in mentions:
        hit = entity_align(  # type: ignore[arg-type]
            entity_type,
            mention,
            catalog,
            name_key=name_key,
            node_label=node_label,
            utterance=utterance,
        )
        if best is None or (hit.ok and not best.ok) or hit.score > best.score:
            best = hit
        if hit.ok:
            return hit
    return best


def _merge_unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(str(item).strip() for item in items if str(item).strip()))


def _seed_mentions_from_lexicon(query: str, entities: dict[str, list[str]], *, round_no: int) -> dict[str, list[str]]:
    """UIE 空抽或漏抽时，用同义词表从整句补 span。第一轮必须能补到部位。"""
    seeded = {key: list(value) for key, value in (entities or {}).items()}
    if round_no <= 2:
        parts = [alias for alias, _ in collect_synonym_spans("body_part", query)]
        if parts:
            seeded["部位"] = _merge_unique(list(seeded.get("部位") or []) + parts)
    symptoms = [alias for alias, _ in collect_synonym_spans("symptom", query)]
    if symptoms:
        seeded["症状"] = _merge_unique(list(seeded.get("症状") or []) + symptoms)
    if round_no == 2:
        diseases = [alias for alias, _ in collect_synonym_spans("disease", query)]
        if diseases:
            seeded["疾病"] = _merge_unique(list(seeded.get("疾病") or []) + diseases)
    return seeded


def fill_consult_slots(
    text: str = "",
    *,
    intent: TextIntent | str | None = "complaint",
    major_round: MajorRound | int = 1,
    body_part: str | None = None,
    body_part_id: int | None = None,
    symptom_catalog: list[str] | None = None,
    seed_entities: dict[str, list[str]] | None = None,
    fill_reason: str | None = None,
) -> ConsultSlots:
    query = (text or "").strip()
    schema = schema_for_intent(intent, major_round=major_round)
    required = required_slots_for(intent, major_round=major_round)
    slots = ConsultSlots(schema=schema)
    round_no = int(major_round or 1)
    from configs.b2b_api import get_b2b_api_config

    qwen_model = str(get_b2b_api_config().get("qwen_fallback_model") or "qwen-turbo")
    if seed_entities:
        slots.fill_source = "qwen-turbo"
        slots.fill_model = qwen_model
        slots.fill_reason = fill_reason or f"qwen-turbo 降级：用模型抽出的 span 填槽"
    else:
        slots.fill_source = "uie"
        slots.fill_model = ""
        slots.fill_reason = fill_reason or "UIE 抽槽"

    if intent != "complaint":
        return slots

    extract_labels = list(schema) or (["部位", "症状"] if round_no <= 1 else ["症状"])
    entities = extract_entity(query, extract_labels)
    if seed_entities:
        for key in ("部位", "疾病", "症状"):
            extra = [str(item).strip() for item in (seed_entities.get(key) or []) if str(item).strip()]
            if extra:
                entities[key] = extra
    if round_no <= 1:
        entities = {key: list(value) for key, value in entities.items() if key != "疾病"}
    elif round_no >= 3:
        entities = {key: list(value) for key, value in entities.items() if key == "症状"}
    entities = _seed_mentions_from_lexicon(query, entities, round_no=round_no)
    extracted = {
        key: _merge_unique(list(entities.get(key) or []))
        for key in ("部位", "疾病", "症状")
    }
    slots.extracted = extracted
    slots.negation = build_negation_trace(query, extracted)
    entities = {
        key: list(slots.negation.get("kept", {}).get(key) or [])
        for key in ("部位", "疾病", "症状")
    }
    slots.entities = entities
    slots.mentions = {k: list(v) for k, v in entities.items()}

    try:
        parts = list_body_parts()
    except Exception:
        parts = []

    symptom_mentions = list(entities.get("症状") or [])
    part_mentions = _trusted_part_mentions(
        list(entities.get("部位") or []),
        symptom_mentions,
    )
    part_hit = _align_first(
        part_mentions,
        entity_type="body_part",
        catalog=parts,
        name_key="part_name",
        node_label="ConsultBodyPart",
        utterance=query,
    )
    if round_no < 3 and part_hit:
        slots.alignments.append(_hit_dict(part_hit) or {})

    skip_body_guess = is_systemic_unlocalized_text(query)
    if round_no < 3 and part_hit and part_hit.ok and not skip_body_guess:
        try:
            confirmed = get_body_part_by_id(part_hit.node_id) or {
                "id": part_hit.node_id,
                "part_name": part_hit.standard_name,
            }
        except Exception:
            confirmed = {"id": part_hit.node_id, "part_name": part_hit.standard_name}
        slots.body_part = str(confirmed.get("part_name") or part_hit.standard_name)
        slots.body_part_id = confirmed.get("id")
        slots.entry_nodes["body_part"] = {
            "label": "ConsultBodyPart",
            "id": slots.body_part_id,
            "name": slots.body_part,
        }
    elif body_part:
        slots.body_part = body_part
        slots.body_part_id = body_part_id
        slots.entry_nodes["body_part"] = {
            "label": "ConsultBodyPart",
            "id": body_part_id,
            "name": body_part,
        }

    disease_mentions: list[str] = []
    disease_hit = None
    if round_no == 2:
        diseases: list[dict[str, Any]] = []
        try:
            diseases = list_diseases_for_part(
                part_id=slots.body_part_id or body_part_id,
                part_name=slots.body_part or body_part,
            )
        except Exception:
            diseases = []

        disease_mentions = list(entities.get("疾病") or [])
        catalog_names = {str(row.get("disease_name") or "").strip() for row in diseases if row.get("disease_name")}
        catalog_hit = match_name_in_catalog(query, catalog_names)
        if catalog_hit and catalog_hit not in disease_mentions:
            disease_mentions = [catalog_hit, *disease_mentions]
        symptom_set = {str(item).strip() for item in symptom_mentions if str(item).strip()}
        disease_mentions = [
            item
            for item in disease_mentions
            if str(item).strip() in catalog_names or str(item).strip() not in symptom_set
        ]
        disease_hit = _align_first(
            disease_mentions,
            entity_type="disease",
            catalog=diseases,
            name_key="disease_name",
            node_label="ConsultDisease",
            utterance=query,
        )
        if disease_hit:
            slots.alignments.append(_hit_dict(disease_hit) or {})
        if disease_hit and disease_hit.ok:
            try:
                confirmed = get_disease_by_id(disease_hit.node_id) or {
                    "id": disease_hit.node_id,
                    "disease_name": disease_hit.standard_name,
                }
            except Exception:
                confirmed = {"id": disease_hit.node_id, "disease_name": disease_hit.standard_name}
            slots.disease_name = str(confirmed.get("disease_name") or disease_hit.standard_name)
            slots.disease_id = confirmed.get("id")
            slots.entry_nodes["disease"] = {
                "label": "ConsultDisease",
                "id": slots.disease_id,
                "name": slots.disease_name,
            }

    # 症状槽：UIE span 原样入槽，再逐条 同义词 → 标准名 → Milvus 粗召回 + DeepSeek 精排
    symptom_spans = list(symptom_mentions)
    if round_no >= 3:
        locked = str(disease_hit.mention or "").strip() if disease_hit and disease_hit.ok else ""
        for mention in disease_mentions:
            text_m = str(mention).strip()
            if text_m and text_m != locked:
                symptom_spans.append(text_m)
    symptom_spans = list(dict.fromkeys(s for s in symptom_spans if s))
    slots.symptoms = list(symptom_spans)

    symptom_rows = [{"id": None, "symptom_name": name} for name in (symptom_catalog or [])]
    if symptom_rows:
        for mention in symptom_spans:
            hit = entity_align(
                "symptom",
                mention,
                symptom_rows,
                name_key="symptom_name",
                node_label="Symptom",
                for_span=True,
                utterance=query,
            )
            slots.alignments.append(_hit_dict(hit) or {})
            if hit.ok and hit.standard_name:
                slots.symptoms.append(hit.standard_name)
        slots.symptoms = list(dict.fromkeys(s for s in slots.symptoms if s))

    if not slots.body_part and not is_systemic_unlocalized_text(query):
        for mention in symptom_mentions + part_mentions:
            guessed = body_from_symptom(mention)
            if not guessed:
                continue
            hit = entity_align(
                "body_part",
                guessed,
                parts,
                name_key="part_name",
                node_label="ConsultBodyPart",
                utterance=query,
            )
            slots.alignments.append(_hit_dict(hit) or {})
            if hit.ok:
                try:
                    confirmed = get_body_part_by_id(hit.node_id) or {
                        "id": hit.node_id,
                        "part_name": hit.standard_name,
                    }
                except Exception:
                    confirmed = {"id": hit.node_id, "part_name": hit.standard_name}
                slots.body_part = str(confirmed.get("part_name") or hit.standard_name)
                slots.body_part_id = confirmed.get("id")
                slots.entry_nodes["body_part"] = {
                    "label": "ConsultBodyPart",
                    "id": slots.body_part_id,
                    "name": slots.body_part,
                }
                break

    if skip_body_guess:
        slots.body_part = None
        slots.body_part_id = None
        slots.entry_nodes.pop("body_part", None)

    if "部位" in required and not slots.entry_nodes.get("body_part"):
        slots.missing.append("部位")
    if "疾病" in required and not slots.entry_nodes.get("disease"):
        slots.missing.append("疾病")
    if "症状" in required and not (slots.symptoms or symptom_mentions or (round_no >= 3 and disease_mentions)):
        slots.missing.append("症状")
    return slots


def lookup_or_self(mention: str) -> str:
    from schema.entity_synonyms import lookup_synonym

    return lookup_synonym("symptom", mention) or mention
