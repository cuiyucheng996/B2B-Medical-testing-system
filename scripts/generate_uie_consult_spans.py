"""生成 B2B 问诊口语 UIE 预标注：标俗称 span，不拿标准病名硬塞模板。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

B2B_ROOT = Path(__file__).resolve().parents[1]
B2B_SRC = B2B_ROOT / "src"
if str(B2B_SRC) not in sys.path:
    sys.path.insert(0, str(B2B_SRC))

from schema.entity_synonyms import SYNONYM_TABLE  # noqa: E402

OUT_TXT = B2B_ROOT / "data" / "uie" / "doccano" / "samples" / "consult_colloquial.txt"
OUT_JSONL = B2B_ROOT / "data" / "uie" / "doccano" / "export" / "consult_colloquial.jsonl"

# 完整口语句 + 要圈的片段（圈用户原话，不圈「咳血」「中风」这种书面名，除非话里真说了）
SPOKEN: list[tuple[str, list[tuple[str, str]]]] = [
    ("昨天都咳出点血出来 吓死我了", [("咳出点血", "疾病")]),
    ("咳嗽的时候痰里有血丝", [("痰里有血", "疾病")]),
    ("早上咳了两口，里面带点血", [("带点血", "疾病")]),
    ("老流鼻血，纸巾一擦全是红的", [("流鼻血", "疾病")]),
    ("刷牙一出血，吐出来一池子红", [("刷牙出血", "疾病")]),
    ("突然嘴歪了，说话大舌头", [("嘴歪", "疾病"), ("说话大舌头", "疾病")]),
    ("半边身子不太听使唤了", [("半边身子不太听使唤", "疾病")]),
    ("他突然抽风口吐白沫", [("抽风", "疾病"), ("口吐白沫", "疾病")]),
    ("老爷子记性差，问啥都忘", [("记性差", "疾病")]),
    ("脑袋跟灌铅似的，天旋地转", [("天旋地转", "疾病")]),
    ("站起来眼前一黑头昏眼花", [("头昏眼花", "疾病")]),
    ("脑壳疼得想撞墙", [("脑壳疼", "疾病")]),
    ("上不来气，走两步就喘", [("上不来气", "疾病")]),
    ("晚上躺着憋得慌，得坐起来喘", [("憋得慌", "疾病")]),
    ("心口窝一阵阵疼", [("心口窝", "部位"), ("一阵阵疼", "症状")]),
    ("心口窝疼，吃完饭更厉害", [("心口窝疼", "疾病")]),
    ("胃堵得慌，吃不下东西", [("胃堵得慌", "疾病")]),
    ("肚子绞痛，在床上直打滚", [("肚子绞痛", "疾病")]),
    ("老是打嗝停不下来", [("老是打嗝", "疾病")]),
    ("吃啥吐啥，胃里直翻", [("胃里直翻", "症状")]),
    ("心跳跟打鼓似的心慌", [("心慌", "疾病")]),
    ("胸口发闷，压着一块石头", [("胸口发闷", "疾病")]),
    ("两肋那块胀痛", [("两肋", "部位"), ("胀痛", "症状")]),
    ("肋骨边一按就疼", [("肋骨边", "部位")]),
    ("拉肚子拉清水，一天七八趟", [("拉肚子", "疾病")]),
    ("好几天大便干得拉不出来", [("拉不出来", "疾病")]),
    ("想尿尿不出来，小肚子胀", [("尿不出来", "疾病")]),
    ("尿频尿急还刺痛", [("尿频", "症状"), ("尿急", "症状")]),
    ("脸上肿得眼睛都眯缝了", [("脸上肿", "疾病")]),
    ("腿肿得鞋都穿不进去", [("腿肿", "疾病")]),
    ("眼睛眼白都发黄", [("发黄", "疾病")]),
    ("整夜睡不着，眼睛瞪到天亮", [("睡不着", "疾病")]),
    ("腰直不起来，弯个腰都费劲", [("腰直不起来", "疾病")]),
    ("关节又酸又僵，早晨下不了地", [("关节", "部位")]),
    ("身上青一块紫一块", [("青一块紫一块", "疾病")]),
    ("我脑袋不舒服，说不清哪儿", [("脑袋", "部位")]),
    ("脑袋不太舒服,有点晕,还浑身没劲", [("脑袋", "部位"), ("不太舒服", "症状")]),
    ("就是脖子转的时候晕", [("脖子", "部位"), ("晕", "症状")]),
    ("胸口这块总不对劲", [("胸口", "部位")]),
    ("肚子那块胀胀的", [("肚子", "部位"), ("胀胀的", "症状")]),
    ("腰这一片又酸又沉", [("腰", "部位")]),
    ("手脚发麻跟过电似的", [("手脚发麻", "症状")]),
    ("想出去玩", []),
    ("谢谢医生", []),
    ("你好在吗", []),
]

# 俗称塞进碎句（不要「中风已经一周了」）
ALIAS_TEMPLATES = [
    "昨天都{s}，吓死我了",
    "晚上{s}，睡觉都不安生",
    "不知道怎么搞的老{s}",
    "这两天动不动就{s}",
    "我跟家里人说我{s}",
]

BODY_SPOKEN = ("脑袋", "脖子", "胸口", "心口", "肚子", "腰", "手脚", "嗓子")
BODY_TEMPLATES = [
    "我{s}这儿不舒服",
    "{s}那边说不上来的别扭",
    "{s}不太舒服,有点晕,还浑身没劲",
]

TYPE_LABEL = {"body_part": "部位", "disease": "疾病", "symptom": "症状"}


def _add_all_occurrences(
    text: str,
    span: str,
    label: str,
    entities: list[dict],
    used: set[tuple[int, int, str]],
) -> None:
    start = 0
    while True:
        i = text.find(span, start)
        if i < 0:
            break
        pos = (i, i + len(span))
        key = (*pos, label)
        if key not in used:
            used.add(key)
            entities.append(
                {"start_offset": pos[0], "end_offset": pos[1], "label": label}
            )
        start = i + 1


def _lexicon_entities(text: str) -> list[dict]:
    """一句里部位/疾病/症状只要表里能对上就都标；同类内部最长优先，跨类允许嵌套。"""
    entities: list[dict] = []
    used: set[tuple[int, int, str]] = set()
    for etype, label in TYPE_LABEL.items():
        aliases = [
            (a or "").strip()
            for a in (SYNONYM_TABLE.get(etype) or {})
            if len((a or "").strip()) >= 2
        ]
        aliases.sort(key=len, reverse=True)
        occupied = [False] * len(text)
        for alias in aliases:
            start = 0
            while True:
                i = text.find(alias, start)
                if i < 0:
                    break
                j = i + len(alias)
                if not any(occupied[i:j]):
                    occupied[i:j] = [True] * (j - i)
                    key = (i, j, label)
                    if key not in used:
                        used.add(key)
                        entities.append(
                            {
                                "start_offset": i,
                                "end_offset": j,
                                "label": label,
                            }
                        )
                start = i + 1
    return entities


def _row(text: str, extra: list[tuple[str, str]] | None = None) -> dict:
    entities = _lexicon_entities(text)
    used = {
        (e["start_offset"], e["end_offset"], e["label"]) for e in entities
    }
    for span, label in extra or []:
        _add_all_occurrences(text, span, label, entities, used)
    entities.sort(key=lambda e: (e["start_offset"], e["end_offset"], e["label"]))
    for i, ent in enumerate(entities):
        ent["id"] = i
    return {"text": text, "entities": entities, "relations": []}


def _aliases() -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for alias in (SYNONYM_TABLE.get("disease") or {}):
        key = (alias or "").strip()
        if len(key) < 2 or key in seen:
            continue
        seen.add(key)
        names.append(key)
    return names


def main() -> None:
    rows: list[dict] = []
    texts: list[str] = []

    for text, spans in SPOKEN:
        rows.append(_row(text, spans))
        texts.append(text)

    for span in _aliases():
        for tmpl in ALIAS_TEMPLATES:
            text = tmpl.format(s=span)
            rows.append(_row(text))
            texts.append(text)

    for part in BODY_SPOKEN:
        for tmpl in BODY_TEMPLATES:
            text = tmpl.format(s=part)
            rows.append(_row(text, [(part, "部位")]))
            texts.append(text)

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(texts) + "\n", encoding="utf-8")
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows, start=1):
            f.write(json.dumps({"id": i, **row}, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {OUT_JSONL}")


if __name__ == "__main__":
    main()
