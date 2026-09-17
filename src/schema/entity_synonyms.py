"""口语/俗称 → 图谱标准名（对应老师 MySQL 同义词表）。"""

from __future__ import annotations

# entity_type -> {用户说法: 标准名}
SYNONYM_TABLE: dict[str, dict[str, str]] = {
    "body_part": {
        "头": "头面部",
        "脑袋": "头面部",
        "脑袋瓜": "头面部",
        "头疼": "头面部",
        "头痛": "头面部",
        "头晕": "头面部",
        "有点晕": "头面部",
        "发晕": "头面部",
        "脑袋不舒服": "头面部",
        "头部": "头面部",
        "脸": "头面部",
        "脖子": "颈部",
        "颈": "颈部",
        "咽喉": "颈部",
        "嗓子": "颈部",
        "喉咙": "颈部",
        "胸口": "胸胁部",
        "心口": "胸胁部",
        "胸": "胸胁部",
        "胸部": "胸胁部",
        "胸闷": "胸胁部",
        "肚子": "胃腹部",
        "肚": "胃腹部",
        "腹": "胃腹部",
        "腹部": "胃腹部",
        "胃": "胃腹部",
        "胃部": "胃腹部",
        "胃疼": "胃腹部",
        "胃痛": "胃腹部",
        "腰": "四肢躯干",
        "腰疼": "四肢躯干",
        "腰痛": "四肢躯干",
        "腰部": "四肢躯干",
        "手脚": "四肢躯干",
        "四肢": "四肢躯干",
        "胳膊": "四肢躯干",
        "腿": "四肢躯干",
    },
    "disease": {
        "感冒": "感冒",
        "着凉": "感冒",
        "偏头疼": "偏头痛",
        "头疼": "头痛",
        "脑袋疼": "头痛",
        "头胀": "头痛",
        "脑壳疼": "头痛",
        "头晕": "眩晕",
        "天旋地转": "眩晕",
        "头昏眼花": "眩晕",
        "抽风": "痫病",
        "癫痫": "痫病",
        "口吐白沫": "痫病",
        "老年痴呆": "痴呆",
        "记性差": "痴呆",
        "流鼻血": "鼻衄",
        "鼻子出血": "鼻衄",
        "牙龈出血": "齿衄",
        "刷牙出血": "齿衄",
        "咳出点血": "咳血",
        "咳出了血": "咳血",
        "咳出血": "咳血",
        "咳嗽带血": "咳血",
        "痰里有血": "咳血",
        "痰中带血": "咳血",
        "咳痰带血": "咳血",
        "咯血": "咳血",
        "呕血": "吐血",
        "呕吐物有血": "吐血",
        "吐出来的东西带血": "吐血",
        "喘不上气": "喘证",
        "气喘": "喘证",
        "呼吸困难": "喘证",
        "上气不接下气": "喘证",
        "两肋疼": "胁痛",
        "肋骨边疼": "胁痛",
        "心慌": "心悸",
        "心跳快": "心悸",
        "心怦怦跳": "心悸",
        "胸口疼": "胸痹",
        "心口疼": "胸痹",
        "胸口发闷": "胸痹",
        "胃疼": "胃痛",
        "心口窝疼": "胃痛",
        "胃胀": "胃痞",
        "胃堵得慌": "胃痞",
        "反胃": "呕吐",
        "恶心呕吐": "呕吐",
        "打嗝停不了": "呃逆",
        "老是打嗝": "呃逆",
        "肚子疼": "腹痛",
        "肚子绞痛": "腹痛",
        "尿不出来": "癃闭",
        "小便不通": "癃闭",
        "拉肚子": "泄泻",
        "拉稀": "泄泻",
        "大便干": "便秘",
        "拉不出来": "便秘",
        "关节疼": "痹证",
        "手脚麻木": "痹证",
        "腰疼": "腰痛",
        "腰酸": "腰痛",
        "身上起紫斑": "紫斑",
        "皮下出血": "紫斑",
        "睡不着": "不寐",
        "失眠": "不寐",
        "脸肿": "水肿",
        "眼皮肿": "水肿",
        "全身肿": "水肿",
        "身体浮肿": "水肿",
        "皮肤发黄": "黄疸",
        "眼睛发黄": "黄疸",
    },
    "symptom": {
        "头疼": "头痛",
        "脑袋疼": "头痛",
        "胸口闷": "胸闷",
        "拉肚子": "腹泻",
        "拉稀": "腹泻",
        "睡不着": "失眠",
        "发烧": "发热",
        "嗓子疼": "咽痛",
        "没劲": "乏力",
        "没力气": "乏力",
        "没有力气": "乏力",
        "四肢乏力": "乏力",
        "浑身没劲": "乏力",
        "有点晕": "头晕",
        "发晕": "头晕",
        "头晕": "头晕",
        "晕乎乎": "头晕",
        "总冒冷汗": "白天醒时无原因出汗",
        "冒冷汗": "白天醒时无原因出汗",
        "出冷汗": "白天醒时无原因出汗",
        "冷汗": "白天醒时无原因出汗",
        "盗汗": "夜间睡觉时出汗",
        "夜里出汗": "夜间睡觉时出汗",
    },
}

# 只有症状、没有部位时，用症状推断图入口部位
SYMPTOM_TO_BODY: dict[str, str] = {
    "头痛": "头面部",
    "头疼": "头面部",
    "脑袋疼": "头面部",
    "头晕": "头面部",
    "偏头痛": "头面部",
    "胸闷": "胸胁部",
    "心悸": "胸胁部",
    "胃痛": "胃腹部",
    "胃疼": "胃腹部",
    "腹痛": "胃腹部",
    "腹胀": "胃腹部",
    "腰痛": "四肢躯干",
    "腰疼": "四肢躯干",
    "腰酸": "四肢躯干",
    "咳嗽": "胸胁部",
    "咽痛": "头面部",
    "乏力": "四肢躯干",
    "没劲": "四肢躯干",
}


def rewrite_synonyms_in_text(mention: str) -> str:
    """整句口语别名替换成标准名，BERT 分类前与训练语料同一套映射。

    症状/疾病优先于部位，避免「头疼」被改成「头面部」。最长别名优先、互不重叠。
    """
    text = (mention or "").replace("　", "").strip()
    if not text:
        return text
    merged: dict[str, str] = {}
    for entity_type in ("symptom", "disease", "body_part"):
        for alias, std in (SYNONYM_TABLE.get(entity_type) or {}).items():
            key = (alias or "").replace(" ", "").replace("　", "").strip()
            std_name = str(std or "").strip()
            if len(key) < 2 or not std_name or key == std_name:
                continue
            merged.setdefault(key, std_name)
    aliases = sorted(merged.items(), key=lambda item: len(item[0]), reverse=True)
    used = [False] * len(text)
    replacements: list[tuple[int, int, str]] = []
    for key, std_name in aliases:
        start = 0
        while True:
            idx = text.find(key, start)
            if idx < 0:
                break
            end = idx + len(key)
            if not any(used[idx:end]):
                replacements.append((idx, end, std_name))
                for pos in range(idx, end):
                    used[pos] = True
            start = idx + 1
    if not replacements:
        return text
    replacements.sort(key=lambda item: item[0])
    out: list[str] = []
    cursor = 0
    for start, end, std_name in replacements:
        out.append(text[cursor:start])
        out.append(std_name)
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


def collect_synonym_spans(entity_type: str, mention: str) -> list[tuple[str, str]]:
    """整句命中的口语片段 → 标准名（长别名优先，已匹配片段不再重复切）。"""
    table = SYNONYM_TABLE.get(entity_type) or {}
    text = (mention or "").replace(" ", "").replace("　", "").strip()
    if not text:
        return []
    aliases = []
    for alias, std in table.items():
        key = (alias or "").replace(" ", "").replace("　", "").strip()
        if len(key) < 2 or not std:
            continue
        aliases.append((key, str(std).strip()))
    aliases.sort(key=lambda item: len(item[0]), reverse=True)
    remaining = text
    found: list[tuple[str, str]] = []
    for key, std in aliases:
        if key not in remaining:
            continue
        found.append((key, std))
        remaining = remaining.replace(key, "\0" * len(key), 1)
    return found


def collect_synonym_standards(entity_type: str, mention: str) -> list[str]:
    """一句里可命中多条口语别名对应的标准名。"""
    return list(dict.fromkeys(std for _, std in collect_synonym_spans(entity_type, mention)))


def lookup_synonym(entity_type: str, mention: str) -> str | None:
    """口语命中同义词表；多条同时命中时取最长别名，避免「血」误配。"""
    table = SYNONYM_TABLE.get(entity_type) or {}
    text = (mention or "").replace(" ", "").replace("　", "").strip()
    if not text:
        return None
    if text in table:
        return table[text]
    hits: list[tuple[str, str]] = []
    for alias, std in table.items():
        key = (alias or "").replace(" ", "").replace("　", "").strip()
        if len(key) < 2:
            continue
        if key in text or text in key:
            hits.append((key, std))
    if not hits:
        return None
    hits.sort(key=lambda item: len(item[0]), reverse=True)
    best_key, best_std = hits[0]
    extras = [std for key, std in hits[1:] if key not in best_key and std != best_std]
    if extras:
        return None
    return best_std


def body_from_symptom(mention: str) -> str | None:
    text = (mention or "").strip()
    if text in SYMPTOM_TO_BODY:
        return SYMPTOM_TO_BODY[text]
    for key, body in SYMPTOM_TO_BODY.items():
        if key and key in text:
            return body
    return lookup_synonym("body_part", text)
