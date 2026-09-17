"""生成 B2B 句子意图分类训练语料（text,intent CSV，与老师项目格式一致）。"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

B2B_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = B2B_ROOT / "data" / "intent_classify"
OUTPUT_CSV = DATA_DIR / "data.csv"
OTHER_CSV = DATA_DIR / "other_data.csv"

SYMPTOMS = (
    "头痛", "头疼", "胸闷", "心悸", "胃痛", "腹痛", "腹胀", "恶心", "呕吐",
    "咳嗽", "发热", "乏力", "失眠", "腰酸", "腰痛", "腹泻", "便秘", "口干",
)
BODIES = ("头部", "胸部", "腹部", "腰部", "全身", "四肢", "咽喉", "胃部")
DURATIONS = ("三天了", "一周", "半个月", "最近", "这两天", "今早开始")
COMPLAINT_TEMPLATES = (
    "我{symptom}{duration}",
    "最近{body}{symptom}",
    "医生您好，我{symptom}{duration}",
    "感觉{body}一直{symptom}",
    "早上起来{symptom}，{duration}",
    "孩子{symptom}，{duration}",
)
CHITCHAT_SAMPLES = (
    "你好", "您好", "在吗", "谢谢医生", "再见", "好的", "嗯嗯", "哈哈",
    "今天天气不错", "早上好", "辛苦了", "打扰了", "我想咨询一下挂号",
    "医保怎么报销", "医院几点上班",
)
VAGUE_SAMPLES = (
    "不舒服", "难受", "说不清", "不知道", "嗯", "啊", "随便", "还行",
    "有点问题", "感觉不太好", "说不上来", "帮我看看", "怎么办",
)
HONORIFIC_COMPLAINT = (
    "您好医生，我{symptom}{duration}",
    "老师好，最近{body}{symptom}",
)


def _complaint_rows(count: int, rng: random.Random) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    templates = list(COMPLAINT_TEMPLATES) + list(HONORIFIC_COMPLAINT)
    for _ in range(count):
        tpl = rng.choice(templates)
        text = tpl.format(
            symptom=rng.choice(SYMPTOMS),
            body=rng.choice(BODIES),
            duration=rng.choice(DURATIONS),
        )
        rows.append((text, "complaint"))
    return rows


def _repeat_samples(samples: tuple[str, ...], count: int, rng: random.Random) -> list[tuple[str, str]]:
    return [(rng.choice(samples), "chitchat" if samples is CHITCHAT_SAMPLES else "vague") for _ in range(count)]


def generate_rows(
    *,
    complaint: int = 400,
    chitchat: int = 200,
    vague: int = 200,
    seed: int = 42,
) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    rows = _complaint_rows(complaint, rng)
    rows.extend(_repeat_samples(CHITCHAT_SAMPLES, chitchat, rng))
    rows.extend(_repeat_samples(VAGUE_SAMPLES, vague, rng))
    rng.shuffle(rows)
    return rows


def write_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "intent"])
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 B2B 意图分类 CSV 语料")
    parser.add_argument("--complaint", type=int, default=400)
    parser.add_argument("--chitchat", type=int, default=200)
    parser.add_argument("--vague", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = generate_rows(
        complaint=args.complaint,
        chitchat=args.chitchat,
        vague=args.vague,
        seed=args.seed,
    )
    write_csv(OUTPUT_CSV, rows)

    other_rows = [(text, "chitchat") for text in CHITCHAT_SAMPLES]
    write_csv(OTHER_CSV, other_rows)

    print(f"已写入: {OUTPUT_CSV} ({len(rows)} 条)")
    print(f"已写入: {OTHER_CSV} ({len(other_rows)} 条)")


if __name__ == "__main__":
    main()
