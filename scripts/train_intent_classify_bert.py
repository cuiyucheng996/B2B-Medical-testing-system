"""B2B 句子意图分类 BERT 微调（参照老师 train_intent_classify_bert.py）。"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

B2B_ROOT = Path(__file__).resolve().parents[1]
B2C_ROOT = B2B_ROOT.parent / "B2C business"
if str(B2B_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(B2B_ROOT / "src"))

CONFIG = {
    "data_path": B2B_ROOT / "data" / "intent_classify" / "data.csv",
    "other_data_path": B2B_ROOT / "data" / "intent_classify" / "other_data.csv",
    "model_name": B2C_ROOT / "pretrained" / "bert-base-chinese",
    "output_dir": B2B_ROOT / "checkpoint" / "intent_classify",
    "max_length": 128,
    "batch_size": 16,
    "epochs": 5,
    "learning_rate": 2e-5,
    "test_size": 0.2,
    "random_state": 42,
}


def check_local_model(path: Path) -> None:
    required = {"config.json", "vocab.txt"}
    weight_files = {"pytorch_model.bin", "model.safetensors"}
    if not path.is_dir():
        raise NotADirectoryError(f"本地模型目录不存在: {path}")
    files = set(os.listdir(path))
    missing = required - files
    if missing:
        raise FileNotFoundError(f"本地模型缺少必要文件: {missing}")
    if not files.intersection(weight_files):
        raise FileNotFoundError("未找到模型权重文件 (pytorch_model.bin 或 model.safetensors)")
    print(f"本地模型校验通过: {path}")


def load_and_preprocess_data(config: dict):
    df = pd.read_csv(config["data_path"], encoding="utf-8")
    other_path = config.get("other_data_path")
    if other_path and Path(other_path).exists():
        df_other = pd.read_csv(other_path, encoding="utf-8")
        df_other["intent"] = "chitchat"
        df = pd.concat([df, df_other], ignore_index=True)
        print(f"已合并 other 数据: {len(df_other)} 条")

    df = df.dropna(subset=["text", "intent"]).reset_index(drop=True)
    print(f"有效数据: {len(df)} 条")

    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["intent"])
    label_mapping = {int(i): str(label) for i, label in enumerate(label_encoder.classes_)}

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "label_mapping.json").open("w", encoding="utf-8") as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=2)

    train_df, test_df = train_test_split(
        df,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=df["label"] if len(df["label"].unique()) > 1 else None,
    )
    print(f"训练集: {len(train_df)}, 测试集: {len(test_df)}")
    return train_df, test_df, label_encoder, label_mapping


class IntentDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.texts = dataframe["text"].tolist()
        self.labels = dataframe["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_token_type_ids=True,
        )
        item = {k: torch.tensor(v) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_metrics(eval_pred):
    import numpy as np

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
    }


def train(config: dict | None = None) -> None:
    cfg = config or CONFIG
    print("开始 B2B 句子意图分类 BERT 微调...")
    check_local_model(Path(cfg["model_name"]))

    train_df, test_df, label_encoder, label_mapping = load_and_preprocess_data(cfg)
    tokenizer = BertTokenizer.from_pretrained(cfg["model_name"])
    train_dataset = IntentDataset(train_df, tokenizer, max_length=cfg["max_length"])
    test_dataset = IntentDataset(test_df, tokenizer, max_length=cfg["max_length"])

    model = BertForSequenceClassification.from_pretrained(
        cfg["model_name"],
        num_labels=len(label_encoder.classes_),
        problem_type="single_label_classification",
    )

    output_dir = Path(cfg["output_dir"])
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_dir=str(output_dir / "logs"),
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        seed=cfg["random_state"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("\n训练开始...")
    train_result = trainer.train()
    print(f"训练完成: {train_result.metrics}")

    print("\n测试集评估:")
    eval_result = trainer.evaluate()
    for key, value in eval_result.items():
        if isinstance(value, float):
            print(f" {key}: {value:.4f}")

    preds_output = trainer.predict(test_dataset)
    import numpy as np

    preds = np.argmax(preds_output.predictions, axis=-1)
    print("\n分类报告:")
    print(
        classification_report(
            test_dataset.labels,
            preds,
            target_names=[label_mapping[i] for i in range(len(label_mapping))],
            zero_division=0,
        )
    )

    best_model_path = output_dir / "best_model"
    trainer.save_model(str(best_model_path))
    tokenizer.save_pretrained(str(best_model_path))
    print(f"\n最佳模型已保存: {best_model_path}")


if __name__ == "__main__":
    data_path = CONFIG["data_path"]
    if not data_path.exists():
        print(f"语料不存在，先生成: {data_path}")
        import runpy

        runpy.run_path(str(B2B_ROOT / "scripts" / "generate_intent_classify_data.py"), run_name="__main__")
    train(CONFIG)
