# B2B 问诊口语 UIE 标注

Span：`部位`、`疾病`、`症状`。

```powershell
uv run python "B2B business/scripts/generate_uie_consult_spans.py"
```

- 未标注 TextLine：`data/uie/doccano/samples/consult_colloquial.txt`
- 预标注 JSONL：`data/uie/doccano/export/consult_colloquial.jsonl`（圈口语 span，如「咳出点血」，不拿「中风」硬填模板）

转成 UIE 训练文件（仍用主项目转换脚本）：

```powershell
uv run python scripts/doccano_convert_uie.py -d "B2B business/data/uie/doccano/export/consult_colloquial.jsonl" -s "B2B business/data/uie/train"
```

公开语料（标签不同，只作参考）：IMCS 对话 NER、CMeEE、CHIP-CDN、Yidu-S4K。
