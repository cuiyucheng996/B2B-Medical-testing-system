# B2B 辨证业务流程

机构侧 LangGraph 四轮问诊，知识数据来自 Neo4j（`ConsultBodyPart` / `ConsultDisease` / `SyndromeType` / `SyndromeDialecticScore`）。

## 流程

```
第一轮 [interrupt] 选择部位（Neo4j ConsultBodyPart）
    ↓
第二轮 [interrupt] 选择疾病（ConsultDisease -[:BELONG]-> ConsultBodyPart）
    ↓
加载该病全部证型（SyndromeType）
    ↓
第三轮 [interrupt] 展示各证型主症并集，用户多选 → 缩窄证型
    ↓
  唯一证型 → 输出结果 → END
  多个证型 → 第四轮
    ↓
第四轮 [interrupt] 展示辅助辩证症状（supplementary_symptoms + SyndromeDialecticScore）
    用户多选 → 再缩窄
    ↓
  唯一 → 输出结果 → END
  仍多个且辅助轮次 < 2 → 再问一轮辅助症状
  仍多个且已满 2 轮 → 输出全部潜在证型 → END
```

## 分层

| 层级 | 路径 |
|------|------|
| 字面量 | `src/schema/b2b.py` |
| Neo4j 查询 | `src/knowledge/neo4j_consult.py` |
| 缩窄算法 | `src/algorithm/syndrome_match.py` |
| 图 | `src/graph/` |
| API | `src/api/server.py` |

## 启动

```powershell
uv run python "B2B business/run.py"
```
