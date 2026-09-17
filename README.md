# B2B business

机构侧（B2B）LangGraph 咨询状态机，**代码全部在本目录**，共用主项目 `uv` 依赖。

## 目录

```
B2B business/
├── run.py                 # 启动 API（推荐）
├── doc/
│   ├── program.md         # 流程说明
│   └── b2b_graph.png      # 图结构
└── src/
    ├── schema/            # 业务字面量
    ├── algorithm/         # 业务算法
    ├── graph/             # LangGraph（对齐主项目 src/graph/）
    ├── api/server.py      # FastAPI
    ├── web/               # 测试台
    └── configs/paths.py   # 路径：本目录 + 主项目根
```

## 快速开始

```powershell
# 在 uvgraph 项目根目录
uv sync
uv run python "B2B business/run.py"
```

浏览器打开 http://127.0.0.1:8001

## 编译图

```powershell
cd "B2B business/src"
uv run python -m graph.builder.consult
```

## 与主项目关系

| 项目 | 位置 | 场景 |
|------|------|------|
| uvgraph 主项目 | `src/graph/` | 医疗 C 端问诊 |
| B2C business | `B2C business/` | Java C 端业务 |
| **B2B business** | **`B2B business/`** | Python 机构侧 LangGraph |

- 依赖：主项目 `pyproject.toml`（`uv sync` 一次即可）
- Checkpointer：复用主项目 `src/graph/checkpoint.py`
- 需要 RAG 时 HTTP 调主项目 `/api/raggraph/retrieve`
