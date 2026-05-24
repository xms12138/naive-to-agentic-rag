# naive-to-agentic-rag

> From naive to agentic RAG — a five-stage Chinese RAG learning project on the *Douluo Dalu* corpus, building toward the UCL CASA0022 IEQ-Ops dissertation.

一个**从零理解 RAG 每一层为什么需要**的学习项目。用《斗罗大陆》中文语料做练手台,在**同语料、同评测集**的前提下,逐层叠加检索技术,亲眼看到每一步带来的真实收益,最终把经验工程化为符合学术验收标准的项目代码。

核心约束:**唯一变量是检索/生成流水线**,语料和 30 题金标问题集全程不变,所以每一阶段的提升都是可信的,而不是错觉。

---

## 两条主线

| 线 | 目录 | 目的 | 语料 | LLM |
|---|---|---|---|---|
| **Track A · 原型** | `douluo/` | 用熟悉语料练手感,理解每层升级解决了什么具体问题 | 《斗罗大陆》前 10 章 | 本地 Ollama Qwen3-8B + 云端 DeepSeek |
| **Track B · 项目** | `rag/` | 把 Track A 的经验工程化为 9 个文件,符合 dissertation Phase A 验收 | ASHRAE / WELL PDF | 云端 DashScope / DeepSeek |
| **阶段 7 · MCP** | `mcp_servers/rag/` | 把 Retriever 暴露为 MCP server,供 agent 调用 | — | — |

---

## Track A 消融实验结果

五代 RAG,30 题金标问题集(事实 / 关系 / 跨章节 / 模糊 / 陷阱 五类各 6 题),手工标注 `hit / partial / miss / hallucinate`:

| 阶段 | 流水线变化 | 准答率 | hit-only | 平均延迟 |
|---|---|---|---|---|
| **A1 朴素** | Dense(bge-small-zh)→ top-3 → LLM | 63.3% | 50.0% | 17.9s |
| **A2 hybrid** | + BM25(jieba)+ RRF 融合 | 76.7% | 63.3% | 20.9s |
| **A3 rerank** | + bge-reranker-v2-m3(CPU)精排 | 86.7% | 80.0% | 44.6s |
| **A4 contextual** | + LLM 生成 ctx_prefix 注入 embed/BM25/rerank | 90.0% | 83.3% | 48.3s |
| **A4.5 LLM 消融** | 检索不变,生成层换云端 DeepSeek | 100% | 96.7% | 1.1s |
| **A5 agentic** | + LangGraph 状态机(decompose/grade/rewrite/多轮检索) | 100% | 86.7% | 57.5s |

**关键发现**:
- A4.5 是整个实验的转折点 —— 同样的 top-5 检索结果,只把生成层从本地量化 Qwen3-8B 换成云端 DeepSeek,准答率从 90% 跳到 100%。**说明 A4 阶段的瓶颈已经从检索层转移到生成层**,这一发现直接决定了 dissertation 各 agent 的本地/云端 LLM 路由。
- 闭卷对照(DeepSeek 不给任何 chunk,凭记忆答题)只有 66.7% —— 排除了"小说本身在训练数据里"的干扰,证明检索确实在起作用。

完整逐阶段分析见 [`douluo/report.md`](douluo/report.md)。

---

## 目录结构

```
rag/
├── CLAUDE.md          # 协作规范(稳定,极少改)
├── PLAN.md            # 一步一步的实施清单(进度真理)
├── douluo/            # Track A 原型(独立 uv 项目)
│   ├── naive_rag.py       # A1 朴素
│   ├── hybrid_rag.py      # A2 Dense + BM25 + RRF
│   ├── rerank_rag.py      # A3 cross-encoder 精排
│   ├── contextual_rag.py  # A4 contextual retrieval
│   ├── agentic_rag.py     # A5 LangGraph agentic
│   ├── golden_questions.json  # 30 题金标评测集
│   └── report.md          # 五阶段对比报告
├── rag/               # Track B 工程化代码(规划中)
├── ops/prompts/       # Track B 的版本化 prompt
└── mcp_servers/rag/   # 阶段 7 MCP server(规划中)
```

> 语料(`corpus/`)涉及版权且体积大,不入库;`.venv/ data/ cache/ runs/` 等本地产物同样忽略。

---

## 当前进度

- **Track A**:A0–A5 + A4.5 消融实验 **已完成**,产出 5 个递进版本 + 完整对比报告。
- **Track B**:规划中(B0–B8)。
- **阶段 7 MCP**:规划中。

逐项进度以 [`PLAN.md`](PLAN.md) 里最后一个未勾选的 `[ ]` 为准。

---

## 技术栈

**Track A**:LangChain + LangGraph · faiss-cpu · rank-bm25(jieba 分词)· bge-small-zh / BGE-M3 · bge-reranker-v2-m3 · Ollama Qwen3-8B · DeepSeek(OpenAI 兼容 SDK)· uv 管理依赖。

硬件:RTX 3060(6 GB 显存)—— 本地只跑量化 LLM,reranker 走 CPU,二者不同时占显存。

---

## 运行(Track A)

```bash
cd douluo
uv sync
# 单条查询
uv run python naive_rag.py --query "唐三的第一武魂是什么"
# 跑完整金标评测集
uv run python rerank_rag.py --eval
```

> Track A 通过 WSL2 调用 Windows 端 Ollama,需先 `ollama pull qwen3:8b`;云端阶段需在 `douluo/.env` 配置 DeepSeek API key。
