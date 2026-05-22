# CLAUDE.md — RAG 学习项目协作指引

> 这是 Zihang He 的 RAG 学习项目仓库,目标是从朴素 RAG 演进到 agentic RAG,最终落到 UCL CASA0022 dissertation 的 IEQ-Ops 项目 Phase A。
>
> 本文件是**稳定的协作规范**,极少改动;**一步一步执行的详细步骤见 [PLAN.md](PLAN.md)**。

---

## 1. 项目目标
我希望尽可能的了解这个项目中每一个步骤。所有哪怕你只是装了一个大的，未来会一直提到的环境或者工具，都解释一下。但是一些小的工具类似dll那种就不要解释了。
把"读懂 RAG 各层技术为什么需要"和"产出真实项目代码"两件事拆成两条主线,顺序推进:

- **Track A · 原型线(douluo/)** — 用《斗罗大陆》语料 + 本地 Qwen3-8B 走完 A1→A5 五代 RAG,理解每一层升级的实际收益。
- **Track B · 项目线(rag/)** — 用 ASHRAE PDF + 云端 LLM 把 Track A 学到的东西工程化为 9 个文件,符合 Phase A 验收标准。
- **阶段 7 · MCP 封装(mcp_servers/rag/)** — 将 Retriever 暴露为 MCP server,可被 specialist agent 调用。

最终产出:9 个工程文件 + 1 份跨阶段对比实验报告 + 1 个可挂载的 MCP server。

---

## 2. 用户背景

- Python 中级,熟悉基本工程实践,但需要 Claude 帮忙写非平凡代码。
- 对 AI / LLM 概念有了解,**第一次系统做 RAG**,目标是先跑通再深入原理。
- 显卡 RTX 3060(显存 6 GB)。这意味着:
  - 本地只能跑量化版 Qwen3-8B(占约 5 GB)。
  - Reranker / embedding 训练或大模型推理不能与 LLM 同时占显存 —— reranker 走 CPU 或单独跑。
  - Track B 的 BGE-M3、bge-reranker-v2-m3 可以 GPU,但不要和本地 LLM 同时加载。
- Track B 使用云端 LLM(DashScope Qwen / DeepSeek),通过 OpenAI 兼容 SDK 调用,需要 Prompt Caching 来摊薄成本。

---

## 3. 工作原则(硬约束,不要违反)

1. **一步一步推进** — 一次只做 PLAN.md 里**一个未勾选的小步骤**,做完让用户验收再继续。不要预先把多个阶段连写。
2. **跑通优先,原理后补** — 但 Track A 的对比实验**不能跳过**,那是用户理解力的载体。
3. **金标问题集 / gold.jsonl 是评测真理** — 没有它,所有"我比上一阶段好"的判断都是错觉。在 A0 阶段建立。
4. **不要中途换语料** — Track A 全程斗罗,Track B 全程 ASHRAE,对比基准不变。
5. **每个新阶段开新 git 分支或独立文件** — 方便随时切回去跑对比。Track A 文件名固定为 `naive_rag.py / hybrid_rag.py / rerank_rag.py / contextual_rag.py / agentic_rag.py`。
6. **Prompt 全部走文件版本化** — Track B 的所有 prompt 必须放在 `ops/prompts/{name}/v{n}.md`,不要内联在 .py 文件里。Track A 可以内联,但养成读 prompt 的习惯。
7. **报错先复现再贴 Claude** — 环境问题(CUDA / 显存 / 依赖冲突)最耗时,先复现并贴完整 traceback,不要凭印象描述。
8. **每个阶段写完代码后,运行一次验证命令再标记完成** — 没跑过的代码不算完成。

---

## 4. 目录结构

```
/home/xms/projects/rag/
├── CLAUDE.md                     # 本文件(协作规范)
├── PLAN.md                       # 详细步骤清单(一步一步执行的依据)
│
├── douluo/                       # Track A 原型(熟悉语料练手感)
│   ├── pyproject.toml            # Track A 独立的 uv 项目
│   ├── corpus/douluo/            # 斗罗大陆按章节切分的 txt
│   ├── golden_questions.json     # 20-30 个金标问题(A0 建立)
│   ├── naive_rag.py              # A1
│   ├── hybrid_rag.py             # A2
│   ├── rerank_rag.py             # A3
│   ├── contextual_rag.py         # A4
│   ├── agentic_rag.py            # A5
│   └── report.md                 # 5 阶段对比实验报告(A5 完成时产出)
│
├── rag/                          # Track B 工程化代码(对应 Phase A 的 9 个文件)
│   ├── schemas.py                # B1
│   ├── config.py                 # B1
│   ├── parse.py                  # B2
│   ├── chunk.py                  # B2
│   ├── contextualize.py          # B3
│   ├── embed.py                  # B4
│   ├── index_dense.py            # B4
│   ├── index_sparse.py           # B5
│   ├── fuse.py                   # B6
│   ├── rerank.py                 # B6
│   ├── retrieve.py               # B6
│   ├── ingest.py                 # B7
│   ├── corpus/                   # ASHRAE / WELL / EN PDF
│   ├── data/                     # sparse_index 等持久化产物
│   ├── eval/{metrics.py,run.py,gold.jsonl}   # B8
│   └── tests/                    # B8 pytest(纯逻辑)
│
├── ops/
│   └── prompts/                  # 所有 Track B 的 prompt 版本化在这里
│       ├── contextualize/v1.md
│       ├── grade/v1.md
│       └── ...
│
├── mcp_servers/
│   └── rag/server.py             # 阶段 7
│
├── pyproject.toml                # Track B 的 uv 项目(根目录)
├── docker-compose.yml            # B0 — Qdrant
└── .env.example                  # B0
```

**两条线物理隔离**:`douluo/` 是独立 uv 项目,Track A 的依赖(rank-bm25 / faiss / langchain-ollama)不污染 Track B 的工程依赖。

---

## 5. 技术栈速查

| 维度 | Track A(原型) | Track B(项目) |
|---|---|---|
| 语料 | 斗罗大陆 txt | ASHRAE / WELL / EN / WHO PDF |
| LLM | 本地 Ollama Qwen3-8B | DashScope Qwen + DeepSeek(OpenAI 兼容 SDK) |
| Embedding | bge-small-zh → BGE-M3 | 直接 BGE-M3 |
| 向量库 | 内存 FAISS | Qdrant + Docker |
| Sparse | rank-bm25 | bm25s |
| Reranker | bge-reranker-v2-m3(CPU) | bge-reranker-v2-m3(GPU/fp16) |
| Schema | 直接 dict | 严格 Pydantic |
| Prompt | 内联 py 文件 | `ops/prompts/{name}/v{n}.md` |
| 编排 | A5 引入 LangGraph | LangChain 零件 + LangGraph 子图 |
| 评测 | 命中率手工统计 | ranx(Recall@K / MRR / nDCG@10) |

---

## 6. 当前进度指针

**永远以 `PLAN.md` 里最后一个未勾选的 `[ ]` 为准。**

启动新对话时,Claude 应当:
1. 先读 `PLAN.md`,找到下一个未勾选的步骤。
2. 不要预先做后面阶段的工作,即使用户问"接下来还有什么"。
3. 用户明确说"做完 X 接着做 Y"时,才能连续推进。

---

## 7. 协作风格约定

- **解释要分层**:用户是 Python 中级 + 第一次做 RAG,概念要先一句话讲清"为什么需要",再贴代码。不要直接甩 100 行未注释代码。
- **代码风格**:Type hint 全开;函数 docstring 写一句"做什么 + 为什么";不要过度注释每一行。
- **遇到取舍**:列出两三个选项 + 各自的取舍,让用户决定。不要替用户做不可逆决定(删文件、覆盖语料、强推依赖大升级)。
- **环境命令**:Track A 和 Track B 是两个 uv 项目,执行任何 `uv add` 之前先 `cd` 到对应目录;Claude 在 Bash 调用里要带绝对路径或显式 `cd`,避免装错地方。
- **当 Claude 写不动时直接说**:不要伪造跑通的输出。如果某一步在本机跑不出来(显存不够、网络封禁等),直接告诉用户改换方案。

---

## 8. 关键术语对照表

新概念第一次出现时,Claude 应当用这个对照解释,避免用户被名词绊住:

| 名词 | 一句话理解 |
|---|---|
| Dense retrieval | 把文本压成 1024 维向量,用余弦相似度找近义。 |
| Sparse retrieval / BM25 | 关键词权重打分,擅长专有名词、术语。 |
| Hybrid | Dense + Sparse 两路结果用 RRF 融合,工业界标配。 |
| RRF | Reciprocal Rank Fusion,8 行公式直接抄,不是黑魔法。 |
| Bi-encoder | query 和 chunk 分别 embed,适合大规模粗排。 |
| Cross-encoder / Reranker | query+chunk 一起进模型打分,准但慢,用于精排 top-5。 |
| Contextual Retrieval | embed 之前给 chunk 补一段 LLM 生成的上下文前缀,提升跨章节召回。 |
| Agentic RAG | 检索→判够不够→不够就改写 query 再检索,用 LangGraph 状态机实现。 |
| Prompt Caching | 把长前缀(整篇 PDF)缓存在 API 端,后续 chunk 请求只发新增部分,成本降一个数量级。 |
| MCP | Model Context Protocol,把工具暴露给 LLM 调用的标准协议。 |

---

## 9. 何时更新本文件

CLAUDE.md 应保持稳定。下列情况才更新:

- 用户技术栈方向变了(比如从 Ollama 换到 vLLM)。
- 目录结构有调整。
- 出现一条新的硬约束(被用户明确说"以后都这么做")。

其他所有进度、阶段细节、踩坑笔记,**都写到 PLAN.md 或对应阶段的产出文档里**,不要塞进 CLAUDE.md。
