# PLAN.md — RAG 学习实施步骤

> 这是一步一步推进的工作清单。每个 `[ ]` 是一个可独立验收的小步骤,做完一项打勾一项。**协作规范、技术栈速查、目录结构见 [CLAUDE.md](CLAUDE.md)**。
>
> 当前总进度入口:从 `Track A · A0` 开始。

---

## 总览

```
[Track A 原型(douluo/)]   A0 → A1 → A2 → A3 → A4 → A5
                                                    ↓
                                                [桥接]
                                                    ↓
[Track B 项目(rag/)]      B0 → B1 → B2 → B3 → B4 → B5 → B6 → B7 → B8
                                                                  ↓
                                                             [阶段 7 MCP]
```

| 阶段 | 预计耗时 | 主产出 |
|---|---|---|
| Track A 原型 | 6-8 天 | 5 个递进版本 + 对比报告 |
| 桥接 | 0.5 天 | 想清楚 8 个工程化"为什么" |
| Track B 项目 | 7-9 天 | rag/ 目录 9 个文件 + pytest + eval |
| 阶段 7 MCP | 1-2 天 | mcp_servers/rag/server.py |
| **总计** | **约 2.5-3 周** | 每天 3-4 小时 |

---

# Track A · 原型线(douluo/)

> 目的:用熟悉的中文语料练手感,真正理解每一层 RAG 升级**解决了什么具体问题**。

---

## A0 · 环境准备与金标问题集(0.5 天)

> A0 是整个 Track A 的灵魂。**金标问题集没建好,后面 A1-A5 的对比全是错觉。**

**步骤**

- [x] 在 `/home/xms/projects/rag/douluo/` 下 `uv init` 建独立项目
- [x] `uv add langchain langchain-community langchain-ollama sentence-transformers faiss-cpu`  *(实际换成 langchain-openai + openai,通过 WSL2 调 Windows 端 ollama)*
- [x] `ollama pull qwen3:8b` 并 `ollama run qwen3:8b` 验证能对话
- [x] 把 `《斗罗大陆》_qinkan.net.txt` 编码统一成 UTF-8  *(原 gb18030,prepare_corpus.py 转码)*
- [x] 清洗:删除盗版水印、"本章未完点击下一页"等噪声行
- [x] 按章节切分到 `corpus/douluo/ch001.txt`...,**只抽前 5-10 章先用**(全本迭代太慢)  *(取前 10 章:引子 + 第一~第九章)*
- [x] 建 `golden_questions.json`,写 20-30 个问题,五类各占一些:  *(30 题,五类各 6)*
  - 事实型:"唐三的第一武魂是什么"(蓝银草)
  - 关系型:"小舞和唐三是什么关系"
  - 跨章节型:"唐三获得了哪些魂环"
  - 模糊型:"海神九考都有哪些"
  - 陷阱题(原文没有):"唐三在霍格沃茨学了什么"——测 hallucination
- [x] 每题手工标注期望答案(JSON 里加 `expected` 字段)

**验收**:能 `cat golden_questions.json | jq length` 输出 ≥ 20,每题有 `query / expected / type`。

---

## A1 · 朴素 RAG(1 天)

> 目标:跑通最小闭环,让你**亲眼看到朴素 RAG 的局限**。

**流水线**

```
txt → RecursiveCharacterTextSplitter (chunk_size=500, overlap=50)
    → bge-small-zh embed → 内存 FAISS
    → query embed → top-3 → 拼 prompt → Qwen3-8B → 答案
```

**关键选择的理由**

- `chunk_size=500`:中文对话一段 100-300 字,500 字能包住 1-2 段完整对话。
- 用 `bge-small-zh` 不用 BGE-M3:small 在 CPU 就能跑,不抢 Qwen3 的 6 GB 显存;到 A2 再升级才能感受差异。

**步骤**

- [x] `uv add` 还没装的依赖  *(langchain-text-splitters,langchain 1.x 已传递依赖)*
- [x] 写 `naive_rag.py`:CLI 接收 `--query` 参数,跑完整流水线返回答案  *(同时支持 `--eval` 批量模式)*
- [x] 跑 20 个金标问题,把结果写到 `runs/a1.jsonl`  *(实际跑全部 30 题)*
- [x] 手工标注每题:命中 / 部分命中 / 没命中 / hallucinate  *(15 hit / 4 partial / 10 miss / 1 hallucinate)*
- [x] 在 `report.md` 起一张表(后面 A2-A5 同列累加)

**预期会发现的问题**(记下来,后续阶段就是解决这些):

- 跨章节问题基本答不对
- 人物名 / 武魂名这类关键词,dense embedding 反而不准
- chunk 切到对话中间语义残缺
- top-3 里经常 1-2 个完全无关的 chunk 污染上下文

**产出**:`naive_rag.py` + `runs/a1.jsonl` + `report.md` 起表头。

---

## A2 · Hybrid Retrieval(1-2 天)

> 目标:解决"关键词类问题 dense embedding 不准"。

**步骤**

- [x] `uv add rank-bm25`  *(rank-bm25==0.2.2)*
- [x] 复制 `naive_rag.py` → `hybrid_rag.py`
- [x] 加 BM25 索引,query 时同时跑 dense top-20 和 sparse top-20  *(jieba 分词 + rank-bm25)*
- [x] 实现 RRF 融合(8 行,直接抄):

```python
def rrf(dense_ids, sparse_ids, k=60):
    scores = {}
    for rank, doc_id in enumerate(dense_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    for rank, doc_id in enumerate(sparse_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)
```

- [x] 融合后取 top-5,拼 prompt,跑 Qwen3-8B  *(代码就绪,等 verbose 验收)*
- [x] 跑同一批 30 题 → `runs/a2.jsonl`  *(总耗时 10.4 min)*
- [x] `report.md` 增加 A2 列,对比 A1  *(19 hit + 4 partial + 6 miss + 1 hallu = 76.7%,提升 13 pp)*

**核心概念**:sparse(关键词)vs dense(语义)的盲区互补。工业界几乎都跑 hybrid。"昊天锤是谁的武魂"这种专有名词类问题提升最明显。

**产出**:`hybrid_rag.py` + `runs/a2.jsonl` + `report.md` 更新。

---

## A3 · Reranker 精排(1 天)

> 目标:解决"top-k 里混入无关 chunk 污染上下文"。

**流水线变化**

```
BM25 + Dense → RRF top-30 → bge-reranker-v2-m3 cross-encoder 重排 → top-5 → prompt
```

**显存取舍**

- Qwen3-8B 量化版约占 5 GB
- bge-reranker-v2-m3 走 **CPU** 即可,慢一点但不抢显存
- 或换更小的 `bge-reranker-base`

**步骤**

- [x] 复制 `hybrid_rag.py` → `rerank_rag.py`
- [x] 加载 `bge-reranker-v2-m3`(CPU)
- [x] RRF 后取 top-30,送 reranker 重排,取 top-5
- [x] 跑同一批 30 题 → `runs/a3.jsonl`,**额外记录每题延迟**  *(准答率 86.7%,24 hit + 2 partial + 3 miss + 1 hallu,平均 44.6s/题,其中 reranker 22.9s)*
- [x] `report.md` 增加 A3 列 + 延迟列  *(核心战果:Q11 hallu→hit;反面案例:Q18 partial→hallu,候选池偏掉时 reranker "自信精排到错处")*

**核心概念**:bi-encoder(粗排,每个 chunk 单独编码,快)vs cross-encoder(精排,query+chunk 一起进模型,准)。几乎所有生产 RAG 都是这个两阶段结构。

**产出**:`rerank_rag.py` + `runs/a3.jsonl` + `report.md` 加延迟。

---

## A4 · Contextual Retrieval(1-2 天)

> 目标:解决 "chunk 脱离上下文后语义残缺"。**这也是 Phase B 的核心技术,先在熟悉语料上练熟。**

**原理**:每个 chunk embed 之前,用 LLM 生成 50-100 字的上下文前缀,prepend 到 chunk 文本上**再 embed 和建 BM25**。

**Prompt 示例**

```
这是一段来自《斗罗大陆》的文本,整本书讲述唐三在斗罗大陆的成长故事。
请用一两句话说明下面这段文字在整本书中的语境(涉及的角色、时间段、关键事件):

{chunk}
```

**工程坑预警**

- 几千个 chunk 每个调一次本地 LLM 生成前缀,**耗时不小**
- 先**只对前 5 章**跑试效果,验证有用了再扩到全部
- 把生成的前缀**缓存到磁盘**(JSON 或 SQLite),不要每次实验都重跑
- 估算:3000 chunks × 3 秒 ≈ 2.5 小时

**步骤**

- [x] 写 `contextual_ingest.py`:对每个 chunk 生成 ctx_prefix,缓存到 `cache/ctx.sqlite`  *(改用 DeepSeek-V4-Flash + 隐式缓存,不用本地 LLM —— 185 chunk 5.3 分钟,¥0.25,缓存命中率 99.3%)*
- [x] 先只跑前 5 章,看人工抽查 10 条前缀质量是否过关  *(跳过:用户选 B 速跑;ch001 前 3 条已验质量,直接全量)*
- [x] 质量 OK 再全量跑  *(185/185 全部 prefix 生成完成,inspect 10 抽查全部合格)*
- [x] 写 `contextual_rag.py`:加载缓存,用 `ctx_prefix + chunk` 重新建 dense 和 sparse 索引,query 流程复用 A3  *(LLM prompt 仍用原文 text,避免"自生成回声")*
- [x] 跑 30 题 → `runs/a4.jsonl`,重点看跨章节、需要上下文判断的问题  *(RRF top-1: 69%→79.3% (+10.3 pp),核心修复 Q18 hallucinate→hit)*
- [x] `report.md` 增加 A4 列  *(25 hit + 2 partial + 3 miss + 0 hallu,准答率 90.0%,增量全部集中在跨章节型 83→100%)*

**直觉验证**:一段被切出来的对话原本只有"他举起了那把锤子",加了前缀变成"在与昊天宗的冲突中,唐三举起了那把锤子"——召回率应该有质变。

**产出**:`contextual_ingest.py` + `contextual_rag.py` + `cache/ctx.sqlite` + `report.md` 更新。

---

## A4.5 · LLM 消融实验:验证瓶颈是检索层还是生成层(0.5 天)

> 目的:在做 A5 之前,先用最便宜的实验证明 "A4 的 5 道 partial/miss 是 Qwen3-8B 本地量化版的能力上限,不是检索缺陷"。如果换云端 LLM 直接 hit,A5 的设计目标要随之调整(详见下一节先决条件)。

**背景**(详见 `report.md` A4 节 + 2026-05-24 对话分析):

A4 实测 **Recall@5 = 96.6%**,Q9 / Q12 / Q13 / Q16 四题的正确 chunk 已经在 top-5 里,但 Qwen3-8B 本地版没合成出来:
- **Q12** chunk 里 "唐门" 出现 ≥3 次(prefix 也写了),LLM 答 "原文未提及"
- **Q13** top-5 完整覆盖 "圣魂村 / 天斗帝国 / 唐昊 / 转生" 全部要素,LLM 只答 "跳崖"
- **Q16** chunk 直接写 "调查过六百四十七个武魂为蓝银草的人",LLM 抹成 "概率极低"
- **Q9** chunk 多次出现 "七舍 / 工读生 / 舍长",LLM 答 "原文未提及"

**实验设计**:只换 LLM,**复用 a4.jsonl 的 top-5 检索结果不重新检索**,确保唯一变量是生成层。

**步骤**

- [x] 写 `a4_swap_llm.py`:从 `runs/a4.jsonl` 读 top-5 chunks + 原 prompt,**只换 LLM 调用** 到 DeepSeek-V4-Flash(复用 `contextual_ingest.py` 的 OpenAI 兼容 client 写法,关 thinking)
- [x] 跑全部 30 题(用户决定扩大范围)→ `runs/a4_swap.jsonl`  *(32.7s,¥0.0427,平均 1.09s/题)*
- [x] Claude 主动判 30 题 hit / partial / miss,对照 a4 原答案  *(Qwen3-8B 25/2/3/0 vs DeepSeek 29/1/0/0,净 +5 题 -1 题)*
- [x] `report.md` 增加 "LLM 消融" 小节,记结果 + 对 A5 设计的影响  *(主对比表加 A4.5 行,A4 节后插完整 4 小节细账)*

**预算估算**:5 题 × ~3KB prompt ≈ 15K tokens,DeepSeek-V3-Flash 成本 < **¥0.01**,耗时 < 1 分钟。

**预期结果分支**(决定 A5 怎么做):

| 结果 | A5 含义 |
|---|---|
| **4 题全 hit** ← **实测落在这一支** | 瓶颈确认在 LLM。A5 主要价值从 "刷准答率" 变成 "练 LangGraph 工程模式"(Track B 主图必用),不追准答率提升 |
| 仍有 ≥2 题 partial/miss | 部分题需要 decompose / rewrite / 多轮检索,A5 的检索反思价值得到验证,按原计划全力做 |
| Q3 仍 miss | 评测口径问题(LLM 复述能力问题),不是 RAG 系统问题,不进 A5 范围 |

**实测结果(30 题扩展版,2026-05-24)**:Q3/Q9/Q12/Q13/Q16 五题全部翻 hit,新增 Q18 hit→partial(漏门房改列食堂)。准答率 90%→100%,严格 hit-only 83.3%→96.7%。**详见 `report.md` A4.5 节 + `runs/a4_swap.jsonl`**。

**追加:数据污染对照(2026-05-24)**:补做 DeepSeek 闭卷实验(`a4_closed_llm.py`,不给任何 chunk,凭训练记忆答 30 题),严格 hit-only 仅 40%,准答率 66.7%。纯 RAG 贡献 = 96.7% - 40% = **+56.7 pp**。陷阱题 Q26→"阿银"、Q28→"玉小刚" 暴露后文知识渗透,Q2/Q16/Q17/Q18/Q22/Q23 闭卷瞎编机制证明记忆不可靠。结论:**斗罗大陆是头部 IP 训练数据污染存在,但 A4.5 实验的主导结论(LLM 换大模型显著提升 RAG 命中率)成立,污染不否定核心结论**。

**Track B 启示**:ASHRAE / WELL / EN PDF 这类标准文档**几乎不在 LLM 训练数据里**(版权 + 非公开),Track B 不需要做污染对照。但**Phase B 评估子图时如果用了 IEQ-Bench 上某些公开案例(论文 / FAQ),要在 eval/run.py 加 closed-book baseline 作 caveat**。

**A5 设计调整结论**:
- A5 不再追准答率,改盯三个新维度:**答案完整度**(Q13/Q14/Q15 多要素细节)、**列举覆盖度**(Q18 子查询拆解)、**延迟 / token 成本**(retry 累积)。
- A5 节点 LLM **按节点分配混合路由**(对齐 IEQ-Ops dissertation specialist 模式):decompose / grade / generate → DeepSeek-V4-Flash(关 thinking),rewrite → 本地 Qwen3-8B。理由:rewrite 是短字符串改写,本地够用;decompose/grade/generate 是 A4.5 证明本地不可靠的三类任务。这样 A5 就是 dissertation specialist 的 1:1 原型,可直接对比"混合 vs 全云端"的成本/延迟差。
- 检索层 Recall@5 = 96.6% 已经触顶,A5 改写 query 重检索的意义更多是练 LangGraph 工程模式,非检索增益。

**产出**:`a4_swap_llm.py` + `runs/a4_swap.jsonl` + `report.md` 新增 "LLM 消融" 节。

---

## A5 · Self-Reflective / Agentic RAG(2 天)

> **先决条件**:完成 A4.5,根据结果调整 A5 的 KPI 重心(详见上节)。
>
> 目标:把 RAG 从"一次性检索"升级为"会反思的循环"。**这里第一次引入 LangGraph。**

**流水线**

```
[query] → [decompose] → [retrieve] → [grade] → 够吗?
                                        ↓ 不够
                                    [rewrite] → 回到 retrieve(max_retries=2)
                                        ↓ 够
                                    [generate]
```

**节点职责 + LLM 分配**(混合路由,对齐 dissertation specialist)

| 节点 | LLM | 职责 |
|---|---|---|
| `decompose` | DeepSeek-V4-Flash | 判断要不要拆分。简单问题不拆,复杂问题(如"唐三与小舞感情线发展过程")拆 2-3 个子查询。 |
| `retrieve` | 无 LLM | 复用 A4 的 contextual + hybrid + rerank。 |
| `grade` | DeepSeek-V4-Flash | 看 chunks,输出 `{"sufficient": bool, "reason": str}`。**prompt 要写严格**,否则 LLM 会敷衍说"够了"。A4.5 已证本地 8B 会"自信地说够了"。 |
| `rewrite` | **本地 Qwen3-8B** | 改写 query。例:"唐三最后变成了什么" → "唐三成为海神的过程"。短字符串改写,本地够用。 |
| `generate` | DeepSeek-V4-Flash | 多轮累积 chunks → 最终答。A4.5 14pp hit-only 差距全在这一步。 |

退出条件:`max_retries=2`,防死循环。

**为什么混合而不全云端**:dissertation 的 specialist 内部就是这套混合路由,A5 是它的 1:1 原型。全云端虽然简单,但跳过了"两套 LLM client 共存 + 在 LangGraph 节点里切换"的工程模式,Phase B 第一次写就要在主图上调试。

**核心概念**(这些就是 Phase B 主图要用的同一套东西):

- `StateGraph` + state schema(TypedDict 或 Pydantic)
- 节点函数签名 `(state) -> state_update`
- 静态边 vs 条件边(`add_conditional_edges`)
- 循环 + 退出计数
- `MemorySaver` checkpointer(项目里换 `PostgresSaver`)

**步骤**

- [x] `uv add langgraph`  *(douluo/ 项目,langgraph 1.2.1 + checkpoint 4.1.1 + prebuilt 1.1.0 + sdk 0.3.15)*
- [x] 设计 state schema(query / sub_queries / chunks / retries / final_answer)  *(扩到 9 字段:`original_query` / `current_query` 分离,避免 rewrite 后丢原 query;新增 `reason`(grade→rewrite 改写线索)、`trace`(每节点 append 决策记录,verbose+jsonl 共用);`chunks` 和 `trace` 用 `Annotated[..., add]` 累加,`chunks` 去重逻辑放在 retrieve 节点内部)*
- [x] 写两个 LLM client wrapper:一个本地 Qwen3-8B(rewrite 节点用),一个 DeepSeek-V4-Flash(其他三个节点用)  *(class + `__call__` 让节点直接 `local(prompt)` / `cloud(prompt)` 调;自动累计 calls / tokens / cost / latency,A5 评测按节点拆本地 vs 云端;连通性实测 LocalLLM 24.9s/call,CloudLLM 0.75s/call & ¥0.000034/call,**33× 速度差**)*
- [x] 实现五个节点 + 条件边(注意 rewrite 单独走本地)  *3a: 5 个节点工厂 + 4 个 prompt 模板 mock 全过;3b: StateGraph 拼图(static edges + add_conditional_edges + MAX_RETRIES=2 强退)+ 简单题(4 节点,28.5s)和越界题(10 节点,174s,触发 2 次 rewrite)端到端 trace 验证通过*
- [x] 写 `agentic_rag.py`,CLI 入口  *三模式:`--query "..."` 单题(stream 边跑边打节点 trace + 最终答案 + 成本统计)/ `--eval [--output]` 30 题批量(写 jsonl + 每题 flush)/ `--mock [--llm-only]` 跑 3a 验收*
- [x] 跑 30 题 → `runs/a5.jsonl`  *耗时 28.8 min,Cloud 101 calls ¥0.0960(cache hit 32.8%),Local 11 calls 400s。retries 分布:0 retry 24 题(avg 33.9s)、1 retry 1 题(Q18)、2 retry 5 题(avg 159s,陷阱题 + Q10)。节点累计延迟:retrieve 70.5% + rewrite 23.2% 是两大头*
- [x] `report.md` 增加 A5 列、延迟、token、混合路由成本分解  *主对比表加 hit-only 列(86.7%);新增 A5 详细节(流水线图、三方对比、A4→A5 翻盘表、4 题退 partial 的两个独立副作用机制分析、按类型/retries 细分、节点延迟成本细账、LangGraph 工程价值总结)*
- [x] **重点测 A1-A4 答不对的复杂问题**,看 A5 能否扳回  *Q3/Q9/Q12 (A4 miss → A5 hit) + Q13/Q16 (A4 partial → A5 hit) 5 题全翻盘(跟 A4.5 完全一致,纯 LLM 升级红利);Q18 (A4.5 partial → A5 hit) 是唯一来自 decompose 拆 sub_queries 的 agentic 收益。代价:Q10/Q14/Q28/Q30 A4 hit → A5 partial(rewrite 多 chunks 或 decompose 拆解引入噪声稀释合成)*

**A5 验收结论**:hit-only 86.7%,**比 A4 +3.4 pp 但比 A4.5 -10 pp**。准答率不是 A5 的 KPI(早在 A4.5 已 100%),A5 真正交付的是 LangGraph 工程模式(StateGraph + 条件边 + 循环 + 强退 + 混合 LLM 路由 + trace 调试),这套结构直接接 Track B 主图。陷阱题 5/5 全拒答正确,grade 严判 + MAX_RETRIES=2 强退的护栏组合证明必要。Track B 进图前要先解决两个 A5 暴露的 agentic 坑:(1) rewrite 后只保留 top-K chunks 防稀释,(2) decompose 限制 sub_query 数量上限。

**这一步你要感受到的 tradeoff**:agentic RAG 慢、贵,但能答对原来答不对的题。这种直觉是以后做技术选型的核心资产。

**产出**:`agentic_rag.py` + `runs/a5.jsonl` + **完整对比报告 `report.md`**(5 阶段 × 20 题 × 命中率/延迟/token,这份报告以后写知乎或简历直接能用)。

---

## A 阶段结束清单

- [x] 5 个可独立运行的版本(A1-A5)  *naive_rag.py / hybrid_rag.py / rerank_rag.py / contextual_rag.py / agentic_rag.py,加 contextual_ingest.py / a4_swap_llm.py / a4_closed_llm.py 共 8 个入口脚本*
- [x] 完整对比实验报告 `report.md`  *主对比表 7 行(A1-A4 / A4.5 / A4.5-closed / A5),每阶段详细节带翻盘表 + 反面案例;主对比表带 hit-only 严格口径列*
- [ ] 对每一层 RAG 技术"为什么要加"有亲身感受  ← 这条用户自己打钩
- [x] 熟悉 LangGraph 的基本使用  *StateGraph + TypedDict + Annotated reducer + 静态边 + 条件边 + 循环 + 强退 + 节点闭包工厂 + stream/invoke 两种执行模式*

**到这里你已经具备实现 Phase A 的全部技术能力。** Track B 是把这些能力工程化落地。

---

# 桥接 · 从原型到项目(半天)

> 切换前问自己几个问题。如果答不上来,回去再翻 Track A 对应阶段。

- [ ] 原型里我用 `dict` 传 chunk,项目里要用 Pydantic Schema —— **为什么?**
      (答:类型安全 + 自动序列化 + Schema 演化)
- [ ] 原型里 prompt 写在 py 文件里,项目里要 `ops/prompts/` 版本化 —— **为什么?**
      (答:可回滚、可 A/B 测试、可被 IEQ-Bench 评测)
- [ ] 原型里我用 txt,项目里要用 PDF —— **为什么 PDF 解析需要 Docling 而不能 PyPDF?**
      (答:表格、公式、章节结构保留)
- [ ] 原型里我用 Ollama 本地跑,项目里要用 DashScope API —— **为什么?**
      (答:ingestion 阶段 contextual prefix 几千次调用,云端更快;Prompt Caching 摊薄成本;本地留给 monitoring loop)

四个问题都能口头答上来,就可以进 Track B。

---

# Track B · 项目线(rag/)

> 对应 `ieq-ops/rag/` 目录的 9 个文件,从根目录 `/home/xms/projects/rag/` 开始建。

---

## B0 · 工程基础设施(1 天)

**对应产出**:`pyproject.toml`、`docker-compose.yml`、`.env.example`

- [ ] 在 `/home/xms/projects/rag/` 根目录 `uv init`
- [ ] `uv add langchain langchain-core langgraph pydantic pydantic-settings structlog`
- [ ] `uv add --dev ruff pytest`
- [ ] 写 `docker-compose.yml`:

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
volumes:
  qdrant_data:
```

- [ ] `docker compose up -d`,浏览器打开 `http://localhost:6333/dashboard` 验证
- [ ] 配置 ruff(`pyproject.toml` 内 `[tool.ruff]`),约定 `uv run ruff check rag/` 必须 0 warning 才 commit
- [ ] 写 `.env.example`:

```
DASHSCOPE_API_KEY=
DEEPSEEK_API_KEY=
QDRANT_URL=http://localhost:6333
EMBED_DEVICE=cuda
EMBED_MODEL=BAAI/bge-m3
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

- [ ] 复制为 `.env` 并填真实 key(`.env` 不提交)
- [ ] 把 `.env` 加进 `.gitignore`

**验收**:`docker compose ps` 显示 qdrant 在跑;`uv run python -c "print('ok')"` 成功。

---

## B1 · Pydantic Schema(1 天)

**对应产出**:`rag/schemas.py`、`rag/config.py`

**`rag/schemas.py`** 至少三个核心类:

```python
from pydantic import BaseModel
from typing import Optional

class Chunk(BaseModel):
    chunk_id: str
    text: str
    page: int
    source_pdf: str
    section: Optional[str] = None    # 章节标题,Docling 能提取

class ContextualizedChunk(BaseModel):
    chunk: Chunk
    ctx_prefix: str                  # LLM 生成的上下文前缀
    embed_text: str                  # ctx_prefix + chunk.text,实际用于 embed 的文本

class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    source: str                      # "dense" / "sparse" / "fused" / "reranked"
```

**`rag/config.py`**:

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    qdrant_url: str = "http://localhost:6333"
    dashscope_api_key: str
    deepseek_api_key: str
    embed_device: str = "cuda"
    embed_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    class Config:
        env_file = ".env"

settings = Settings()
```

**步骤**

- [ ] 写 `rag/schemas.py`
- [ ] 写 `rag/config.py`
- [ ] 验证 `python -c "from rag.schemas import Chunk; print(Chunk(chunk_id='x', text='y', page=1, source_pdf='z').model_dump())"` 不报错
- [ ] 验证 `python -c "from rag.config import settings; print(settings.qdrant_url)"` 能读到 .env

**验收**:两个文件 import 不报错,字段类型清晰。

---

## B2 · PDF 解析 + 切块(1-2 天)

**对应产出**:`rag/parse.py`、`rag/chunk.py`

- [ ] `uv add docling`
- [ ] 写 `rag/parse.py`:

```python
from docling.document_converter import DocumentConverter

def parse_pdf(pdf_path: str):
    converter = DocumentConverter()
    return converter.convert(pdf_path).document
```

- [ ] 写 `rag/chunk.py`:

```python
from docling.chunking import HybridChunker
from rag.schemas import Chunk
import hashlib

def chunk_document(doc, source_pdf: str) -> list[Chunk]:
    chunker = HybridChunker(tokenizer="bert-base-uncased", max_tokens=512)
    chunks = []
    for c in chunker.chunk(doc):
        chunks.append(Chunk(
            chunk_id=hashlib.md5(c.text.encode()).hexdigest()[:12],
            text=c.text,
            page=c.meta.page if c.meta else 0,
            source_pdf=source_pdf,
            section=c.meta.headings[0] if c.meta and c.meta.headings else None
        ))
    return chunks
```

- [ ] 把至少一份 ASHRAE 62.1 PDF 放到 `rag/corpus/`
- [ ] 跑通 `parse → chunk`,打印前 5 个 chunk 的 text 和 section
- [ ] 验证表格内容(ASHRAE 的最小通风率 cfm/person 表)是否被保留

**为什么不用 PyPDF**:ASHRAE 标准里全是表格,PyPDF 解出来是乱码。Docling 保留表格结构、公式、章节层级。

**为什么用 HybridChunker**:沿 heading 边界切,不会把"Table 6-1"和它的内容切开。

**验收**:能打印出至少一个带 `section="..."` 字段、`page` 正确的 Chunk。

---

## B3 · Contextual Retrieval(1-2 天)

**对应产出**:`rag/contextualize.py`、`ops/prompts/contextualize/v1.md`

- [ ] `uv add openai`(DashScope / DeepSeek 都用兼容 SDK)
- [ ] 写 `ops/prompts/contextualize/v1.md`:

```
你是一名暖通空调标准专家。下面是 {source_pdf} 的全文,以及其中一个文本片段。
请用一两句话说明这个片段在整篇文档中的语境(属于哪一章节、讨论什么主题、关键术语是什么)。
直接输出前缀,不要任何寒暄。

【全文】
{full_text}

【片段】
{chunk_text}

【上下文前缀】
```

- [ ] 在 `rag/contextualize.py` 里:
  - 用 OpenAI SDK 创建 DashScope client(`base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"`)
  - **关键**:用 Prompt Caching 把整篇 PDF 作为固定前缀缓存,后续每个 chunk 请求只发 chunk + 指令。成本可降到 1/10 以下。具体调用方式见 DashScope Context Cache 或 DeepSeek Prompt Cache 文档。
  - **立即把 ContextualizedChunk 持久化到 SQLite/JSONL**,中间断了能续跑

```python
from rag.schemas import Chunk, ContextualizedChunk

def contextualize(chunks: list[Chunk], full_text: str) -> list[ContextualizedChunk]:
    result = []
    for chunk in chunks:
        ctx = call_llm_with_cache(full_text, chunk.text)   # 复用 cache
        result.append(ContextualizedChunk(
            chunk=chunk,
            ctx_prefix=ctx,
            embed_text=f"{ctx}\n\n{chunk.text}"
        ))
        # TODO: 这里要写盘,别等全跑完
    return result
```

- [ ] 先用一份 PDF 跑通,看输出质量
- [ ] 抽样 10 条 ctx_prefix,验证是否准确描述了所在章节和主题

**验收**:`rag/data/ctx_chunks.jsonl` 存在,行数等于 chunks 数;前缀有意义。

---

## B4 · Embedding + 向量索引(1-2 天)

**对应产出**:`rag/embed.py`、`rag/index_dense.py`

- [ ] `uv add FlagEmbedding qdrant-client`
- [ ] 写 `rag/embed.py`:

```python
from FlagEmbedding import BGEM3FlagModel

class Embedder:
    def __init__(self):
        self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device="cuda")

    def encode(self, texts: list[str]):
        return self.model.encode(
            texts,
            batch_size=4,
            return_dense=True,
            return_sparse=True,
        )
        # output["dense_vecs"]: shape (N, 1024)
        # output["lexical_weights"]: list of dict, {token_id: weight}
```

**关键点**:BGE-M3 一次前向同时产 dense + sparse,**一份编码两路索引**,省一半时间。

- [ ] 写 `rag/index_dense.py`(基于 Qdrant 客户端):

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

class DenseIndex:
    def __init__(self, settings):
        self.client = QdrantClient(settings.qdrant_url)
        self.collection = "standards_v1"

    def create(self):
        self.client.create_collection(
            self.collection,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )

    def upsert(self, ctx_chunks, dense_vecs):
        points = [
            PointStruct(
                id=i,
                vector=vec.tolist(),
                payload={
                    "chunk_id": c.chunk.chunk_id,
                    "text": c.chunk.text,
                    "ctx_prefix": c.ctx_prefix,
                    "source_pdf": c.chunk.source_pdf,
                    "page": c.chunk.page,
                }
            )
            for i, (c, vec) in enumerate(zip(ctx_chunks, dense_vecs))
        ]
        self.client.upsert(self.collection, points=points)

    def search(self, query_vec, limit=20):
        return self.client.search(self.collection, query_vec.tolist(), limit=limit)
```

- [ ] 写 10 个假向量进 Qdrant,搜索 top-3,打印分数和 payload

**验收**:Qdrant dashboard 能看到 collection `standards_v1`,内含点位。

---

## B5 · 稀疏索引 BM25(1 天)

**对应产出**:`rag/index_sparse.py`

- [ ] `uv add bm25s`
- [ ] 写 `rag/index_sparse.py`:

```python
import bm25s

class SparseIndex:
    def __init__(self):
        self.retriever = None
        self.chunk_ids = []

    def build(self, ctx_chunks):
        corpus = [c.embed_text for c in ctx_chunks]   # 用带前缀的文本
        self.chunk_ids = [c.chunk.chunk_id for c in ctx_chunks]
        self.retriever = bm25s.BM25()
        self.retriever.index(bm25s.tokenize(corpus))

    def search(self, query: str, k: int = 20):
        results, scores = self.retriever.retrieve(bm25s.tokenize([query]), k=k)
        return [(self.chunk_ids[i], s) for i, s in zip(results[0], scores[0])]

    def save(self, path: str):
        self.retriever.save(path)

    def load(self, path: str):
        self.retriever = bm25s.BM25.load(path)
```

**为什么 BM25 也要用带前缀的文本**:保持 dense 和 sparse 检索的语料一致,否则 RRF 融合时两边对的不是同一份文档。

- [ ] 用一小批 chunk 建索引,跑一个 query 验证 top-k 合理

**验收**:`rag/data/sparse_index` 文件生成,query 能返回 chunk_id + score。

---

## B6 · 融合 + 精排 + 检索入口(1 天)

**对应产出**:`rag/fuse.py`、`rag/rerank.py`、`rag/retrieve.py`

- [ ] `rag/fuse.py`(直接抄 A2 的 8 行 RRF)
- [ ] `rag/rerank.py`:

```python
from FlagEmbedding import FlagReranker

class Reranker:
    def __init__(self, settings):
        self.model = FlagReranker(settings.reranker_model, use_fp16=True)

    def rerank(self, query: str, chunks: list, top_k: int = 5):
        pairs = [[query, c.text] for c in chunks]
        scores = self.model.compute_score(pairs)
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
```

- [ ] `rag/retrieve.py`:

```python
class Retriever:
    def __init__(self, dense_index, sparse_index, embedder, reranker):
        self.dense = dense_index
        self.sparse = sparse_index
        self.embedder = embedder
        self.reranker = reranker

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        # 1. Dense
        q_vec = self.embedder.encode([query])["dense_vecs"][0]
        dense_hits = self.dense.search(q_vec, limit=20)
        dense_ids = [h.payload["chunk_id"] for h in dense_hits]

        # 2. Sparse
        sparse_hits = self.sparse.search(query, k=20)
        sparse_ids = [h[0] for h in sparse_hits]

        # 3. RRF 融合
        fused_ids = rrf(dense_ids, sparse_ids)[:30]

        # 4. 从 fused_ids 拼回完整 chunk
        candidates = self._fetch_chunks(fused_ids)

        # 5. Rerank
        reranked = self.reranker.rerank(query, candidates, top_k=top_k)

        return [
            RetrievedChunk(chunk=c, score=s, source="reranked")
            for c, s in reranked
        ]

    @classmethod
    def from_config(cls):
        # 工厂方法:从 settings 加载所有组件
        ...
```

- [ ] 写一个简短 demo,从 .env / Qdrant 加载,跑一个 query,打印 top-5 + 分数

**验收**:`from rag.retrieve import Retriever; r = Retriever.from_config(); r.retrieve("...", top_k=5)` 能返回 5 个有 page / source_pdf / score 的结果。

---

## B7 · Ingestion 全链路打通(1 天)

**对应产出**:`rag/ingest.py`

```python
# python -m rag.ingest --corpus rag/corpus/ --limit 1
import click
from rag.parse import parse_pdf
from rag.chunk import chunk_document
from rag.contextualize import contextualize
from rag.embed import Embedder
from rag.index_dense import DenseIndex
from rag.index_sparse import SparseIndex

@click.command()
@click.option("--corpus", required=True)
@click.option("--limit", default=None, type=int)
def ingest(corpus, limit):
    pdfs = list_pdfs(corpus)[:limit] if limit else list_pdfs(corpus)

    embedder = Embedder()
    dense = DenseIndex(settings)
    sparse = SparseIndex()
    all_ctx_chunks = []

    for pdf in pdfs:
        doc = parse_pdf(pdf)
        chunks = chunk_document(doc, source_pdf=pdf)
        full_text = doc.export_to_markdown()
        ctx_chunks = contextualize(chunks, full_text)
        all_ctx_chunks.extend(ctx_chunks)

    texts = [c.embed_text for c in all_ctx_chunks]
    output = embedder.encode(texts)

    dense.create()
    dense.upsert(all_ctx_chunks, output["dense_vecs"])
    sparse.build(all_ctx_chunks)
    sparse.save("rag/data/sparse_index")

if __name__ == "__main__":
    ingest()
```

**步骤**

- [ ] `uv add click`
- [ ] 写 `rag/ingest.py`
- [ ] 跑 `python -m rag.ingest --corpus rag/corpus/ --limit 1`
- [ ] Qdrant dashboard 验证 chunks 进库
- [ ] `rag/data/sparse_index` 文件存在

**验收**:单 PDF 全链路跑通,无错误。

---

## B8 · 评测 + pytest(1 天)

**对应产出**:`rag/eval/metrics.py`、`rag/eval/run.py`、`rag/eval/gold.jsonl`、`rag/tests/`

- [ ] `uv add ranx`
- [ ] 写 `rag/eval/gold.jsonl`,30-50 条标注:

```jsonl
{"query": "ASHRAE 62.1 minimum outdoor air rate for office", "relevant_chunks": ["chunk_abc123", "chunk_def456"]}
{"query": "WELL standard CO2 threshold", "relevant_chunks": ["chunk_xyz789"]}
```

- [ ] 写 `rag/eval/metrics.py`,用 ranx 算 Recall@K / MRR / nDCG@10
- [ ] 写 `rag/eval/run.py`,加载 Retriever,对 gold 全跑一遍,输出指标
- [ ] 写 `rag/tests/test_fuse.py`(纯逻辑测试):

```python
def test_rrf_both_lists():
    from rag.fuse import rrf
    dense = ["a", "b", "c"]
    sparse = ["b", "a", "d"]
    result = rrf(dense, sparse)
    assert result.index("a") < result.index("c")
    assert result.index("b") < result.index("d")

def test_rrf_no_duplicates():
    result = rrf(["a", "b"], ["a", "c"])
    assert result.count("a") == 1
```

- [ ] 写 `rag/tests/test_schemas.py`,测 Chunk / ContextualizedChunk 的字段约束

**硬约束(来自 CLAUDE.md)**:**不要测 prompt**,prompt 的好坏交给 IEQ-Bench。pytest 只测纯逻辑(RRF、Pydantic schema、tokenize 等)。

**最终验证清单**:

- [ ] `docker compose up -d qdrant` 启动
- [ ] `uv sync` 依赖完整
- [ ] `python -m rag.ingest --corpus rag/corpus/ --limit 1` 通过
- [ ] `python -c "from rag.retrieve import Retriever; r = Retriever.from_config(); print(r.retrieve('ASHRAE minimum outdoor air rate for office', top_k=5))"` 通过
- [ ] `python -m rag.eval.run --gold rag/eval/gold.jsonl` 输出 Recall@5 / MRR / nDCG@10
- [ ] `pytest rag/tests/` 全部通过
- [ ] `uv run ruff check rag/` 0 warning

**全部 ✓ 即 Phase A 完工。**

---

# 阶段 7 · MCP RAG Server 封装(1-2 天,强烈推荐)

**对应产出**:`mcp_servers/rag/server.py`

- [ ] `uv add fastmcp`
- [ ] 写 `mcp_servers/rag/server.py`:

```python
from fastmcp import FastMCP
from rag.retrieve import Retriever
from rag.schemas import RetrievedChunk

mcp = FastMCP("rag-server")
retriever = Retriever.from_config()

@mcp.tool()
def search_standards(query: str, top_k: int = 5) -> list[dict]:
    """
    Search HVAC / IEQ standards (ASHRAE, WELL, EN, WHO) for relevant clauses.
    Use this when you need authoritative reference on ventilation rates, thermal
    comfort thresholds, lighting levels, or acoustic standards.

    Args:
        query: Natural language query about HVAC/IEQ standards.
        top_k: Number of results to return (default 5).

    Returns:
        List of dicts with text, source_pdf, page, score.
    """
    results = retriever.retrieve(query, top_k=top_k)
    return [r.model_dump() for r in results]

if __name__ == "__main__":
    mcp.run()
```

**理解要点**

- MCP 的 stdio vs SSE transport
- tool docstring 是 LLM 选 tool 的依据(写得越准 specialist 越能选对)
- 为什么生产系统把 RAG 做成独立 server(进程隔离、独立扩缩容、可复用)

**步骤**

- [ ] 本地用 stdio transport 起服务
- [ ] 挂到 Claude Desktop 或 Cursor,用自然语言问 ASHRAE 标准
- [ ] 验证 specialist agent 能用上这个 tool 并返回有意义的 chunk

**验收**:从 LLM 客户端发出"What is ASHRAE 62.1 minimum outdoor air rate for office?"能得到带 page / source 的引用回答。

---

# 收尾

- [ ] 知乎博文一篇:基于 Track A 对比报告 + Track B 工程化经验
- [ ] 更新 LinkedIn / 简历:把"contextual + hybrid + agentic RAG 系统"作为项目条目

---

# 最终能力对照

完工后你具备:

- 独立实现生产级 contextual + hybrid + agentic RAG
- 熟悉 Pydantic / FastMCP / LangGraph / Qdrant / BGE 全栈
- 可量化评测体系(Recall / MRR / nDCG)
- 公开可发表的对比实验报告
- 直接进入 Phase B(IEQ-Ops 主图编排)的工程能力

Phase B 的 Self-Reflective Agentic RAG 子图就是把 A5 的代码搬过来,套到 B 的 Retriever 上 —— 这就是为什么这条路线值得走。
