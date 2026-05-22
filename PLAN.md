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

- [ ] 在 `/home/xms/projects/rag/douluo/` 下 `uv init` 建独立项目
- [ ] `uv add langchain langchain-community langchain-ollama sentence-transformers faiss-cpu`
- [ ] `ollama pull qwen3:8b` 并 `ollama run qwen3:8b` 验证能对话
- [ ] 把 `《斗罗大陆》_qinkan.net.txt` 编码统一成 UTF-8
- [ ] 清洗:删除盗版水印、"本章未完点击下一页"等噪声行
- [ ] 按章节切分到 `corpus/douluo/ch001.txt`...,**只抽前 5-10 章先用**(全本迭代太慢)
- [ ] 建 `golden_questions.json`,写 20-30 个问题,五类各占一些:
  - 事实型:"唐三的第一武魂是什么"(蓝银草)
  - 关系型:"小舞和唐三是什么关系"
  - 跨章节型:"唐三获得了哪些魂环"
  - 模糊型:"海神九考都有哪些"
  - 陷阱题(原文没有):"唐三在霍格沃茨学了什么"——测 hallucination
- [ ] 每题手工标注期望答案(JSON 里加 `expected` 字段)

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

- [ ] `uv add` 还没装的依赖
- [ ] 写 `naive_rag.py`:CLI 接收 `--query` 参数,跑完整流水线返回答案
- [ ] 跑 20 个金标问题,把结果写到 `runs/a1.jsonl`
- [ ] 手工标注每题:命中 / 部分命中 / 没命中 / hallucinate
- [ ] 在 `report.md` 起一张表(后面 A2-A5 同列累加)

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

- [ ] `uv add rank-bm25`
- [ ] 复制 `naive_rag.py` → `hybrid_rag.py`
- [ ] 加 BM25 索引,query 时同时跑 dense top-20 和 sparse top-20
- [ ] 实现 RRF 融合(8 行,直接抄):

```python
def rrf(dense_ids, sparse_ids, k=60):
    scores = {}
    for rank, doc_id in enumerate(dense_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    for rank, doc_id in enumerate(sparse_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)
```

- [ ] 融合后取 top-5,拼 prompt,跑 Qwen3-8B
- [ ] 跑同一批 20 题 → `runs/a2.jsonl`
- [ ] `report.md` 增加 A2 列,对比 A1

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

- [ ] 复制 `hybrid_rag.py` → `rerank_rag.py`
- [ ] 加载 `bge-reranker-v2-m3`(CPU)
- [ ] RRF 后取 top-30,送 reranker 重排,取 top-5
- [ ] 跑同一批 20 题 → `runs/a3.jsonl`,**额外记录每题延迟**
- [ ] `report.md` 增加 A3 列 + 延迟列

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

- [ ] 写 `contextual_ingest.py`:对每个 chunk 生成 ctx_prefix,缓存到 `cache/ctx.sqlite`
- [ ] 先只跑前 5 章,看人工抽查 10 条前缀质量是否过关
- [ ] 质量 OK 再全量跑
- [ ] 写 `contextual_rag.py`:加载缓存,用 `ctx_prefix + chunk` 重新建 dense 和 sparse 索引,query 流程复用 A3
- [ ] 跑 20 题 → `runs/a4.jsonl`,重点看跨章节、需要上下文判断的问题
- [ ] `report.md` 增加 A4 列

**直觉验证**:一段被切出来的对话原本只有"他举起了那把锤子",加了前缀变成"在与昊天宗的冲突中,唐三举起了那把锤子"——召回率应该有质变。

**产出**:`contextual_ingest.py` + `contextual_rag.py` + `cache/ctx.sqlite` + `report.md` 更新。

---

## A5 · Self-Reflective / Agentic RAG(2 天)

> 目标:把 RAG 从"一次性检索"升级为"会反思的循环"。**这里第一次引入 LangGraph。**

**流水线**

```
[query] → [decompose] → [retrieve] → [grade] → 够吗?
                                        ↓ 不够
                                    [rewrite] → 回到 retrieve(max_retries=2)
                                        ↓ 够
                                    [generate]
```

**节点职责**

- `decompose`:LLM 判断要不要拆分。简单问题不拆,复杂问题(如"唐三与小舞感情线发展过程")拆 2-3 个子查询。
- `retrieve`:复用 A4 的 contextual + hybrid + rerank。
- `grade`:LLM 看 chunks,输出 `{"sufficient": bool, "reason": str}`。**prompt 要写严格**,否则 LLM 会敷衍说"够了"。
- `rewrite`:改写 query。例:"唐三最后变成了什么" → "唐三成为海神的过程"。
- 退出条件:`max_retries=2`,防死循环。

**核心概念**(这些就是 Phase B 主图要用的同一套东西):

- `StateGraph` + state schema(TypedDict 或 Pydantic)
- 节点函数签名 `(state) -> state_update`
- 静态边 vs 条件边(`add_conditional_edges`)
- 循环 + 退出计数
- `MemorySaver` checkpointer(项目里换 `PostgresSaver`)

**步骤**

- [ ] `uv add langgraph`
- [ ] 设计 state schema(query / sub_queries / chunks / retries / final_answer)
- [ ] 实现四个节点 + 条件边
- [ ] 写 `agentic_rag.py`,CLI 入口
- [ ] 跑 20 题 → `runs/a5.jsonl`,**额外记录**:平均 retry 次数、总延迟、总 token 消耗
- [ ] `report.md` 增加 A5 列、延迟、token 三列
- [ ] **重点测 A1-A4 答不对的复杂问题**,看 A5 能否扳回

**这一步你要感受到的 tradeoff**:agentic RAG 慢、贵,但能答对原来答不对的题。这种直觉是以后做技术选型的核心资产。

**产出**:`agentic_rag.py` + `runs/a5.jsonl` + **完整对比报告 `report.md`**(5 阶段 × 20 题 × 命中率/延迟/token,这份报告以后写知乎或简历直接能用)。

---

## A 阶段结束清单

- [ ] 5 个可独立运行的版本(A1-A5)
- [ ] 完整对比实验报告 `report.md`
- [ ] 对每一层 RAG 技术"为什么要加"有亲身感受
- [ ] 熟悉 LangGraph 的基本使用

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
