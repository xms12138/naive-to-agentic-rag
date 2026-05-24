"""A5 Self-Reflective / Agentic RAG —— 在 A4 检索栈上加 LangGraph 反思循环。

流水线:
    [query]
      → decompose (DeepSeek)    拆 sub_queries(简单题不拆,直接 [original_query])
      → retrieve  (无 LLM)        for sub_query in sub_queries: 复用 A4 hybrid+rerank;
                                  累加到 state.chunks 并按 (source, chunk_idx) 去重
      → grade     (DeepSeek)    看 query + chunks,判 {sufficient: bool, reason: str}
      → 条件边     sufficient or retries >= MAX_RETRIES → generate
                  否则           → rewrite
      → rewrite   (本地 Qwen3-8B) 把 current_query 改写成 new query,retries +1
                                  → 回 retrieve(循环)
      → generate  (DeepSeek)    用累计的全部 chunks 合成最终答案 + 对比纯模型不联网答案

混合 LLM 路由(对齐 dissertation specialist 模式):
    decompose / grade / generate → DeepSeek-V4-Flash(关 thinking,A4.5 已证本地 8B
                                    在 grade 和 generate 上不可靠,14pp hit-only 差距)
    rewrite                      → 本地 Qwen3-8B(短字符串改写,本地够用 + 练混合切换)
    retrieve                     → 无 LLM,纯检索栈

为什么需要 trace 字段:
    反思循环最难调试的是"为什么绕了 2 轮还失败"。每个节点跑完往 trace append 一条记录,
    verbose 模式直出整条决策路径,a5.jsonl 也能事后回放。
"""

import argparse
import json
import os
import re
import sys
import time
from operator import add
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from openai import OpenAI
from sentence_transformers import CrossEncoder

from contextual_rag import (
    FINAL_TOP_K,
    GOLDEN_PATH,
    RERANKER_MODEL,
    build_indexes,
    fuse_to_candidates,
    load_chunks_with_prefix,
)
from contextual_rag import rerank as cross_encoder_rerank


# ============================================================
# State Schema
# ============================================================

class AgentState(TypedDict):
    """LangGraph 共享状态。每个字段的更新策略:
      - 普通字段:节点返回时覆盖
      - Annotated[..., add]:节点返回时 append(用于累加)

    字段语义见下方注释。
    """

    # ─── 输入 ───
    original_query: str
    """用户最初的 query,全程不变(报告/调试时对照用)。"""

    # ─── decompose 节点产出 ───
    sub_queries: list[str]
    """拆出的子查询;不拆则 [original_query]。"""

    # ─── retrieve 节点产出(累加 + 节点内去重)───
    chunks: Annotated[list[dict], add]
    """多轮检索累计的 chunks。dict 形状对齐 contextual_rag.fuse_to_candidates 输出,
    额外加 retrieve_round / from_subquery 字段方便溯源。
    去重逻辑放在 retrieve 节点内部(按 source + chunk_idx),不依赖 reducer。"""

    # ─── grade 节点产出 ───
    sufficient: bool
    """当前累计的 chunks 够不够答 original_query。"""
    reason: str
    """grade 给的"不够"原因 → 传给 rewrite 当改写线索。够时为 ""。"""

    # ─── rewrite 节点产出 ───
    current_query: str
    """当前正用于 retrieve 的 query。初始 = original_query;rewrite 后被覆盖。"""
    retries: int
    """已 rewrite 的次数。达 MAX_RETRIES=2 时条件边强制走 generate。"""

    # ─── generate 节点产出 ───
    base_llm_answer: str
    """DeepSeek v4-flash 不联网的直接答案（纯模型知识库基线）。"""
    final_answer: str
    """最终组装的对比答案。"""

    # ─── 追溯字段(调试/报告)───
    trace: Annotated[list[dict], add]
    """每节点跑完 append 一条 {"node": str, ...节点特定字段}。verbose 和 jsonl 都用。"""


# ============================================================
# 常量
# ============================================================

MAX_RETRIES = 2  # rewrite 上限,达到后条件边强制 generate

# 本地 LLM(rewrite 节点用)
LOCAL_BASE_URL = "http://localhost:11434/v1/"
LOCAL_MODEL = "qwen3:8b"

# 云端 LLM(decompose / grade / generate 用)默认值,可被 .env 覆盖
DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-flash"


# ============================================================
# LLM Wrapper —— 两个 client,带累计统计
# ============================================================

class LocalLLM:
    """Qwen3-8B via Ollama(OpenAI 兼容协议)。给 rewrite 节点用。"""

    def __init__(self, base_url: str = LOCAL_BASE_URL, model: str = LOCAL_MODEL):
        self.client = OpenAI(base_url=base_url, api_key="ollama")
        self.model = model
        self.calls = 0
        self.total_latency = 0.0

    def __call__(self, prompt: str) -> str:
        t0 = time.time()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt + " /no_think"}],
        )
        self.calls += 1
        self.total_latency += time.time() - t0
        return (resp.choices[0].message.content or "").strip()


class CloudLLM:
    """DeepSeek-V4-Flash(OpenAI 兼容)。给 decompose / grade / generate 节点用。"""

    def __init__(self):
        load_dotenv(Path(__file__).parent / ".env")
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            sys.exit("ERROR: DEEPSEEK_API_KEY 未在 .env 中设置")
        self.client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", DEEPSEEK_DEFAULT_BASE_URL),
        )
        self.model = os.getenv("DEEPSEEK_MODEL", DEEPSEEK_DEFAULT_MODEL)
        self.calls = 0
        self.total_in = 0     # prompt_tokens(含 cache hit)
        self.total_out = 0    # completion_tokens
        self.total_hit = 0    # prompt_cache_hit_tokens
        self.total_latency = 0.0

    def __call__(
        self,
        prompt: str,
        *,
        max_tokens: int = 400,
        temperature: float = 0.2,
    ) -> str:
        t0 = time.time()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={"thinking": {"type": "disabled"}},
        )
        latency = time.time() - t0
        usage = resp.usage.model_dump() if resp.usage else {}
        self.calls += 1
        self.total_in += usage.get("prompt_tokens", 0) or 0
        self.total_out += usage.get("completion_tokens", 0) or 0
        self.total_hit += usage.get("prompt_cache_hit_tokens", 0) or 0
        self.total_latency += latency
        return (resp.choices[0].message.content or "").strip()

    def cost_yuan(self) -> float:
        """V4-flash 计价(元 / M tokens):cache_hit 0.02 / input miss 1.0 / output 2.0"""
        miss = max(self.total_in - self.total_hit, 0)
        return (
            self.total_hit * 0.02
            + miss * 1.0
            + self.total_out * 2.0
        ) / 1_000_000


# ============================================================
# Prompt 模板
# ============================================================

DECOMPOSE_PROMPT = """你的任务是判断一个关于《斗罗大陆》小说的问题是否需要拆分成多个子查询来检索原文。

判断规则:
1. 简单事实型问题(如"唐三的武魂是什么"、"小舞和唐三是什么关系")不拆,sub_queries 只放原问题本身。
2. 含列举词(几个、哪些、九大、所有、都有)、多要素(成长过程、感情发展)、对比类的复杂问题,拆成 2-3 个具体的子查询,每个子查询单独可检索。

只输出 JSON,不要任何其他文字、不要 markdown code fence:
{{"sub_queries": ["...", "..."]}}

原问题:{query}

JSON:"""


GRADE_PROMPT = """你的任务是严格判断给定的原文片段能否完整、准确地回答用户问题。

判断标准:
1. 缺关键信息(人名、地名、事件名、数量、时间等)→ sufficient=false
2. 列举题只覆盖一部分(如问"N 个 X"只找到 M<N 个)→ sufficient=false,reason 中指出缺什么
3. 片段明确、完整地包含答案所有要素 → sufficient=true

宁可严格不要宽松——sufficient=true 后将停止检索直接合成答案,漏要素无法挽回。

只输出 JSON,不要任何其他文字、不要 markdown code fence:
{{"sufficient": true/false, "reason": "..."}}

reason 字段:如果 sufficient=false,具体说明缺失了哪些信息(后续会用这条线索改写 query 重检索)。如果 sufficient=true,reason 留空字符串。

问题:{query}

【已检索片段】
{context}

JSON:"""


REWRITE_PROMPT = """原查询:{query}
该查询检索后被判定信息不足,具体原因:{reason}

请改写一个新的查询,使其更可能检索到上述缺失的信息。
要求:只输出新查询本身一行,不要解释、不要引号、不要"新查询:"之类的前缀。

新查询:"""


GENERATE_PROMPT = """你是一个《斗罗大陆》小说知识助手。我会给你从小说原文中检索到的若干片段,请基于这些片段回答用户的问题。

规则:
1. 只能根据提供的原文片段作答,不要使用你自己的知识,也不要编造内容。
2. 如果片段中没有相关信息,直接回答"原文未提及"。
3. 回答简洁、准确,不需要复述原文。

【检索到的原文片段】
{context}

【问题】
{question}

【回答】"""


# ============================================================
# JSON 解析辅助
# ============================================================

def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON in LLM response: {text[:200]!r}")
    return json.loads(m.group(0))


def _clean_rewrite_output(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^(新查询|新的查询|新 query)[::]\s*", "", text)
    text = text.strip("\"'“”‘’")
    return text.strip()


# ============================================================
# 节点工厂
# ============================================================

def make_decompose_node(cloud: "CloudLLM"):
    def decompose(state: AgentState) -> dict:
        t0 = time.time()
        query = state["original_query"]
        prompt = DECOMPOSE_PROMPT.format(query=query)
        resp = cloud(prompt, max_tokens=300, temperature=0.0)

        try:
            data = _extract_json(resp)
            sub_queries = data.get("sub_queries") or [query]
            if not isinstance(sub_queries, list) or not all(isinstance(s, str) and s.strip() for s in sub_queries):
                sub_queries = [query]
        except Exception:
            sub_queries = [query]

        latency = round(time.time() - t0, 2)
        return {
            "sub_queries": sub_queries,
            "current_query": query,
            "retries": 0,
            "trace": [{
                "node": "decompose",
                "latency_sec": latency,
                "sub_queries": sub_queries,
                "n_sub": len(sub_queries),
                "raw_response_preview": resp[:150],
            }],
        }
    return decompose


def make_retrieve_node(resources: dict):
    chunks_all = resources["chunks"]
    embedder = resources["embedder"]
    faiss_index = resources["faiss_index"]
    bm25 = resources["bm25"]
    reranker = resources["reranker"]

    def retrieve(state: AgentState) -> dict:
        t0 = time.time()
        round_num = state.get("retries", 0)
        if round_num == 0:
            queries = state["sub_queries"]
        else:
            queries = [state["current_query"]]

        seen = {(c["source"], c["chunk_idx"]) for c in (state.get("chunks") or [])}
        new_chunks: list[dict] = []
        for q in queries:
            candidates, _, _ = fuse_to_candidates(q, embedder, faiss_index, bm25, chunks_all)
            top, _ = cross_encoder_rerank(q, candidates, reranker, final_k=FINAL_TOP_K)
            for c in top:
                key = (c["source"], c["chunk_idx"])
                if key in seen:
                    continue
                seen.add(key)
                c_copy = dict(c)
                c_copy["retrieve_round"] = round_num
                c_copy["from_subquery"] = q
                new_chunks.append(c_copy)

        latency = round(time.time() - t0, 2)
        return {
            "chunks": new_chunks,
            "trace": [{
                "node": "retrieve",
                "round": round_num,
                "queries": queries,
                "new_chunks": len(new_chunks),
                "total_chunks_after": len(seen),
                "latency_sec": latency,
            }],
        }
    return retrieve


def make_grade_node(cloud: "CloudLLM"):
    def grade(state: AgentState) -> dict:
        t0 = time.time()
        chunks = state.get("chunks") or []
        context = (
            "\n\n---\n\n".join(f"【片段 {i + 1}】{c['text']}" for i, c in enumerate(chunks))
            if chunks else "(无)"
        )
        prompt = GRADE_PROMPT.format(query=state["original_query"], context=context)
        resp = cloud(prompt, max_tokens=300, temperature=0.0)

        try:
            data = _extract_json(resp)
            sufficient = bool(data.get("sufficient", True))
            reason = str(data.get("reason", "") or "")
        except Exception as e:
            sufficient, reason = True, f"(grade parse failed: {e})"

        latency = round(time.time() - t0, 2)
        return {
            "sufficient": sufficient,
            "reason": reason,
            "trace": [{
                "node": "grade",
                "sufficient": sufficient,
                "reason": reason,
                "n_chunks_seen": len(chunks),
                "latency_sec": latency,
            }],
        }
    return grade


def make_rewrite_node(local: "LocalLLM"):
    def rewrite(state: AgentState) -> dict:
        t0 = time.time()
        old_query = state.get("current_query") or state["original_query"]
        reason = state.get("reason", "") or "(无)"
        prompt = REWRITE_PROMPT.format(query=old_query, reason=reason)
        raw = local(prompt)
        new_query = _clean_rewrite_output(raw)
        if not new_query:
            new_query = old_query

        latency = round(time.time() - t0, 2)
        return {
            "current_query": new_query,
            "retries": state.get("retries", 0) + 1,
            "trace": [{
                "node": "rewrite",
                "old_query": old_query,
                "new_query": new_query,
                "raw_preview": raw[:150],
                "latency_sec": latency,
            }],
        }
    return rewrite


def make_generate_node(cloud: "CloudLLM"):
    """generate 节点:用累计 chunks 合成最终答案，同时获取不联网基线答案。

    state 读: original_query, chunks
    state 写: final_answer, base_llm_answer, trace
    """
    def generate(state: AgentState) -> dict:
        t0 = time.time()
        query = state["original_query"]
        chunks = state.get("chunks") or []
        
        # 1. 获取纯模型不联网原始答案 (Baseline)
        base_prompt = f"请直接回答下面的问题，如果不知道请说不知道。\n\n【问题】\n{query}\n\n【回答】"
        base_llm_answer = cloud(base_prompt, max_tokens=400, temperature=0.2)

        # 2. 生成基于 RAG 检索片段的答案
        context = (
            "\n\n---\n\n".join(
                f"【片段 {i + 1} · 来自 {c['source']}】\n{c['text']}"
                for i, c in enumerate(chunks)
            )
            if chunks else "(无)"
        )
        rag_prompt = GENERATE_PROMPT.format(context=context, question=query)
        rag_answer = cloud(rag_prompt, max_tokens=600, temperature=0.2)

        # 3. 拼接输出格式，直观呈现在控制台
        combined_final_answer = (
            f"【DeepSeek v4-flash 纯模型不联网答案】\n{base_llm_answer}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"【Agentic RAG 最终整合答案】\n{rag_answer}"
        )

        latency = round(time.time() - t0, 2)
        return {
            "base_llm_answer": base_llm_answer,
            "final_answer": combined_final_answer,
            "trace": [{
                "node": "generate",
                "n_chunks": len(chunks),
                "latency_sec": latency,
            }],
        }
    return generate


# ============================================================
# 检索资源(一次性加载,给 retrieve 节点用)
# ============================================================

def init_retrieval_resources() -> dict:
    print("[init] 加载语料 + ctx prefix...")
    chunks = load_chunks_with_prefix()
    print(f"[init] {len(chunks)} chunks loaded")

    print("[init] 构建 FAISS + BM25...")
    embedder, faiss_index, bm25, _ = build_indexes(chunks, verbose=False)
    print(f"[init] FAISS ntotal={faiss_index.ntotal}, BM25 corpus={bm25.corpus_size}")

    print(f"[init] 加载 reranker {RERANKER_MODEL} (CPU)...")
    t0 = time.time()
    reranker = CrossEncoder(RERANKER_MODEL, device="cpu")
    print(f"[init] reranker ready ({time.time() - t0:.1f}s)")

    return {
        "chunks": chunks,
        "embedder": embedder,
        "faiss_index": faiss_index,
        "bm25": bm25,
        "reranker": reranker,
    }


# ============================================================
# LangGraph 拼图
# ============================================================

def route_after_grade(state: AgentState) -> str:
    if state["sufficient"]:
        return "generate"
    if state.get("retries", 0) >= MAX_RETRIES:
        return "generate"
    return "rewrite"


def build_graph(cloud: "CloudLLM", local: "LocalLLM", resources: dict):
    graph = StateGraph(AgentState)
    graph.add_node("decompose", make_decompose_node(cloud))
    graph.add_node("retrieve",  make_retrieve_node(resources))
    graph.add_node("grade",     make_grade_node(cloud))
    graph.add_node("rewrite",   make_rewrite_node(local))
    graph.add_node("generate",  make_generate_node(cloud))

    graph.add_edge(START, "decompose")
    graph.add_edge("decompose", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {"rewrite": "rewrite", "generate": "generate"},
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)

    return graph.compile()


def initial_state(query: str) -> dict:
    return {
        "original_query": query,
        "current_query": query,
        "sub_queries": [],
        "chunks": [],
        "sufficient": False,
        "reason": "",
        "retries": 0,
        "base_llm_answer": "",
        "final_answer": "",
        "trace": [],
    }


# ============================================================
# Mock 测试
# ============================================================

def _section(title: str) -> None:
    print("\n" + "─" * 72)
    print(f"  {title}")
    print("─" * 72)


def _pretty(d: dict) -> None:
    print(json.dumps(d, ensure_ascii=False, indent=2))


def test_llm_nodes() -> None:
    cloud = CloudLLM()
    local = LocalLLM()

    decompose = make_decompose_node(cloud)
    grade = make_grade_node(cloud)
    rewrite = make_rewrite_node(local)
    generate = make_generate_node(cloud)

    _section("Test 1 · decompose 简单题(期望 sub_queries=[原 query])")
    out = decompose({"original_query": "唐三的第一武魂是什么"})
    _pretty(out)

    _section("Test 2 · decompose 复杂题(期望 sub_queries 拆 2-3 条)")
    out = decompose({"original_query": "海神九考都有哪些"})
    _pretty(out)

    _section("Test 3 · grade 信息不足(期望 sufficient=false 且 reason 非空)")
    fake = {
        "original_query": "唐三的第二武魂是什么",
        "chunks": [{"text": "唐三的第一武魂是蓝银草。", "source": "ch001", "chunk_idx": 0}],
    }
    out = grade(fake)
    _pretty(out)

    _section("Test 4 · grade 信息充足(期望 sufficient=true)")
    fake = {
        "original_query": "唐三的第一武魂是什么",
        "chunks": [{"text": "唐三六岁时,武魂觉醒为蓝银草,这是一种垃圾武魂。", "source": "ch002", "chunk_idx": 5}],
    }
    out = grade(fake)
    _pretty(out)

    _section("Test 5 · rewrite(本地 Qwen3-8B)")
    fake = {
        "current_query": "唐三的武魂",
        "reason": "缺少'第二武魂'的具体名称",
        "retries": 0,
    }
    out = rewrite(fake)
    _pretty(out)

    _section("Test 6 · generate(期望基于 chunk 合成答案 + 输出不联网原始答案)")
    fake = {
        "original_query": "唐三的第一武魂是什么",
        "chunks": [{"text": "唐三六岁时,武魂觉醒为蓝银草,这是一种垃圾武魂。", "source": "ch002", "chunk_idx": 5}],
    }
    out = generate(fake)
    _pretty(out)

    _section("LLM 调用统计")
    print(f"  Cloud  calls={cloud.calls}  in={cloud.total_in}  out={cloud.total_out}  "
          f"hit={cloud.total_hit}  latency={cloud.total_latency:.1f}s  cost=¥{cloud.cost_yuan():.4f}")
    print(f"  Local  calls={local.calls}  latency={local.total_latency:.1f}s")


def test_retrieve_node() -> None:
    _section("Test 7 · retrieve 节点(加载 FAISS + BM25 + reranker)")
    resources = init_retrieval_resources()
    retrieve = make_retrieve_node(resources)

    state = {
        "original_query": "唐三的第一武魂是什么",
        "sub_queries": ["唐三的第一武魂是什么"],
        "current_query": "唐三的第一武魂是什么",
        "retries": 0,
        "chunks": [],
    }
    out = retrieve(state)
    print(f"\n[round 0] 新增 {len(out['chunks'])} chunks")
    for i, c in enumerate(out["chunks"][:3]):
        preview = c["text"][:60].replace("\n", " ")
        print(f"  [{i + 1}] {c['source']}#{c['chunk_idx']:02d}  round={c['retrieve_round']}  {preview}...")


# ============================================================
# 单题 verbose 模式
# ============================================================

def _print_trace_entry(entry: dict) -> None:
    node = entry["node"]
    lat = entry.get("latency_sec", 0)
    head = f"\n  ━━━ [{node:<9}] {lat:>5.2f}s ━━━"
    print(head)

    if node == "decompose":
        n_sub = entry.get("n_sub", len(entry.get("sub_queries", [])))
        print(f"    sub_queries ({n_sub}): {entry['sub_queries']}")
    elif node == "retrieve":
        print(f"    round={entry['round']}  new_chunks={entry['new_chunks']}  "
              f"total_chunks_after={entry['total_chunks_after']}")
        for q in entry["queries"]:
            print(f"    · query: {q}")
    elif node == "grade":
        status = "✓ 够" if entry["sufficient"] else "✗ 不够"
        print(f"    {status}  n_chunks_seen={entry['n_chunks_seen']}")
        if entry["reason"]:
            print(f"    reason: {entry['reason']}")
    elif node == "rewrite":
        print(f"    {entry['old_query']!r}")
        print(f"      → {entry['new_query']!r}")
    elif node == "generate":
        print(f"    n_chunks={entry['n_chunks']}")


def run_single_query(query: str, graph, cloud: "CloudLLM", local: "LocalLLM") -> dict:
    print("\n" + "=" * 72)
    print(f"  [query] {query}")
    print("=" * 72)

    t0 = time.time()
    last_trace_len = 0
    final_state = None
    for state in graph.stream(initial_state(query), stream_mode="values"):
        final_state = state
        trace = state.get("trace") or []
        for entry in trace[last_trace_len:]:
            _print_trace_entry(entry)
        last_trace_len = len(trace)

    elapsed = time.time() - t0
    assert final_state is not None

    print("\n" + "─" * 72)
    print("  最终答案")
    print("─" * 72)
    print(f"  {final_state['final_answer']}")

    print("\n" + "─" * 72)
    print("  本题统计")
    print("─" * 72)
    print(f"  耗时:      {elapsed:.1f}s")
    print(f"  retries:  {final_state['retries']}  (rewrite 触发次数)")
    print(f"  chunks:   {len(final_state['chunks'])}  (累计去重后)")
    print(f"  Cloud:    {cloud.calls} calls  ¥{cloud.cost_yuan():.4f}  "
          f"in={cloud.total_in}(hit {cloud.total_hit})  out={cloud.total_out}  "
          f"L{cloud.total_latency:.1f}s")
    print(f"  Local:    {local.calls} calls  L{local.total_latency:.1f}s")

    return final_state


# ============================================================
# 批量评测模式
# ============================================================

def _snapshot_costs(cloud: "CloudLLM", local: "LocalLLM") -> dict:
    return {
        "cloud_calls": cloud.calls,
        "cloud_in": cloud.total_in,
        "cloud_out": cloud.total_out,
        "cloud_hit": cloud.total_hit,
        "cloud_latency": round(cloud.total_latency, 2),
        "cloud_cost_yuan": round(cloud.cost_yuan(), 6),
        "local_calls": local.calls,
        "local_latency": round(local.total_latency, 2),
    }


def _hit_check(source_chapter: str, top_chunks: list[dict]) -> str:
    if source_chapter == "none":
        return "—"
    for c in top_chunks:
        if c["source"] in source_chapter:
            return "✓"
    return "✗"


def run_eval(graph, cloud: "CloudLLM", local: "LocalLLM", output_path: Path) -> None:
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    print(f"\n[eval] {len(golden)} 题  →  {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_hit = n_not_trap = 0

    with output_path.open("w", encoding="utf-8") as f:
        for q in golden:
            t0 = time.time()
            base = _snapshot_costs(cloud, local)
            final_state = graph.invoke(initial_state(q["query"]))
            elapsed = time.time() - t0
            after = _snapshot_costs(cloud, local)
            delta = {k: round(after[k] - base[k], 6) for k in base}

            record = {
                "id": q["id"],
                "query": q["query"],
                "type": q["type"],
                "expected": q["expected"],
                "source_chapter": q["source_chapter"],
                # A5 特有
                "sub_queries": final_state["sub_queries"],
                "retries": final_state["retries"],
                "base_llm_answer": final_state["base_llm_answer"],  # 沉淀不联网答案以便做 Benchmark 对比
                "final_answer": final_state["final_answer"],
                "n_chunks_final": len(final_state["chunks"]),
                "retrieved": [
                    {
                        "rank": i + 1,
                        "source": c["source"],
                        "chunk_idx": c["chunk_idx"],
                        "retrieve_round": c.get("retrieve_round"),
                        "from_subquery": c.get("from_subquery"),
                        "rerank_score": round(c.get("rerank_score", 0.0), 4),
                        "text": c["text"],
                    }
                    for i, c in enumerate(final_state["chunks"])
                ],
                "trace": final_state["trace"],
                "elapsed_sec": round(elapsed, 2),
                "cost_delta": delta,
                "judgment": None,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

            hit = _hit_check(q["source_chapter"], final_state["chunks"])
            if q["source_chapter"] != "none":
                n_not_trap += 1
                if hit == "✓":
                    n_hit += 1

            ans_show = final_state["final_answer"].replace("\n", " ")[:45]
            print(f"  [{q['id']:02d}/{len(golden)}] {q['type']:<5}  "
                  f"hit={hit}  ret={final_state['retries']}  "
                  f"n_sub={len(final_state['sub_queries'])}  "
                  f"L{elapsed:5.1f}s  ¥{delta['cloud_cost_yuan']:.4f}  → {ans_show}")

    print(f"\n[summary] 章节级命中(非陷阱题分母 {n_not_trap}):")
    print(f"          chunks 命中: {n_hit}/{n_not_trap} ({n_hit / n_not_trap * 100:.1f}%)")
    print(f"[summary] Cloud 累计: {cloud.calls} calls  ¥{cloud.cost_yuan():.4f}  "
          f"in={cloud.total_in}(hit {cloud.total_hit})  out={cloud.total_out}  "
          f"L{cloud.total_latency:.1f}s")
    print(f"[summary] Local 累计: {local.calls} calls  L{local.total_latency:.1f}s")
    print(f"[done] 写入 {output_path}")


# ============================================================
# CLI 入口
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="A5 Self-Reflective / Agentic RAG (LangGraph)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
模式:
  单题 verbose:  uv run python agentic_rag.py --query "唐三的第一武魂是什么"
  批量评测:      uv run python agentic_rag.py --eval [--output runs/a5.jsonl]
  3a 节点 mock:  uv run python agentic_rag.py --mock [--llm-only]
""",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", help="单题模式")
    group.add_argument("--eval", action="store_true", help="批量模式")
    group.add_argument("--mock", action="store_true", help="3a 模式")

    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "runs" / "a5.jsonl",
                        help="--eval 输出路径")
    parser.add_argument("--llm-only", action="store_true", help="仅 --mock 模式有效")
    args = parser.parse_args()

    if args.mock:
        print("=" * 72)
        print("  A5 · 3a 节点 Mock 验证")
        print("=" * 72)
        test_llm_nodes()
        if not args.llm_only:
            test_retrieve_node()
        print("\n✓ 3a mock 测试完成")
        return

    cloud = CloudLLM()
    local = LocalLLM()
    resources = init_retrieval_resources()
    graph = build_graph(cloud, local, resources)
    print(f"[graph] compiled  nodes: decompose → retrieve → grade ⇄ rewrite → generate")
    print(f"[graph] MAX_RETRIES = {MAX_RETRIES}")

    if args.query:
        run_single_query(args.query, graph, cloud, local)
    elif args.eval:
        run_eval(graph, cloud, local, args.output)


if __name__ == "__main__":
    main()