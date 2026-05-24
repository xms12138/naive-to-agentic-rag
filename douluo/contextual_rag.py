"""A4 Contextual RAG —— Anthropic Contextual Retrieval + A3 Hybrid + Rerank。

流水线(相比 A3 的唯一变化:embed/BM25/reranker 都改用 ctx_prefix+text):
    ch*.txt → splitter(500/50)
            ↓
            JOIN cache/ctx.sqlite (A4 contextual_ingest 产物)
            ↓
            每个 chunk 多两个字段:
              · ctx_prefix:LLM 生成的 50-100 字上下文前缀
              · embed_text = ctx_prefix + "\\n\\n" + text(用于 embed / BM25 / reranker)
            ↓
            ├─→ bge-small-zh-v1.5 encode(embed_text) → FAISS dense top-20
            └─→ jieba 分词(embed_text)              → BM25 sparse top-20
            → RRF 融合 → top-30 候选
            → bge-reranker-v2-m3((query, embed_text))精排 → top-5
            → prompt(只用 chunk.text 原文,prefix 不进 prompt 避免"自生成回声")
            → Qwen3-8B → answer

为什么 prefix 进检索不进 prompt:
    · 检索阶段:prefix 是 query 找 chunk 的"额外路标",显式标注"本段讲诺丁学院冲突"
      让 query "诺丁学院冲突" 在 dense / sparse 双路都能命中(A3 失败的 Q18 直击靶心)
    · LLM 阶段:LLM 应当从原文推理,而不是从自己生成的 50 字总结里抄答案 ——
      让 prefix 进 prompt 会形成"LLM 看到自己写的总结再答原题"的回声链路,
      容易把 hallucinated prefix 进一步放大成 hallucinated answer

模式 1:单题 verbose(10 步,新增 Step 0 加载 prefix + Step 6.5 看 prefix 起的作用)
    uv run python contextual_rag.py --query "唐三初到诺丁学院第一天遇到了哪几次冲突?"

模式 2:批量评测(30 题 → runs/a4.jsonl,与 a3.jsonl 同字段方便对比)
    uv run python contextual_rag.py --eval --output runs/a4.jsonl
"""
import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import faiss
import jieba
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

CORPUS_DIR = Path(__file__).parent / "corpus" / "douluo"
GOLDEN_PATH = Path(__file__).parent / "golden_questions.json"
CACHE_DB = Path(__file__).parent / "cache" / "ctx.sqlite"

EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
LLM_MODEL = "qwen3:8b"
LLM_BASE_URL = "http://localhost:11434/v1/"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

DENSE_TOP_K = 20
SPARSE_TOP_K = 20
CANDIDATE_TOP_K = 30
FINAL_TOP_K = 5
RRF_K = 60

PROMPT_TEMPLATE = """你是一个《斗罗大陆》小说知识助手。我会给你从小说原文中检索到的若干片段,请基于这些片段回答用户的问题。

规则:
1. 只能根据提供的原文片段作答,不要使用你自己的知识,也不要编造内容。
2. 如果片段中没有相关信息,直接回答"原文未提及"。
3. 回答简洁、准确,不需要复述原文。

【检索到的原文片段】
{context}

【问题】
{question}

【回答】"""


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


# ---------- 切块(与 A1/A2/A3 完全一致,保证 chunk_idx 对齐)----------

def load_and_split() -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "?", "!", ";", ",", ""],
        length_function=len,
    )
    chunks = []
    for txt_path in sorted(CORPUS_DIR.glob("ch*.txt")):
        for i, piece in enumerate(splitter.split_text(txt_path.read_text(encoding="utf-8"))):
            chunks.append({"text": piece, "source": txt_path.stem, "chunk_idx": i})
    return chunks


def load_chunks_with_prefix() -> list[dict]:
    """读切块 + JOIN cache/ctx.sqlite,给每个 chunk 加 ctx_prefix 和 embed_text。

    缺失 prefix 时报错退出(A4 必须先跑完 contextual_ingest)。
    """
    chunks = load_and_split()
    if not CACHE_DB.exists():
        sys.exit(f"ERROR: {CACHE_DB} 不存在,请先跑 contextual_ingest.py")

    conn = sqlite3.connect(CACHE_DB)
    rows = conn.execute("SELECT source, chunk_idx, ctx_prefix FROM ctx_prefix").fetchall()
    prefix_map = {(s, i): p for s, i, p in rows}

    missing = []
    for c in chunks:
        key = (c["source"], c["chunk_idx"])
        if key not in prefix_map:
            missing.append(f"{c['source']}#{c['chunk_idx']:02d}")
            continue
        c["ctx_prefix"] = prefix_map[key]
        c["embed_text"] = f"{c['ctx_prefix']}\n\n{c['text']}"

    if missing:
        sys.exit(f"ERROR: {len(missing)} chunk 缺 ctx_prefix: {missing[:5]}... 请重跑 contextual_ingest.py")

    return chunks


def build_prompt(query: str, top_chunks: list[dict]) -> str:
    """LLM prompt 只用 chunk.text 原文 —— prefix 仅供检索,不进 LLM。"""
    context = "\n\n---\n\n".join(
        f"【片段 {c['rank']} · 来自 {c['source']}】\n{c['text']}" for c in top_chunks
    )
    return PROMPT_TEMPLATE.format(context=context, question=query)


def call_llm(prompt: str, client: OpenAI) -> str:
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt + " /no_think"}],
    )
    return (resp.choices[0].message.content or "").strip()


def tokenize_zh(text: str) -> list[str]:
    return [t for t in jieba.lcut(text) if t.strip()]


def rrf(dense_ids: list[int], sparse_ids: list[int], k: int = RRF_K) -> dict[int, float]:
    scores: dict[int, float] = {}
    for rank, idx in enumerate(dense_ids):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, idx in enumerate(sparse_ids):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return scores


# ---------- 索引 ←★ A4 唯一与 A3 不同的地方:用 embed_text(prefix+text)替代 text ----------

def build_indexes(chunks: list[dict], verbose: bool = False):
    """FAISS + BM25 都基于 ctx_prefix + text 建立,这是 contextual retrieval 的关键。"""
    embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
    chunk_vecs = embedder.encode(
        [c["embed_text"] for c in chunks],
        normalize_embeddings=True,
        show_progress_bar=verbose,
        batch_size=32,
    ).astype(np.float32)
    faiss_index = faiss.IndexFlatIP(chunk_vecs.shape[1])
    faiss_index.add(chunk_vecs)
    bm25 = BM25Okapi([tokenize_zh(c["embed_text"]) for c in chunks])
    return embedder, faiss_index, bm25, chunk_vecs


def dense_search(query, embedder, faiss_index, top_k=DENSE_TOP_K):
    q_vec = embedder.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, ids = faiss_index.search(q_vec, top_k)
    return [(int(i), float(s)) for i, s in zip(ids[0], scores[0])], q_vec


def sparse_search(query, bm25, top_k=SPARSE_TOP_K):
    scores = bm25.get_scores(tokenize_zh(query))
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in top_idx]


# ---------- 候选池 + reranker(reranker 也用 embed_text)----------

def fuse_to_candidates(query, embedder, faiss_index, bm25, chunks,
                      candidate_k: int = CANDIDATE_TOP_K):
    dense_results, _ = dense_search(query, embedder, faiss_index)
    sparse_results = sparse_search(query, bm25)

    dense_ids = [i for i, _ in dense_results]
    sparse_ids = [i for i, _ in sparse_results]
    dense_score_map = dict(dense_results)
    sparse_score_map = dict(sparse_results)
    dense_rank_map = {idx: r + 1 for r, idx in enumerate(dense_ids)}
    sparse_rank_map = {idx: r + 1 for r, idx in enumerate(sparse_ids)}

    rrf_scores = rrf(dense_ids, sparse_ids)
    fused_ids = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[:candidate_k]

    candidates = []
    for rrf_rank, idx in enumerate(fused_ids, start=1):
        in_d, in_s = idx in dense_rank_map, idx in sparse_rank_map
        source = "both" if in_d and in_s else ("dense" if in_d else "sparse")
        candidates.append({
            **chunks[idx],
            "global_idx": idx,
            "rrf_rank": rrf_rank,
            "rrf_score": rrf_scores[idx],
            "dense_rank": dense_rank_map.get(idx),
            "sparse_rank": sparse_rank_map.get(idx),
            "dense_score": dense_score_map.get(idx),
            "sparse_score": sparse_score_map.get(idx),
            "fusion_source": source,
        })
    return candidates, dense_results, sparse_results


def rerank(query: str, candidates: list[dict], reranker: CrossEncoder,
           final_k: int = FINAL_TOP_K) -> tuple[list[dict], float]:
    """Reranker 看 embed_text(prefix + text) —— 跟 Anthropic 原版做法一致。"""
    t0 = time.time()
    pairs = [[query, c["embed_text"]] for c in candidates]
    scores = reranker.predict(pairs)
    latency = time.time() - t0

    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:final_k]
    for new_rank, c in enumerate(ranked, start=1):
        c["rank"] = new_rank
    return ranked, latency


# ---------- 模式 1:单题 verbose ----------

def run_single(args) -> None:
    banner("Step 0 · 加载 ch001-ch010 切块 + JOIN cache/ctx.sqlite")
    chunks = load_chunks_with_prefix()
    avg_prefix = sum(len(c["ctx_prefix"]) for c in chunks) / len(chunks)
    print(f"  chunk 总数: {len(chunks)},全部有 prefix")
    print(f"  prefix 平均长度: {avg_prefix:.0f} chars(50-100 字目标)")
    print(f"  embed_text = ctx_prefix + chunk.text(检索用);LLM prompt 仍用原 text")

    banner("Step 1 · 双索引构建(FAISS + BM25,基于 embed_text)")
    embedder, faiss_index, bm25, chunk_vecs = build_indexes(chunks, verbose=True)
    print(f"  Dense:  {EMBED_MODEL}, 维度 {chunk_vecs.shape[1]}, FAISS 向量数 {faiss_index.ntotal}")
    print(f"  Sparse: jieba + BM25Okapi, 语料 {bm25.corpus_size} docs")

    banner("Step 2 · 加载 cross-encoder")
    t0 = time.time()
    reranker = CrossEncoder(RERANKER_MODEL, device="cpu")
    print(f"  {RERANKER_MODEL} ready, 加载耗时 {time.time() - t0:.1f}s")

    banner("Step 3 · Query 解析")
    print(f"  query: {args.query}")
    print(f"  jieba 分词: {tokenize_zh(args.query)}")

    banner(f"Step 4 · Dense 检索(top-{DENSE_TOP_K},预览前 5)")
    dense_results, _ = dense_search(args.query, embedder, faiss_index)
    for r, (idx, s) in enumerate(dense_results[:5], 1):
        c = chunks[idx]
        print(f"    [{r}] cos={s:.4f}  {c['source']}#{c['chunk_idx']:02d}  {c['text'][:65].replace(chr(10), ' ')}...")

    banner(f"Step 5 · Sparse 检索(top-{SPARSE_TOP_K},预览前 5)")
    sparse_results = sparse_search(args.query, bm25)
    for r, (idx, s) in enumerate(sparse_results[:5], 1):
        c = chunks[idx]
        print(f"    [{r}] bm25={s:.4f}  {c['source']}#{c['chunk_idx']:02d}  {c['text'][:65].replace(chr(10), ' ')}...")

    banner(f"Step 6 · RRF 融合 → top-{CANDIDATE_TOP_K} 候选(只列前 8)")
    candidates, _, _ = fuse_to_candidates(args.query, embedder, faiss_index, bm25, chunks)
    for c in candidates[:8]:
        d, s = c["dense_rank"] or "—", c["sparse_rank"] or "—"
        print(f"    [{c['rrf_rank']:>2}] RRF={c['rrf_score']:.4f}  d={d!s:>3}  s={s!s:>3}  "
              f"({c['fusion_source']:<6}) {c['source']}#{c['chunk_idx']:02d}")

    banner(f"Step 6.5 · 看 prefix 起的作用:top-3 候选的 prefix 长什么样")
    for c in candidates[:3]:
        print(f"  · {c['source']}#{c['chunk_idx']:02d}  prefix: {c['ctx_prefix'][:140]}{'...' if len(c['ctx_prefix']) > 140 else ''}")

    banner(f"Step 7 · Cross-encoder 精排 → top-{FINAL_TOP_K}(reranker 也看 prefix+text)")
    top, rerank_latency = rerank(args.query, candidates, reranker)
    print(f"  reranker 跑 {len(candidates)} 对 (query, prefix+chunk) 耗时 {rerank_latency:.2f}s\n")
    print(f"  {'最终':<4} {'RRF原排':<6} {'rerank分':<10} {'来源':<6} {'位置':<12} 内容预览")
    print(f"  {'─' * 4} {'─' * 6} {'─' * 10} {'─' * 6} {'─' * 12} {'─' * 40}")
    for c in top:
        loc = f"{c['source']}#{c['chunk_idx']:02d}"
        delta = c['rrf_rank'] - c['rank']
        arrow = f"↑{delta}" if delta > 0 else (f"↓{-delta}" if delta < 0 else "—")
        print(f"  [{c['rank']}]  RRF#{c['rrf_rank']:<2}({arrow:>4}) {c['rerank_score']:>+8.3f}  "
              f"{c['fusion_source']:<6} {loc:<12} {c['text'][:50].replace(chr(10), ' ')}...")

    if args.no_llm:
        print("\n[--no-llm] 跳过 LLM 调用,结束。")
        return

    banner("Step 8 · 拼接 prompt(LLM 只看原文,不看 prefix)")
    prompt = build_prompt(args.query, top)
    print(prompt)

    banner("Step 9 · Qwen3-8B 回答")
    client = OpenAI(base_url=LLM_BASE_URL, api_key="ollama")
    print(call_llm(prompt, client))


# ---------- 模式 2:批量评测 ----------

def hit_check(source_chapter: str, top1_source: str) -> str:
    if source_chapter == "none":
        return "—"
    return "✓" if top1_source in source_chapter else "✗"


def run_eval(output_path: Path) -> None:
    print("[setup] 加载语料 + prefix + 双索引...")
    chunks = load_chunks_with_prefix()
    embedder, faiss_index, bm25, _ = build_indexes(chunks, verbose=False)
    print(f"[setup] FAISS={faiss_index.ntotal} 向量, BM25 corpus={bm25.corpus_size} docs")

    print(f"[setup] 加载 reranker {RERANKER_MODEL} (CPU)...")
    t0 = time.time()
    reranker = CrossEncoder(RERANKER_MODEL, device="cpu")
    print(f"[setup] reranker ready ({time.time() - t0:.1f}s)")

    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    print(f"[setup] 加载金标问题: {len(golden)} 题\n")

    client = OpenAI(base_url=LLM_BASE_URL, api_key="ollama")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_hit_rerank = n_hit_rrf = n_promoted = n_not_trap = 0

    with output_path.open("w", encoding="utf-8") as f:
        for q in golden:
            t0 = time.time()
            candidates, dense_results, sparse_results = fuse_to_candidates(
                q["query"], embedder, faiss_index, bm25, chunks
            )
            rrf_top1 = candidates[0]
            top, rerank_latency = rerank(q["query"], candidates, reranker)
            t_retrieve = time.time() - t0

            prompt = build_prompt(q["query"], top)
            t_llm0 = time.time()
            answer = call_llm(prompt, client)
            t_llm = time.time() - t_llm0
            total_latency = time.time() - t0
            rerank_top1 = top[0]

            record = {
                "id": q["id"],
                "query": q["query"],
                "type": q["type"],
                "expected": q["expected"],
                "source_chapter": q["source_chapter"],
                "retrieved": [
                    {
                        "rank": c["rank"],
                        "source": c["source"],
                        "chunk_idx": c["chunk_idx"],
                        "rrf_rank": c["rrf_rank"],
                        "rerank_score": round(c["rerank_score"], 4),
                        "rrf_score": round(c["rrf_score"], 6),
                        "dense_rank": c["dense_rank"],
                        "sparse_rank": c["sparse_rank"],
                        "fusion_source": c["fusion_source"],
                        "ctx_prefix": c["ctx_prefix"],
                        "text": c["text"],
                    }
                    for c in top
                ],
                "dense_top1": f"{chunks[dense_results[0][0]]['source']}#{chunks[dense_results[0][0]]['chunk_idx']:02d}",
                "sparse_top1": f"{chunks[sparse_results[0][0]]['source']}#{chunks[sparse_results[0][0]]['chunk_idx']:02d}",
                "rrf_top1": f"{rrf_top1['source']}#{rrf_top1['chunk_idx']:02d}",
                "rerank_top1": f"{rerank_top1['source']}#{rerank_top1['chunk_idx']:02d}",
                "promoted_top1": rerank_top1["global_idx"] != rrf_top1["global_idx"],
                "answer": answer,
                "judgment": None,
                "latency_sec": round(total_latency, 2),
                "latency_rerank_sec": round(rerank_latency, 2),
                "latency_llm_sec": round(t_llm, 2),
                "latency_retrieve_sec": round(t_retrieve, 2),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            f_rrf = hit_check(q["source_chapter"], rrf_top1["source"])
            f_rer = hit_check(q["source_chapter"], rerank_top1["source"])
            if q["source_chapter"] != "none":
                n_not_trap += 1
                if f_rrf == "✓":
                    n_hit_rrf += 1
                if f_rer == "✓":
                    n_hit_rerank += 1
                if rerank_top1["global_idx"] != rrf_top1["global_idx"]:
                    n_promoted += 1

            ans_show = answer.replace("\n", " ")[:50]
            promo = " ↺" if rerank_top1["global_idx"] != rrf_top1["global_idx"] else "  "
            print(f"  [{q['id']:02d}/{len(golden)}] {q['type']:<5}  "
                  f"rrf={f_rrf} rer={f_rer}{promo}  "
                  f"rer_top1={rerank_top1['source']}#{rerank_top1['chunk_idx']:02d}  "
                  f"L{total_latency:5.2f}s(r{rerank_latency:.2f})  → {ans_show}")

    print(f"\n[summary] 章节级 top-1 命中(非陷阱题分母 {n_not_trap}):")
    print(f"          RRF top-1:     {n_hit_rrf}/{n_not_trap} ({n_hit_rrf / n_not_trap * 100:.1f}%)")
    print(f"          Rerank top-1:  {n_hit_rerank}/{n_not_trap} ({n_hit_rerank / n_not_trap * 100:.1f}%)")
    print(f"          重排改动 top-1:{n_promoted}/{len(golden)} 题(promoted_top1)")
    print(f"[done] 写入 {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="A4 Contextual RAG (prefix + hybrid + rerank)")
    parser.add_argument("--query", help="单题 verbose 模式")
    parser.add_argument("--eval", action="store_true", help="批量评测 golden_questions.json")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "runs" / "a4.jsonl",
                        help="批量模式输出路径,默认 runs/a4.jsonl")
    parser.add_argument("--no-llm", action="store_true", help="单题模式跳过 LLM")
    args = parser.parse_args()

    if args.eval:
        run_eval(args.output)
    elif args.query:
        run_single(args)
    else:
        parser.error("需要 --query 或 --eval 之一")


if __name__ == "__main__":
    main()
