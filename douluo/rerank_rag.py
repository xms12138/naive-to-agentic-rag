"""A3 Rerank RAG —— Hybrid + bge-reranker-v2-m3 精排 / 单题可视化 / 批量评测。

流水线(相比 A2 的唯一变化:RRF 融合后取 top-30 当候选池,送 cross-encoder 精排取 top-5):
    ch*.txt → splitter(500/50)
            ├─→ bge-small-zh-v1.5 → FAISS dense top-20
            └─→ jieba + rank-bm25  → sparse top-20
            → RRF 融合 → top-30 候选
            → bge-reranker-v2-m3(CPU,cross-encoder)精排 → top-5
            → prompt → Qwen3-8B → answer

A2 vs A3 关键区别(为什么要加这一层):
    - bi-encoder(dense/sparse)粗排靠的是"独立编码 + 事后比较",query 和 chunk 各编各的,丢失深度交互
    - cross-encoder(reranker)把 (query, chunk) 拼起来同进模型,从第一层就互相 attention
    - 代价:不能预计算,30 个候选实时跑 30 次前向,CPU 大概 1-3 秒,但能压噪声、压住"4:1 多数信号"陷阱

模式 1:单题 verbose(9 步,Step 7 重点看 RRF rank → Rerank rank 的"翻牌"对比)
    uv run python rerank_rag.py --query "罗三炮和大师是什么关系?"

模式 2:批量评测(30 题 → runs/a3.jsonl,额外记录 reranker 延迟、rerank_top1)
    uv run python rerank_rag.py --eval --output runs/a3.jsonl
"""
import argparse
import json
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
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
LLM_MODEL = "qwen3:8b"
LLM_BASE_URL = "http://localhost:11434/v1/"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# 检索超参
DENSE_TOP_K = 20         # 粗排:dense 召回数
SPARSE_TOP_K = 20        # 粗排:sparse 召回数
CANDIDATE_TOP_K = 30     # A3 新增:RRF 融合后保留的候选池大小,送给 reranker
FINAL_TOP_K = 5          # reranker 精排后喂给 LLM 的数量
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


# ---------- 沿用 A2 ----------

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


def tokenize_zh(text: str) -> list[str]:
    return [t for t in jieba.lcut(text) if t.strip()]


def rrf(dense_ids: list[int], sparse_ids: list[int], k: int = RRF_K) -> dict[int, float]:
    scores: dict[int, float] = {}
    for rank, idx in enumerate(dense_ids):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, idx in enumerate(sparse_ids):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return scores


def build_prompt(query: str, top_chunks: list[dict]) -> str:
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


# ---------- 索引(沿用 A2)----------

def build_indexes(chunks: list[dict], verbose: bool = False):
    embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
    chunk_vecs = embedder.encode(
        [c["text"] for c in chunks],
        normalize_embeddings=True,
        show_progress_bar=verbose,
        batch_size=32,
    ).astype(np.float32)
    faiss_index = faiss.IndexFlatIP(chunk_vecs.shape[1])
    faiss_index.add(chunk_vecs)
    bm25 = BM25Okapi([tokenize_zh(c["text"]) for c in chunks])
    return embedder, faiss_index, bm25, chunk_vecs


def dense_search(query, embedder, faiss_index, top_k=DENSE_TOP_K):
    q_vec = embedder.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, ids = faiss_index.search(q_vec, top_k)
    return [(int(i), float(s)) for i, s in zip(ids[0], scores[0])], q_vec


def sparse_search(query, bm25, top_k=SPARSE_TOP_K):
    scores = bm25.get_scores(tokenize_zh(query))
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in top_idx]


# ---------- A3 新增:候选池 + reranker ----------

def fuse_to_candidates(query, embedder, faiss_index, bm25, chunks,
                      candidate_k: int = CANDIDATE_TOP_K):
    """A2 的 hybrid_retrieve 拆开:只做到 RRF top-30,不取 final top-5。

    返回 (candidates, dense_results, sparse_results),candidates 是 dict 列表,
    带 rrf_rank / dense_rank / sparse_rank / fusion_source,但还没经过 reranker。
    """
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
        in_d = idx in dense_rank_map
        in_s = idx in sparse_rank_map
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
    """Cross-encoder 精排。返回 (top_chunks, latency_sec)。

    每个候选改写两个字段:
        - rerank_score: cross-encoder sigmoid 输出的相关性分(0~1,越高越相关)
        - rank: 精排后的最终排名(从 1 开始)
        - 同时保留原本的 rrf_rank,方便对比"翻牌"情况
    """
    t0 = time.time()
    pairs = [[query, c["text"]] for c in candidates]
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
    banner("Step 1 · 加载 ch001-ch010 并切成 chunk")
    chunks = load_and_split()
    print(f"  chunk 总数: {len(chunks)}")

    banner("Step 2 · 双索引构建(FAISS + BM25,沿用 A2)")
    embedder, faiss_index, bm25, chunk_vecs = build_indexes(chunks, verbose=True)
    print(f"  Dense:  {EMBED_MODEL}, 维度 {chunk_vecs.shape[1]}, FAISS 向量数 {faiss_index.ntotal}")
    print(f"  Sparse: jieba + BM25Okapi, 语料 {bm25.corpus_size} docs")

    banner("Step 2.5 · 加载 cross-encoder(首次约下载 600 MB)")
    t0 = time.time()
    reranker = CrossEncoder(RERANKER_MODEL, device="cpu")
    print(f"  {RERANKER_MODEL} ready, 加载耗时 {time.time() - t0:.1f}s")

    banner("Step 3 · Query 解析")
    print(f"  query: {args.query}")
    print(f"  jieba 分词: {tokenize_zh(args.query)}")

    banner(f"Step 4 · Dense 检索(FAISS,top-{DENSE_TOP_K},预览前 5)")
    dense_results, _ = dense_search(args.query, embedder, faiss_index)
    for r, (idx, s) in enumerate(dense_results[:5], 1):
        c = chunks[idx]
        print(f"    [{r}] cosine={s:.4f}  {c['source']}#{c['chunk_idx']:02d}  {c['text'][:70].replace(chr(10), ' ')}...")

    banner(f"Step 5 · Sparse 检索(BM25,top-{SPARSE_TOP_K},预览前 5)")
    sparse_results = sparse_search(args.query, bm25)
    for r, (idx, s) in enumerate(sparse_results[:5], 1):
        c = chunks[idx]
        print(f"    [{r}] bm25={s:.4f}  {c['source']}#{c['chunk_idx']:02d}  {c['text'][:70].replace(chr(10), ' ')}...")

    banner(f"Step 6 · RRF 融合 → top-{CANDIDATE_TOP_K} 候选池(只列前 8)")
    candidates, _, _ = fuse_to_candidates(args.query, embedder, faiss_index, bm25, chunks)
    for c in candidates[:8]:
        d, s = c["dense_rank"] or "—", c["sparse_rank"] or "—"
        print(f"    [{c['rrf_rank']:>2}] RRF={c['rrf_score']:.4f}  d={d!s:>3}  s={s!s:>3}  "
              f"({c['fusion_source']:<6}) {c['source']}#{c['chunk_idx']:02d}  "
              f"{c['text'][:60].replace(chr(10), ' ')}...")

    banner(f"Step 7 · Cross-encoder 精排 → top-{FINAL_TOP_K} ← A3 的灵魂步骤")
    top, rerank_latency = rerank(args.query, candidates, reranker)
    print(f"  reranker 跑 {len(candidates)} 个 (query, chunk) 对耗时 {rerank_latency:.2f}s\n")
    print("  RRF→Rerank 翻牌对比(看哪些被顶上来 / 哪些被压下去):")
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

    banner("Step 8 · 拼接 prompt(完整内容)")
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
    print("[setup] 加载语料 + 双索引...")
    chunks = load_and_split()
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

    n_hit_rerank = 0
    n_hit_rrf = 0
    n_promoted = 0  # reranker 把非 RRF-top1 顶到 top1 的次数
    n_not_trap = 0

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
    parser = argparse.ArgumentParser(description="A3 Rerank RAG (Hybrid + bge-reranker-v2-m3)")
    parser.add_argument("--query", help="单题 verbose 模式")
    parser.add_argument("--eval", action="store_true", help="批量评测 golden_questions.json")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "runs" / "a3.jsonl",
                        help="批量模式输出路径,默认 runs/a3.jsonl")
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
