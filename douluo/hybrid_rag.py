"""A2 Hybrid RAG —— Dense + BM25 + RRF 融合 / 单题可视化 / 批量评测。

流水线(相比 A1 的唯一变化:多了 sparse 一路 + RRF 融合,top 从 3 → 5):
    ch*.txt → RecursiveCharacterTextSplitter(500/50)
            ├─→ bge-small-zh-v1.5 → FAISS dense top-20
            └─→ jieba 分词 → rank-bm25 sparse top-20
            → RRF 融合 → top-5
            → prompt → Qwen3-8B → answer

模式 1:单题 verbose(8 步可视化,清楚看到 dense / sparse / 融合三个阶段)
    uv run python hybrid_rag.py --query "唐三的第一武魂是什么?"

模式 2:批量评测(同 30 题 → runs/a2.jsonl,额外记录 dense_top1 / sparse_top1 / fused_top1)
    uv run python hybrid_rag.py --eval --output runs/a2.jsonl
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
from sentence_transformers import SentenceTransformer

CORPUS_DIR = Path(__file__).parent / "corpus" / "douluo"
GOLDEN_PATH = Path(__file__).parent / "golden_questions.json"
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
LLM_MODEL = "qwen3:8b"
LLM_BASE_URL = "http://localhost:11434/v1/"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# A2 新增超参
DENSE_TOP_K = 20      # 第一路:dense 召回数
SPARSE_TOP_K = 20     # 第二路:sparse 召回数
FINAL_TOP_K = 5       # RRF 融合后喂给 LLM 的数量(A1 是 3,A2 起改成 5)
RRF_K = 60            # RRF 公式常数,论文 (Cormack et al. 2009) 默认值,几乎不用调

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


# ---------- 复用自 A1 ----------

def load_and_split() -> list[dict]:
    """读 ch*.txt → 切块 → 保留 source 元数据。"""
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


# ---------- A2 新增:分词 + RRF ----------

def tokenize_zh(text: str) -> list[str]:
    """jieba 中文分词,过滤纯空白 token。BM25 索引和 query 都用这个,保持一致。"""
    return [t for t in jieba.lcut(text) if t.strip()]


def rrf(dense_ids: list[int], sparse_ids: list[int], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal Rank Fusion。输入两路有序 chunk_idx 列表,输出 {chunk_idx: rrf_score}(未排序)。

    公式: score(d) = Σ 1 / (k + rank_i(d)),对每路里出现过的排名累加。
    精髓:只看排名不看原始分数,自动解决两路分数量纲不可比的问题。
    """
    scores: dict[int, float] = {}
    for rank, idx in enumerate(dense_ids):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, idx in enumerate(sparse_ids):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return scores


# ---------- 索引 + 检索 ----------

def build_indexes(chunks: list[dict], verbose: bool = False):
    """编码 chunks + 建 FAISS dense 索引 + jieba 分词后建 BM25 sparse 索引。

    返回 (embedder, faiss_index, bm25, chunk_vecs)。
    """
    embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
    chunk_vecs = embedder.encode(
        [c["text"] for c in chunks],
        normalize_embeddings=True,
        show_progress_bar=verbose,
        batch_size=32,
    ).astype(np.float32)
    faiss_index = faiss.IndexFlatIP(chunk_vecs.shape[1])
    faiss_index.add(chunk_vecs)

    tokenized = [tokenize_zh(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)

    return embedder, faiss_index, bm25, chunk_vecs


def dense_search(query: str, embedder, faiss_index, top_k: int = DENSE_TOP_K):
    """返回 ([(chunk_idx, cosine_score)...], query_vec)。"""
    q_vec = embedder.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, ids = faiss_index.search(q_vec, top_k)
    return [(int(i), float(s)) for i, s in zip(ids[0], scores[0])], q_vec


def sparse_search(query: str, bm25: BM25Okapi, top_k: int = SPARSE_TOP_K):
    """返回 [(chunk_idx, bm25_score)...],按 score 降序。"""
    q_tokens = tokenize_zh(query)
    scores = bm25.get_scores(q_tokens)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in top_idx]


def hybrid_retrieve(query, embedder, faiss_index, bm25, chunks, final_top_k=FINAL_TOP_K):
    """A2 检索入口:dense + sparse → RRF → top-k。

    返回 (top_chunks, query_vec, dense_results, sparse_results)。
    top_chunks 每个元素带 dense_rank / sparse_rank / fusion_source,方便后续分析。
    """
    dense_results, q_vec = dense_search(query, embedder, faiss_index)
    sparse_results = sparse_search(query, bm25)

    dense_ids = [i for i, _ in dense_results]
    sparse_ids = [i for i, _ in sparse_results]
    dense_score_map = dict(dense_results)
    sparse_score_map = dict(sparse_results)
    dense_rank_map = {idx: r + 1 for r, idx in enumerate(dense_ids)}
    sparse_rank_map = {idx: r + 1 for r, idx in enumerate(sparse_ids)}

    rrf_scores = rrf(dense_ids, sparse_ids)
    fused_ids = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)

    top = []
    for rank, idx in enumerate(fused_ids[:final_top_k], start=1):
        in_dense = idx in dense_rank_map
        in_sparse = idx in sparse_rank_map
        if in_dense and in_sparse:
            source = "both"
        elif in_dense:
            source = "dense"
        else:
            source = "sparse"
        top.append({
            **chunks[idx],
            "global_idx": idx,
            "rank": rank,
            "score": rrf_scores[idx],
            "dense_rank": dense_rank_map.get(idx),
            "sparse_rank": sparse_rank_map.get(idx),
            "dense_score": dense_score_map.get(idx),
            "sparse_score": sparse_score_map.get(idx),
            "fusion_source": source,
        })
    return top, q_vec, dense_results, sparse_results


# ---------- 模式 1:单题 verbose ----------

def run_single(args) -> None:
    banner("Step 1 · 加载 ch001-ch010 并切成 chunk")
    chunks = load_and_split()
    lens = [len(c["text"]) for c in chunks]
    by_ch: dict[str, int] = {}
    for c in chunks:
        by_ch[c["source"]] = by_ch.get(c["source"], 0) + 1
    print(f"  chunk 总数: {len(chunks)}")
    print(f"  chunk 长度: min={min(lens)}, mean={sum(lens) / len(lens):.0f}, max={max(lens)}")
    print(f"  按章节分布: {', '.join(f'{k}={v}' for k, v in sorted(by_ch.items()))}")

    banner(f"Step 2 · 双索引构建(FAISS + BM25)")
    embedder, faiss_index, bm25, chunk_vecs = build_indexes(chunks, verbose=True)
    print(f"  Dense:  {EMBED_MODEL}, 维度 {chunk_vecs.shape[1]}, FAISS 向量数 {faiss_index.ntotal}")
    print(f"  Sparse: jieba + BM25Okapi, 语料 {bm25.corpus_size} docs, 平均文档长度 {bm25.avgdl:.1f} tokens")

    banner("Step 3 · Query 解析")
    print(f"  query: {args.query}")
    q_tokens = tokenize_zh(args.query)
    print(f"  jieba 分词: {q_tokens}")

    banner(f"Step 4 · Dense 检索(FAISS,取 top-{DENSE_TOP_K},预览前 5)")
    dense_results, q_vec = dense_search(args.query, embedder, faiss_index)
    print(f"  query_vec shape: {q_vec.shape}")
    for r, (idx, s) in enumerate(dense_results[:5], 1):
        c = chunks[idx]
        preview = c["text"][:80].replace("\n", " ")
        print(f"    [{r}] cosine={s:.4f}  {c['source']}#{c['chunk_idx']:02d}  {preview}...")

    banner(f"Step 5 · Sparse 检索(BM25,取 top-{SPARSE_TOP_K},预览前 5)")
    sparse_results = sparse_search(args.query, bm25)
    for r, (idx, s) in enumerate(sparse_results[:5], 1):
        c = chunks[idx]
        preview = c["text"][:80].replace("\n", " ")
        print(f"    [{r}] bm25={s:.4f}  {c['source']}#{c['chunk_idx']:02d}  {preview}...")

    banner(f"Step 6 · RRF 融合 → top-{FINAL_TOP_K}")
    top, _, _, _ = hybrid_retrieve(args.query, embedder, faiss_index, bm25, chunks)
    # 互补性提示
    n_both = sum(1 for c in top if c["fusion_source"] == "both")
    n_dense_only = sum(1 for c in top if c["fusion_source"] == "dense")
    n_sparse_only = sum(1 for c in top if c["fusion_source"] == "sparse")
    print(f"  最终 top-{FINAL_TOP_K} 来源: both={n_both}, dense-only={n_dense_only}, sparse-only={n_sparse_only}")
    for c in top:
        d_rank = c["dense_rank"] if c["dense_rank"] else "—"
        s_rank = c["sparse_rank"] if c["sparse_rank"] else "—"
        preview = c["text"][:200].replace("\n", " ")
        print(f"\n  [{c['rank']}] RRF={c['score']:.4f}  dense_rank={d_rank}  sparse_rank={s_rank}  ({c['fusion_source']})")
        print(f"      {c['source']}#{c['chunk_idx']:02d}  {preview}{'...' if len(c['text']) > 200 else ''}")

    if args.no_llm:
        print("\n[--no-llm] 跳过 LLM 调用,结束。")
        return

    banner("Step 7 · 拼接 prompt(完整内容)")
    prompt = build_prompt(args.query, top)
    print(prompt)

    banner("Step 8 · Qwen3-8B 回答")
    client = OpenAI(base_url=LLM_BASE_URL, api_key="ollama")
    print(call_llm(prompt, client))


# ---------- 模式 2:批量评测 ----------

def hit_check(source_chapter: str, top1_source: str) -> str:
    """章节级 top-1 命中:top1 的章节是否出现在 expected source_chapter 里。"""
    if source_chapter == "none":
        return "—"  # 陷阱题
    return "✓" if top1_source in source_chapter else "✗"


def run_eval(output_path: Path, final_top_k: int) -> None:
    print("[setup] 加载语料 + 双索引(CPU 编码 ~10 秒)...")
    chunks = load_and_split()
    embedder, faiss_index, bm25, _ = build_indexes(chunks, verbose=False)
    print(f"[setup] FAISS={faiss_index.ntotal} 向量, BM25 corpus={bm25.corpus_size} docs")

    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    print(f"[setup] 加载金标问题: {len(golden)} 题\n")

    client = OpenAI(base_url=LLM_BASE_URL, api_key="ollama")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_hit_fused = 0
    n_hit_dense = 0
    n_hit_sparse = 0
    n_dense_only = 0  # dense 中而 sparse 漏
    n_sparse_only = 0  # sparse 中而 dense 漏
    n_not_trap = 0

    with output_path.open("w", encoding="utf-8") as f:
        for q in golden:
            t0 = time.time()
            top, _, dense_results, sparse_results = hybrid_retrieve(
                q["query"], embedder, faiss_index, bm25, chunks, final_top_k
            )
            prompt = build_prompt(q["query"], top)
            answer = call_llm(prompt, client)
            latency = time.time() - t0

            dense_top1_idx = dense_results[0][0]
            sparse_top1_idx = sparse_results[0][0]
            dense_top1 = chunks[dense_top1_idx]
            sparse_top1 = chunks[sparse_top1_idx]

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
                        "rrf_score": round(c["score"], 6),
                        "dense_rank": c["dense_rank"],
                        "sparse_rank": c["sparse_rank"],
                        "fusion_source": c["fusion_source"],
                        "text": c["text"],
                    }
                    for c in top
                ],
                "dense_top1": f"{dense_top1['source']}#{dense_top1['chunk_idx']:02d}",
                "sparse_top1": f"{sparse_top1['source']}#{sparse_top1['chunk_idx']:02d}",
                "fused_top1": f"{top[0]['source']}#{top[0]['chunk_idx']:02d}",
                "answer": answer,
                "judgment": None,  # 手工标 hit / partial / miss / hallucinate
                "latency_sec": round(latency, 2),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            f_fused = hit_check(q["source_chapter"], top[0]["source"])
            f_dense = hit_check(q["source_chapter"], dense_top1["source"])
            f_sparse = hit_check(q["source_chapter"], sparse_top1["source"])
            if q["source_chapter"] != "none":
                n_not_trap += 1
                if f_fused == "✓":
                    n_hit_fused += 1
                if f_dense == "✓":
                    n_hit_dense += 1
                if f_sparse == "✓":
                    n_hit_sparse += 1
                if f_dense == "✓" and f_sparse != "✓":
                    n_dense_only += 1
                if f_sparse == "✓" and f_dense != "✓":
                    n_sparse_only += 1

            ans_show = answer.replace("\n", " ")[:50]
            print(f"  [{q['id']:02d}/{len(golden)}] {q['type']:<5}  "
                  f"d={f_dense} s={f_sparse} f={f_fused}  "
                  f"f_top1={top[0]['source']}#{top[0]['chunk_idx']:02d}  "
                  f"{latency:5.2f}s  → {ans_show}")

    print(f"\n[summary] 章节级 top-1 命中(陷阱题 {len(golden) - n_not_trap} 题不计入,分母 {n_not_trap}):")
    print(f"          仅 dense:  {n_hit_dense}/{n_not_trap} ({n_hit_dense / n_not_trap * 100:.1f}%)")
    print(f"          仅 sparse: {n_hit_sparse}/{n_not_trap} ({n_hit_sparse / n_not_trap * 100:.1f}%)")
    print(f"          融合后:    {n_hit_fused}/{n_not_trap} ({n_hit_fused / n_not_trap * 100:.1f}%)")
    print(f"          互补效果:  dense 救场 sparse {n_dense_only} 次,sparse 救场 dense {n_sparse_only} 次")
    print(f"[done] 写入 {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="A2 Hybrid RAG (Dense + BM25 + RRF)")
    parser.add_argument("--query", help="单题 verbose 模式:输入问题")
    parser.add_argument("--eval", action="store_true", help="批量模式:跑 golden_questions.json")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "runs" / "a2.jsonl",
                        help="批量模式输出路径,默认 runs/a2.jsonl")
    parser.add_argument("--top-k", type=int, default=FINAL_TOP_K, help=f"RRF 融合后的最终 top-k,默认 {FINAL_TOP_K}")
    parser.add_argument("--no-llm", action="store_true", help="单题模式:跳过 LLM 只看检索")
    args = parser.parse_args()

    if args.eval:
        run_eval(args.output, args.top_k)
    elif args.query:
        run_single(args)
    else:
        parser.error("需要 --query 或 --eval 之一")


if __name__ == "__main__":
    main()
