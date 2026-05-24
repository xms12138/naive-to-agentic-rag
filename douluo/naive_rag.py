"""A1 朴素 RAG —— 完整流水线 + 每一步可视化 / 批量评测。

流水线:
    ch*.txt → RecursiveCharacterTextSplitter(500/50)
            → bge-small-zh-v1.5 embed (CPU)
            → 内存 FAISS (cosine via L2-normalized IP)
            → query embed → top-3
            → prompt → Qwen3-8B (Windows 端 ollama) → answer

模式 1:单题 verbose,适合手动检查每一步
    uv run python naive_rag.py --query "鬼见愁悬崖扔下一块石头要数几秒?"

模式 2:批量评测,跑 golden_questions.json,输出 jsonl
    uv run python naive_rag.py --eval --output runs/a1.jsonl
"""
import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from sentence_transformers import SentenceTransformer

CORPUS_DIR = Path(__file__).parent / "corpus" / "douluo"
GOLDEN_PATH = Path(__file__).parent / "golden_questions.json"
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
LLM_MODEL = "qwen3:8b"
LLM_BASE_URL = "http://localhost:11434/v1/"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

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


# ---------- 核心可复用函数 ----------

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


def build_index(chunks: list[dict], verbose: bool = False) -> tuple[SentenceTransformer, faiss.Index, np.ndarray]:
    """加载 embedder + 编码所有 chunk + 建 FAISS 索引。返回 (embedder, index, chunk_vecs)。"""
    embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
    chunk_vecs = embedder.encode(
        [c["text"] for c in chunks],
        normalize_embeddings=True,
        show_progress_bar=verbose,
        batch_size=32,
    ).astype(np.float32)
    index = faiss.IndexFlatIP(chunk_vecs.shape[1])
    index.add(chunk_vecs)
    return embedder, index, chunk_vecs


def retrieve(query: str, embedder, index, chunks, top_k: int) -> tuple[list[dict], np.ndarray]:
    """query → top_k chunks(带 rank / score 字段) + query_vec。"""
    query_vec = embedder.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, ids = index.search(query_vec, top_k)
    top = [
        {**chunks[int(idx)], "score": float(score), "rank": rank}
        for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), start=1)
    ]
    return top, query_vec


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


# ---------- 模式 1:单题 verbose ----------

def run_single(args) -> None:
    banner("Step 1 · 加载 ch001-ch010 并切成 chunk")
    chunks = load_and_split()
    lens = [len(c["text"]) for c in chunks]
    by_ch: dict[str, int] = {}
    for c in chunks:
        by_ch[c["source"]] = by_ch.get(c["source"], 0) + 1
    print(f"  chunk 总数: {len(chunks)}")
    print(f"  chunk 长度 (字符): min={min(lens)}, mean={sum(lens) / len(lens):.0f}, max={max(lens)}")
    print(f"  按章节分布: {', '.join(f'{k}={v}' for k, v in sorted(by_ch.items()))}")
    print(f"\n  ▶ 前 3 个 chunk 预览:")
    for c in chunks[:3]:
        text_preview = c["text"][:180].replace("\n", " ")
        print(f"    [{c['source']}#{c['chunk_idx']:02d}] ({len(c['text'])} 字)  {text_preview}...")

    banner(f"Step 2 · 加载 {EMBED_MODEL} 并编码 {len(chunks)} 个 chunk")
    embedder, index, chunk_vecs = build_index(chunks, verbose=True)
    print(f"  embedding 维度: {chunk_vecs.shape[1]}")
    print(f"  chunk_vecs shape: {chunk_vecs.shape}  dtype: {chunk_vecs.dtype}")
    print(f"  chunk_vecs[0] L2 范数: {np.linalg.norm(chunk_vecs[0]):.6f}  (归一化后应≈1.0)")
    print(f"  chunk_vecs[0] 前 8 维: {np.round(chunk_vecs[0][:8], 4).tolist()}")

    banner("Step 3 · 建立 FAISS 索引")
    print(f"  索引类型: IndexFlatIP (cosine via L2-normalized inner product)")
    print(f"  索引中向量数: {index.ntotal}")

    banner("Step 4 · 对 query 进行 embedding")
    print(f"  query: {args.query}")
    top, query_vec = retrieve(args.query, embedder, index, chunks, args.top_k)
    print(f"  query_vec shape: {query_vec.shape}")
    print(f"  query_vec 前 8 维: {np.round(query_vec[0][:8], 4).tolist()}")

    banner(f"Step 5 · FAISS 检索 top-{args.top_k}")
    for c in top:
        text_preview = c["text"][:300].replace("\n", " ")
        print(f"\n  [{c['rank']}] score={c['score']:.4f}   {c['source']}#{c['chunk_idx']:02d}")
        print(f"      {text_preview}{'...' if len(c['text']) > 300 else ''}")

    if args.no_llm:
        print("\n[--no-llm] 跳过 LLM 调用,结束。")
        return

    banner("Step 6 · 拼接 prompt(完整内容,准备喂给 Qwen3-8B)")
    prompt = build_prompt(args.query, top)
    print(prompt)

    banner("Step 7 · Qwen3-8B 回答")
    client = OpenAI(base_url=LLM_BASE_URL, api_key="ollama")
    print(call_llm(prompt, client))


# ---------- 模式 2:批量评测 ----------

def hit_check(source_chapter: str, top1_source: str) -> str:
    """简单的章节级命中判定:top-1 的章节是否出现在 expected source_chapter 里。"""
    if source_chapter == "none":
        return "—"   # 陷阱题,无所谓召回
    # source_chapter 形如 "ch003" 或 "ch001+ch002"
    return "✓" if top1_source in source_chapter else "✗"


def run_eval(output_path: Path, top_k: int) -> None:
    print(f"[setup] 加载语料 + 建索引(CPU 编码 ~10 秒)...")
    chunks = load_and_split()
    embedder, index, _ = build_index(chunks, verbose=False)
    print(f"[setup] 索引就绪: {len(chunks)} chunks, {index.d} 维")

    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    print(f"[setup] 加载金标问题: {len(golden)} 题\n")

    client = OpenAI(base_url=LLM_BASE_URL, api_key="ollama")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_hit = 0
    n_not_trap = 0
    with output_path.open("w", encoding="utf-8") as f:
        for q in golden:
            t0 = time.time()
            top, _ = retrieve(q["query"], embedder, index, chunks, top_k)
            prompt = build_prompt(q["query"], top)
            answer = call_llm(prompt, client)
            latency = time.time() - t0

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
                        "score": round(c["score"], 4),
                        "text": c["text"],
                    }
                    for c in top
                ],
                "answer": answer,
                "judgment": None,        # 留给用户手工标:hit / partial / miss / hallucinate
                "latency_sec": round(latency, 2),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            top1 = top[0]
            flag = hit_check(q["source_chapter"], top1["source"])
            if q["source_chapter"] != "none":
                n_not_trap += 1
                if flag == "✓":
                    n_hit += 1
            # answer 截断显示
            ans_show = answer.replace("\n", " ")[:55]
            print(f"  [{q['id']:02d}/{len(golden)}] {q['type']:<5}  "
                  f"top1={top1['source']}#{top1['chunk_idx']:02d} {flag}  "
                  f"{latency:5.2f}s  → {ans_show}")

    print(f"\n[summary] 章节级 top-1 命中: {n_hit}/{n_not_trap} ({n_hit / n_not_trap * 100:.1f}%)")
    print(f"          (陷阱题不计入,共 {len(golden) - n_not_trap} 题)")
    print(f"[done] 写入 {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="A1 朴素 RAG")
    parser.add_argument("--query", help="单题 verbose 模式:输入问题")
    parser.add_argument("--eval", action="store_true", help="批量模式:跑 golden_questions.json")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "runs" / "a1.jsonl",
                        help="批量模式输出路径,默认 runs/a1.jsonl")
    parser.add_argument("--top-k", type=int, default=3)
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
