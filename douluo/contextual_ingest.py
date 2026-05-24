"""A4 Contextual Ingestion —— 用 DeepSeek 为每个 chunk 生成上下文前缀,缓存到 SQLite。

核心思想 (Anthropic 2024 Contextual Retrieval):
    每个 chunk embed 前,让 LLM 看"完整文档 + 这个 chunk",
    写 50-100 字"这段在书中的语境"前缀;
    最终用 `ctx_prefix + chunk.text` 一起 embed 和建 BM25。

为什么这能解 A3 留下的 Q18 那种题:
    chunk 原文 "...门房刁难..." 只在 ch005 关键词里有,
    加了前缀 "本段讲唐三第一天进入诺丁学院遇到门房刁难和宿舍王圣冲突",
    检索时 "诺丁学院冲突" query 在 dense / sparse 两路都能命中。

DeepSeek 隐式缓存机制 (本脚本省钱核心):
    full_text (~13K tokens) 放 system message,每次请求完全一致 → 自动触发
    KV 缓存命中(0.02 元/M, 1 折);user message 只放 chunk + 指令 (~500 tokens 新增)。
    第一次请求全价,后续每次只为 ~500 token 新增部分付全价 + ~13K cache_hit 折扣价。

输出: cache/ctx.sqlite (表 ctx_prefix)
    primary key (source, chunk_idx),text_hash 防止切块逻辑改了拿到脏缓存。

用法:
    uv run python contextual_ingest.py --limit 2         # 实测 API 通不通
    uv run python contextual_ingest.py --limit-ch 5      # 跑前 5 章人工抽查
    uv run python contextual_ingest.py                   # 全量 (185 chunk, 估算 15 min)
    uv run python contextual_ingest.py --inspect 10      # 抽 10 条 prefix 检查质量(不调 API)
"""
import argparse
import hashlib
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

CORPUS_DIR = Path(__file__).parent / "corpus" / "douluo"
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DB = CACHE_DIR / "ctx.sqlite"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

load_dotenv(Path(__file__).parent / ".env")
API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

SYSTEM_PROMPT_TEMPLATE = """你是一位《斗罗大陆》小说研究者。下面是这部小说前 10 章(引子 + 第一~第九章)的全文。
后续我会反复发来"片段",每次请你用一两句话(50-100 字)说明这个片段在前 10 章中的语境。

输出要求:
- 直接给出前缀本身,不要寒暄、序号、引号、引言。
- 必须指明:这段大致属于哪一章节、主要事件、涉及的关键角色。
- 如果片段跨章节或承接前文,简单点明关联(例如"承接上一章 X 事件")。
- 控制在 50-100 字,精炼即可,不要复述原文。

【前 10 章全文】
{full_text}
"""

USER_PROMPT_TEMPLATE = """【片段 {source}#{chunk_idx:02d}】
{chunk_text}

【上下文前缀】"""


# ---------- 切块(与 hybrid_rag / rerank_rag 完全一致,保证 chunk_idx 对齐)----------

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


def load_full_text() -> str:
    """前 10 章拼成一份带分隔符的长文本,作为 contextual prompt 的固定前缀。"""
    parts: list[str] = []
    for txt_path in sorted(CORPUS_DIR.glob("ch*.txt")):
        parts.append(f"\n\n========== {txt_path.stem} ==========\n\n")
        parts.append(txt_path.read_text(encoding="utf-8"))
    return "".join(parts).strip()


def text_hash(text: str) -> str:
    """chunk 文本的 SHA-256 前 16 位,做缓存 invalidation。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------- SQLite 缓存 ----------

def init_db() -> sqlite3.Connection:
    CACHE_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(CACHE_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ctx_prefix (
            source TEXT NOT NULL,
            chunk_idx INTEGER NOT NULL,
            text_hash TEXT NOT NULL,
            ctx_prefix TEXT NOT NULL,
            model TEXT NOT NULL,
            prompt_tokens INTEGER,
            prompt_cache_hit_tokens INTEGER,
            prompt_cache_miss_tokens INTEGER,
            completion_tokens INTEGER,
            latency_sec REAL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (source, chunk_idx)
        )
    """)
    conn.commit()
    return conn


def get_cached(conn: sqlite3.Connection, source: str, chunk_idx: int, hash_: str) -> str | None:
    """命中条件:同 (source, chunk_idx) 且 text_hash 一致,否则视为脏缓存。"""
    row = conn.execute(
        "SELECT ctx_prefix FROM ctx_prefix WHERE source=? AND chunk_idx=? AND text_hash=?",
        (source, chunk_idx, hash_),
    ).fetchone()
    return row[0] if row else None


def upsert(conn: sqlite3.Connection, source: str, chunk_idx: int, hash_: str,
           prefix: str, model: str, usage: dict, latency: float) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO ctx_prefix
           (source, chunk_idx, text_hash, ctx_prefix, model,
            prompt_tokens, prompt_cache_hit_tokens, prompt_cache_miss_tokens,
            completion_tokens, latency_sec, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (source, chunk_idx, hash_, prefix, model,
         usage.get("prompt_tokens"),
         usage.get("prompt_cache_hit_tokens"),
         usage.get("prompt_cache_miss_tokens"),
         usage.get("completion_tokens"),
         latency,
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


# ---------- DeepSeek 调用 ----------

def call_deepseek(client: OpenAI, system_prompt: str, user_prompt: str) -> tuple[str, dict, float]:
    # deepseek-v4 系列默认开 thinking,会把 max_tokens 吃光在 reasoning_content 上。
    # 这个简单任务(50 字总结)不需要思考链,关掉省 token、省时间、省钱。
    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=200,
        extra_body={"thinking": {"type": "disabled"}},
    )
    latency = time.time() - t0
    text = (resp.choices[0].message.content or "").strip()
    usage = resp.usage.model_dump() if resp.usage else {}
    return text, usage, latency


# ---------- 主流程 ----------

def run_ingest(limit: int | None, limit_ch: int | None) -> None:
    if not API_KEY:
        sys.exit("ERROR: DEEPSEEK_API_KEY 未在 .env 中设置")

    chunks = load_and_split()
    if limit_ch:
        chunks = [c for c in chunks if int(c["source"][2:]) <= limit_ch]
        print(f"[setup] --limit-ch {limit_ch} → 只处理 ch001~ch{limit_ch:03d},共 {len(chunks)} chunk")
    if limit:
        chunks = chunks[:limit]
        print(f"[setup] --limit {limit} → 只处理前 {limit} chunk")
    if not limit and not limit_ch:
        print(f"[setup] 全量 {len(chunks)} chunk")

    full_text = load_full_text()
    print(f"[setup] full_text {len(full_text):,} chars (中文 ≈ {len(full_text) * 3 // 2 // 1000}K tokens)")
    print(f"[setup] model={MODEL}, base_url={BASE_URL}")

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(full_text=full_text)
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    conn = init_db()

    n_done = n_cached = n_called = 0
    total_latency = 0.0
    total_cache_hit = total_cache_miss = total_completion = 0
    t_start = time.time()

    for i, c in enumerate(chunks, 1):
        hash_ = text_hash(c["text"])
        cached = get_cached(conn, c["source"], c["chunk_idx"], hash_)
        if cached:
            n_cached += 1
            n_done += 1
            print(f"  [{i:3d}/{len(chunks)}] {c['source']}#{c['chunk_idx']:02d}  CACHED  {cached[:60].replace(chr(10),' ')}...")
            continue

        user_prompt = USER_PROMPT_TEMPLATE.format(
            source=c["source"], chunk_idx=c["chunk_idx"], chunk_text=c["text"]
        )
        try:
            prefix, usage, latency = call_deepseek(client, system_prompt, user_prompt)
        except Exception as e:
            print(f"  [{i:3d}/{len(chunks)}] ERROR: {type(e).__name__}: {e}")
            raise

        upsert(conn, c["source"], c["chunk_idx"], hash_, prefix, MODEL, usage, latency)
        n_called += 1
        n_done += 1
        total_latency += latency
        total_cache_hit += usage.get("prompt_cache_hit_tokens") or 0
        total_cache_miss += usage.get("prompt_cache_miss_tokens") or 0
        total_completion += usage.get("completion_tokens") or 0

        cache_info = f"hit={usage.get('prompt_cache_hit_tokens', 0):>5} miss={usage.get('prompt_cache_miss_tokens', 0):>5}"
        print(f"  [{i:3d}/{len(chunks)}] {c['source']}#{c['chunk_idx']:02d}  {latency:5.1f}s  {cache_info}  out={usage.get('completion_tokens', 0)}")
        print(f"        → {prefix.replace(chr(10), ' ')[:140]}{'...' if len(prefix) > 140 else ''}")

    total_time = time.time() - t_start
    denom = max(total_cache_hit + total_cache_miss, 1)
    cache_hit_rate = total_cache_hit / denom
    # DeepSeek v4-flash 价格 (元/M tokens): cache_hit 0.02, cache_miss(input) 1.0, output 2.0
    cost_hit = total_cache_hit / 1_000_000 * 0.02
    cost_miss = total_cache_miss / 1_000_000 * 1.0
    cost_out = total_completion / 1_000_000 * 2.0
    cost_total = cost_hit + cost_miss + cost_out

    print(f"\n[done] {n_done} chunk ({n_cached} 命中本地 SQLite, {n_called} 实调 DeepSeek)")
    if n_called:
        print(f"[done] 总耗时 {total_time/60:.1f} min, 平均 API 延迟 {total_latency / n_called:.2f}s/chunk")
        print(f"[done] tokens: cache_hit={total_cache_hit:,} cache_miss={total_cache_miss:,} completion={total_completion:,}")
        print(f"[done] DeepSeek 隐式缓存命中率: {cache_hit_rate*100:.1f}%")
        print(f"[done] 成本: cache_hit ¥{cost_hit:.4f} + cache_miss ¥{cost_miss:.4f} + output ¥{cost_out:.4f} = ¥{cost_total:.4f}")


def run_inspect(n: int) -> None:
    conn = init_db()
    rows = conn.execute(
        "SELECT source, chunk_idx, ctx_prefix FROM ctx_prefix ORDER BY RANDOM() LIMIT ?",
        (n,),
    ).fetchall()
    if not rows:
        print("(cache 为空,先跑一次 ingest)")
        return
    chunks = {(c["source"], c["chunk_idx"]): c["text"] for c in load_and_split()}
    for src, idx, prefix in rows:
        text = chunks.get((src, idx), "(text not found)")
        print(f"\n{'='*72}")
        print(f"  {src}#{idx:02d}")
        print('=' * 72)
        print(f"原文: {text[:200].replace(chr(10),' ')}{'...' if len(text) > 200 else ''}")
        print(f"前缀: {prefix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="A4 Contextual ingestion (DeepSeek v4-flash)")
    parser.add_argument("--limit", type=int, help="只处理前 N 个 chunk")
    parser.add_argument("--limit-ch", type=int, help="只处理 ch001..ch{N} 的 chunk")
    parser.add_argument("--inspect", type=int, help="从缓存随机抽 N 条打印检查质量(不调 API)")
    args = parser.parse_args()

    if args.inspect:
        run_inspect(args.inspect)
    else:
        run_ingest(args.limit, args.limit_ch)


if __name__ == "__main__":
    main()
