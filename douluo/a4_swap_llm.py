"""A4.5 LLM 消融实验 —— 复用 A4 top-5 检索结果,把生成层从 Qwen3-8B 换成 DeepSeek-V4-Flash。

为什么做这个实验:
    A4 实测 Recall@5 = 96.6%,正确 chunk 几乎都在 top-5 里。但仍有 3 miss + 2 partial,
    手工分析(report.md A4 节 + PLAN.md A4.5)定位到 Q3/Q9/Q12/Q13/Q16 这 5 题的 top-5
    都已经含原文要点,Qwen3-8B 本地量化版没合成出来 —— 怀疑瓶颈在生成层而非检索层。
    本脚本固定检索输出(读 runs/a4.jsonl 的 top-5),只换 LLM,确保唯一变量是生成质量。

跟 contextual_rag.py 的 prompt 完全一致 (PROMPT_TEMPLATE / build_prompt 直接 import),
区别只在 client、model、不加 Qwen 专属的 " /no_think" 后缀。

输入: runs/a4.jsonl
输出: runs/a4_swap.jsonl(每行带 a4 原 answer + deepseek answer,逐题对比)

用法:
    uv run python a4_swap_llm.py             # 跑全部 30 题
    uv run python a4_swap_llm.py --limit 3   # 先跑 3 题验环境通不通
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from contextual_rag import build_prompt

INPUT_PATH = Path(__file__).parent / "runs" / "a4.jsonl"
OUTPUT_PATH = Path(__file__).parent / "runs" / "a4_swap.jsonl"


def call_deepseek(client: OpenAI, model: str, prompt: str) -> tuple[str, dict, float]:
    """单次 DeepSeek 调用。关 thinking(简单 QA 不需要思考链,省 token / 时间 / 钱)。"""
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=400,
        extra_body={"thinking": {"type": "disabled"}},
    )
    latency = time.time() - t0
    text = (resp.choices[0].message.content or "").strip()
    usage = resp.usage.model_dump() if resp.usage else {}
    return text, usage, latency


def main() -> None:
    parser = argparse.ArgumentParser(description="A4.5 LLM 消融:复用 A4 top-5,换 DeepSeek 生成")
    parser.add_argument("--input", type=Path, default=INPUT_PATH, help="A4 批量结果 jsonl")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="输出路径")
    parser.add_argument("--limit", type=int, help="只跑前 N 题(调试)")
    args = parser.parse_args()

    load_dotenv(Path(__file__).parent / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    if not api_key:
        sys.exit("ERROR: DEEPSEEK_API_KEY 未在 .env 中设置")

    print(f"[setup] input={args.input}")
    print(f"[setup] output={args.output}")
    print(f"[setup] model={model}, base_url={base_url}")

    with args.input.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        records = records[: args.limit]
    print(f"[setup] 加载 {len(records)} 题(A4 检索结果固定不变,只换生成层)\n")

    client = OpenAI(api_key=api_key, base_url=base_url)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    total_in = total_out = total_hit = 0
    total_latency = 0.0
    t_start = time.time()

    with args.output.open("w", encoding="utf-8") as f:
        for r in records:
            # 把 a4.jsonl 的 retrieved 字段塞回 build_prompt 期望的形状
            top = [
                {"rank": c["rank"], "source": c["source"], "text": c["text"]}
                for c in r["retrieved"]
            ]
            prompt = build_prompt(r["query"], top)

            try:
                answer, usage, latency = call_deepseek(client, model, prompt)
            except Exception as e:
                print(f"  [{r['id']:02d}] ERROR: {type(e).__name__}: {e}")
                raise

            out = {
                "id": r["id"],
                "query": r["query"],
                "type": r["type"],
                "expected": r["expected"],
                "source_chapter": r["source_chapter"],
                "rerank_top1": r["rerank_top1"],
                "answer_qwen3": r["answer"],
                "answer_deepseek": answer,
                "latency_qwen3_sec": r.get("latency_llm_sec"),
                "latency_deepseek_sec": round(latency, 2),
                "usage_deepseek": usage,
                "judgment_qwen3": None,    # 后续由 Claude 填(对照 report.md A4 节)
                "judgment_deepseek": None, # 后续由 Claude 填
                "retrieved_top1_source": r["retrieved"][0]["source"] if r["retrieved"] else None,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

            total_in += usage.get("prompt_tokens", 0) or 0
            total_out += usage.get("completion_tokens", 0) or 0
            total_hit += usage.get("prompt_cache_hit_tokens", 0) or 0
            total_latency += latency

            ans_ds = answer.replace("\n", " ")[:60]
            ans_qw = (r["answer"] or "").replace("\n", " ")[:60]
            print(f"  [{r['id']:02d}/{len(records)}] {r['type']:<5}  L{latency:5.2f}s  "
                  f"in={usage.get('prompt_tokens', 0):>4} out={usage.get('completion_tokens', 0):>3}")
            print(f"        Qwen3:     {ans_qw}")
            print(f"        DeepSeek:  {ans_ds}")

    total_time = time.time() - t_start
    miss_in = max(total_in - total_hit, 0)
    # DeepSeek v4-flash 计价(元 / M tokens):cache_hit 0.02,input miss 1.0,output 2.0
    cost = total_hit / 1_000_000 * 0.02 + miss_in / 1_000_000 * 1.0 + total_out / 1_000_000 * 2.0

    print(f"\n[summary] {len(records)} 题完成,总耗时 {total_time:.1f}s,平均 {total_latency/len(records):.2f}s/题")
    print(f"          tokens: input={total_in:,} (hit={total_hit:,}, miss={miss_in:,}), output={total_out:,}")
    print(f"          预估费用: ¥{cost:.4f}")
    print(f"[done] 写入 {args.output}")


if __name__ == "__main__":
    main()
