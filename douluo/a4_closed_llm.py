"""A4.5 数据污染对照 —— DeepSeek 闭卷答 30 题,验证预训练记忆 vs RAG 贡献。

为什么要做这个实验:
    《斗罗大陆》是中国头部网文 IP(2008-2014 连载,网络随处可见),DeepSeek 预训练几乎一定
    见过全本。A4.5 开卷实验里 DeepSeek 答对的题,可能是从 top-5 chunk 抓的(理想),
    也可能是凭训练记忆答的(污染)。两者无法从答案表象上区分。

    本脚本是 A4.5 的对照组:**不给任何 chunk**,让 DeepSeek 凭训练数据答同样 30 题。
    差额 = RAG 系统的真实贡献。

    + 闭卷 hit ≈ 开卷 hit  → RAG 几乎无用,A4.5 的提升被污染解释,需换冷门语料重做
    + 闭卷 hit ≈ Qwen3-8B  → 8B 量化版本身能力也接近,A4.5 的提升主要来自模型规模/复述能力
    + 闭卷 hit << 开卷 hit → DeepSeek 主要从 chunks 拿信息,A4.5 结论稳

陷阱题专项(Q25-Q30 expected 是"原文未提及"):
    + 这些题问的是"前 10 章里的事实",而 DeepSeek 知道全本。
    + 闭卷时它可能凭后文知识答对(如 Q26 "唐三母亲" → 阿银,Q28 "大师真名" → 玉小刚),
      但这种"答对"其实违背了 expected("前 10 章里没说") —— 反而是污染的直接证据。

prompt 设计:
    + 主动提示"基于你对《斗罗大陆》的了解"(让它放心调用训练记忆,这样如果它说"不知道"
      就是真的不知道,而不是被 prompt 约束住的)
    + 简洁、不复述、与开卷模板尽量相似(便于对比)
    + 不加"如果你不确定就说不知道"这种引导(那会让闭卷 hit 率人为降低)

用法:
    uv run python a4_closed_llm.py             # 跑全部 30 题
    uv run python a4_closed_llm.py --limit 3   # 调试
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

GOLDEN_PATH = Path(__file__).parent / "golden_questions.json"
OUTPUT_PATH = Path(__file__).parent / "runs" / "a4_closed.jsonl"

CLOSED_PROMPT = """你是一个《斗罗大陆》小说知识助手。请基于你对这本小说的了解回答下面的问题。

要求:
1. 回答简洁、准确,不需要复述问题。
2. 如果你确实不知道答案,直接回答"不知道"。

【问题】
{question}

【回答】"""


def call_deepseek(client: OpenAI, model: str, prompt: str) -> tuple[str, dict, float]:
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
    parser = argparse.ArgumentParser(description="A4.5 闭卷对照:DeepSeek 不带 chunks 凭记忆答 30 题")
    parser.add_argument("--golden", type=Path, default=GOLDEN_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--limit", type=int, help="只跑前 N 题(调试)")
    args = parser.parse_args()

    load_dotenv(Path(__file__).parent / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    if not api_key:
        sys.exit("ERROR: DEEPSEEK_API_KEY 未在 .env 中设置")

    golden = json.loads(args.golden.read_text(encoding="utf-8"))
    if args.limit:
        golden = golden[: args.limit]

    print(f"[setup] golden={args.golden}, output={args.output}")
    print(f"[setup] model={model}, base_url={base_url}")
    print(f"[setup] 加载 {len(golden)} 题(闭卷:不给任何 chunk)\n")

    client = OpenAI(api_key=api_key, base_url=base_url)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    total_in = total_out = total_latency = 0.0

    with args.output.open("w", encoding="utf-8") as f:
        for q in golden:
            prompt = CLOSED_PROMPT.format(question=q["query"])
            try:
                answer, usage, latency = call_deepseek(client, model, prompt)
            except Exception as e:
                print(f"  [{q['id']:02d}] ERROR: {type(e).__name__}: {e}")
                raise

            out = {
                "id": q["id"],
                "query": q["query"],
                "type": q["type"],
                "expected": q["expected"],
                "source_chapter": q["source_chapter"],
                "answer_deepseek_closed": answer,
                "latency_sec": round(latency, 2),
                "usage": usage,
                "judgment_closed": None,  # 后续由 Claude 填
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

            total_in += usage.get("prompt_tokens", 0) or 0
            total_out += usage.get("completion_tokens", 0) or 0
            total_latency += latency

            ans = answer.replace("\n", " ")[:90]
            print(f"  [{q['id']:02d}/{len(golden)}] {q['type']:<5}  L{latency:5.2f}s  in={usage.get('prompt_tokens', 0):>3} out={usage.get('completion_tokens', 0):>3}")
            print(f"        Q: {q['query']}")
            print(f"        E: {q['expected'][:80]}")
            print(f"        A: {ans}")

    cost = total_in / 1_000_000 * 1.0 + total_out / 1_000_000 * 2.0
    print(f"\n[summary] {len(golden)} 题完成,总耗时 {total_latency:.1f}s,平均 {total_latency/len(golden):.2f}s/题")
    print(f"          tokens: input={int(total_in):,}, output={int(total_out):,}")
    print(f"          预估费用: ¥{cost:.4f}")
    print(f"[done] 写入 {args.output}")


if __name__ == "__main__":
    main()
