"""A0 章节预切:把原始斗罗大陆 txt 清洗后按"第X集...第Y章"边界切成章节文件。

属于 Track A Hybrid Chunking 的第 1 阶段(Structural Splitting)。
第 2 阶段(RecursiveCharacterTextSplitter)在 A1 的 naive_rag.py 里做。

用法:
    uv run python prepare_corpus.py --src /path/to/斗罗大陆.txt --limit 10
    uv run python prepare_corpus.py --src /path/to/斗罗大陆.txt --limit 999   # 切全本
"""
import argparse
import re
import sys
from pathlib import Path


def decode_bytes(raw: bytes) -> tuple[str, str]:
    """编码兜底解码。中文盗版 txt 常见 GB18030 / GBK,UTF-8 也保留尝试。"""
    for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    raise RuntimeError("尝试 utf-8 / gb18030 / gbk 全部失败,需手工指定编码")


def clean_global(text: str) -> str:
    """删除全文级噪声:开头版权声明、结尾广告块、中间残留水印行。"""
    # 头:截到第一个"第一集"前(版权 / 申明)
    text = re.sub(r"^.*?(?=^第一集)", "", text, count=1, flags=re.DOTALL | re.MULTILINE)
    # 中英文括号的"本书完"
    text = re.sub(r"[(（]本书完[)）]", "", text)
    # 尾部"更多精彩好书 ... Qinkan.net"整块(到文件末尾)
    text = re.sub(r"更多精彩好书.*?Qinkan\.net.*\Z", "", text, flags=re.DOTALL | re.IGNORECASE)
    # 中间残留任何含 Qinkan.net 的整行
    text = re.sub(r"^.*Qinkan\.net.*$\n?", "", text, flags=re.MULTILINE | re.IGNORECASE)
    return text


# 章节标题正则:^第X集 集名 (第Y章|引子) 章名$
# 示例:
#   "第一集 斗罗世界 引子 穿越的唐家三少"
#   "第一集 斗罗世界 第一章 斗罗大陆，异界唐三"
#   "第四十八集 完美融合 第三百三十六章 大结局，最后一个条件"
CHAPTER_RE = re.compile(
    r"^第[一二三四五六七八九十百零\d]+集\s+\S+\s+"
    r"(?:第[一二三四五六七八九十百千零\d]+章|引子)\s+[^\n]+$",
    re.MULTILINE,
)


def split_chapters(text: str, dst_dir: Path, limit: int) -> int:
    """按章节正则定位边界,前 limit 章各写一份 ch{NNN}.txt;返回实际写入章数。"""
    matches = list(CHAPTER_RE.finditer(text))
    print(f"[scan] 找到章节边界 {len(matches)} 个 (完整本预期 ~337)")

    n = min(limit, len(matches))
    for i in range(n):
        start = matches[i].start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].strip()

        title, _, body = block.partition("\n")
        body = re.sub(r"\n{3,}", "\n\n", body.strip())  # 压缩 3+ 空行为 1 个

        out = dst_dir / f"ch{i + 1:03d}.txt"
        out.write_text(f"{title}\n\n{body}\n", encoding="utf-8")
        print(f"  {out.name}: {title}  ({len(body)} chars)")

    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="斗罗大陆 A0 章节预切")
    parser.add_argument("--src", required=True, type=Path, help="原始完整 txt 路径")
    parser.add_argument("--limit", type=int, default=10, help="切前 N 章(含引子),默认 10")
    args = parser.parse_args()

    if not args.src.exists():
        sys.exit(f"源文件不存在: {args.src}")

    dst_dir = Path(__file__).parent / "corpus" / "douluo"
    dst_dir.mkdir(parents=True, exist_ok=True)

    raw = args.src.read_bytes()
    text, enc = decode_bytes(raw)
    print(f"[decode] {enc}  ({len(raw)} bytes -> {len(text)} chars)")

    text = clean_global(text)
    n = split_chapters(text, dst_dir, args.limit)
    print(f"\n[done] 写入 {n} 章到 {dst_dir}")


if __name__ == "__main__":
    main()
uv run python prepare_corpus.py --src /home/xms/projects/rag/斗罗大陆.txt --limit 10