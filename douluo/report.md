# 《斗罗大陆》RAG 演进对比报告

> Track A 原型实验记录,从朴素 RAG 走到 Agentic RAG。
> 每个阶段同语料、同评测集,**唯一变量是检索流水线**。

---

## 实验设置

| 项 | 配置 |
|---|---|
| 语料 | 《斗罗大陆》前 10 章(引子 + 第一~第九章),源文件 gb18030 → UTF-8 |
| 切块 | RecursiveCharacterTextSplitter,`chunk_size=500, chunk_overlap=50`,中文标点优先切 |
| chunk 总数 | **185** (按章节分布: ch001=6, ch002=18, ch003=19, ch004=21, ch005=21, ch006=16, ch007=22, ch008=21, ch009=21, ch010=20) |
| 评测集 | `golden_questions.json`,**30 题** × 5 类(事实/关系/跨章节/模糊/陷阱) |
| 评测标准 | 手工标注 `hit` / `partial` / `miss` / `hallucinate`;准答率 = (hit + partial) / 30 |
| LLM | Qwen3-8B 本地(WSL2 调 Windows 端 Ollama,`/no_think` 关闭思考链) |
| 评测命中口径 | top-1 章节命中只是检索层指标;最终判定看答案对 expected 的吻合度 |

---

## 主对比表

| 阶段 | 流水线变化 | hit | partial | miss | hallu. | 准答率 | **hit-only** | 平均延迟 | rerank/retrieve 段 |
|---|---|---|---|---|---|---|---|---|---|
| A1 朴素 | Dense (bge-small-zh) → top-3 → LLM | 15 | 4 | 10 | 1 | 63.3% | 50.0% | 17.9s | — |
| A2 hybrid | + BM25(jieba)+ RRF 融合,top-3 → top-5 | 19 | 4 | 6 | 1 | 76.7% | 63.3% | 20.9s | — |
| A3 rerank | + RRF top-30 → bge-reranker-v2-m3(CPU)精排 → top-5 | 24 | 2 | 3 | 1 | 86.7% | 80.0% | 44.6s | 22.9s |
| **A4 contextual** | + DeepSeek-V4-Flash 生成 ctx_prefix(50-100 字)prepend 到 embed/BM25/reranker 输入 | 25 | 2 | 3 | **0** | 90.0% | 83.3% | 48.3s | 26.3s |
| **A4.5 LLM 消融** | **复用 A4 top-5 检索结果不变,只把生成层从 Qwen3-8B 本地量化版换成 DeepSeek-V4-Flash** | **29** | **1** | **0** | **0** | **100.0%** | **96.7%** | **1.1s** | 26.3s(同 A4)|
| A4.5-closed 闭卷 | DeepSeek 不带任何 chunk,凭训练记忆答 30 题(数据污染对照,排除"小说在训练数据里"的干扰) | 12 | 8 | 10 | 0 | 66.7% | 40.0% | 1.0s | — |
| **A5 agentic** | **+ LangGraph 状态机(decompose / grade / rewrite / 多轮 retrieve);混合 LLM 路由(decompose/grade/generate=DeepSeek-V4-Flash;rewrite=本地 Qwen3-8B)** | 26 | 4 | **0** | **0** | **100.0%** | **86.7%** | 57.5s | retrieve 29.7s × 1.4 轮 |

**注**:准答率 = (hit + partial) / 30(口径与 A1-A4 历史一致);**hit-only 是严格命中口径**,A5 因 4 题退 partial 跟 A4.5 拉开 10 pp 差距。详见下文 A5 节。

---

## A1 朴素 RAG · 详细结果

### 按类型细分

| 类型 | 总数 | hit | partial | miss | hallu. |
|---|---|---|---|---|---|
| 事实型 | 6 | 3 | 0 | 3 | 0 |
| 关系型 | 6 | 2 | 2 | 2 | 0 |
| 跨章节型 | 6 | 1 | 1 | 4 | 0 |
| 模糊型 | 6 | 4 | 1 | 1 | 0 |
| 陷阱型 | 6 | 5 | 0 | 0 | 1 |

### 阅读这张表的要点

- **陷阱型 5/6 hit**:朴素 RAG 在反幻觉上意外坚挺。prompt 里"原文未提及"的兜底规则起了作用。
- **模糊型(列举)4 hit + 1 partial**:答案集中在一两章里时表现好(玄天宝录六种功夫、十大魂师称号、五级魂兽颜色)。
- **跨章节型 1 hit + 4 miss**:**最弱项**,完全符合 PLAN.md 预言。一次检索只看一个语义焦点,跨章合成超出朴素 RAG 能力。
- **事实型 3 miss**:专有名词(蓝银草 / 素云涛 / 唐门)被 dense 漏召。

### 延迟分布

- **平均 17.9s / 题,中位 16.8s,最大 43.0s,30 题总耗时 9 分钟**
- 主要时间消耗在 LLM 调用上(Qwen3-8B 本地,/no_think 已开)
- 延迟数字主要用于和 A3 / A5 对比(reranker 加几秒,agentic 因为多轮重检索可能翻倍)

---

## A1 暴露的 4 个典型失败模式

> 这 4 个模式不是 bug,而是朴素 RAG 的**结构性局限**。
> A2-A5 每一层在解决其中一个或几个。

### 1. 专有名词被 dense embedding 漏召 → **A2 BM25 解决**

**例**:Q1 *"唐三的第一武魂是什么?"*
- expected: 蓝银草(出处 ch003)
- 实际 top-1: ch002#14(无关章节)
- LLM 答: "原文未提及"

**原因**:`bge-small-zh` 对"蓝银草"这种稀有 token 不敏感,倾向召回包含"唐三"和"武魂"等高频词的段落。

**A2 怎么解**:BM25 对关键词的稀有度敏感,"蓝银草" IDF 高,直接拉到 top。Dense 抓概念 + BM25 抓术语,RRF 融合互补。

---

### 2. 跨章节信息无法关联 → **A4 Contextual Retrieval 解决**

**例**:Q13 *"唐三是如何从前世来到斗罗大陆的?"*
- expected 需要 ch001(跳鬼见愁) + ch002(转生到圣魂村)
- 实际 top-1: ch005#00(完全跑题)
- LLM 答: "原文未提及"

**原因**:每个 chunk 单独编码后,跨章语义关联丢失。"鬼见愁跳崖"和"圣魂村出生"两个事件在向量空间里距离很远,query 一次只能拉一个。

**A4 怎么解**:每个 chunk embed 之前,LLM 生成 50-100 字前缀("本段来自 ch001,讲唐三在唐门跳鬼见愁明志"),前缀让 chunk 自带跨章上下文,召回率应有质变。

---

### 3. 章节召回对但 LLM 不敢答 → **A3 Reranker 解决**

**例**:Q15 *"唐三的第一个魂环来自哪一头魂兽?"*
- top-1 召回到 ch009#17(**正确章节**)
- top-2/top-3 是 ch005/ch008 的无关片段
- LLM 答: "原文未提及"

**原因**:top-1 命中,但 top-2/3 的噪声把 LLM 拉到保守边。LLM 不知道哪个片段是答案,选择最安全的回答。

**A3 怎么解**:cross-encoder(bge-reranker-v2-m3)把 query 和每个 chunk 一起进模型打分,精排后 top-3 / top-5 噪声明显减少,LLM 才敢断言。

---

### 4. LLM 把混淆概念吞下 → 部分靠 **A5 Agentic 反思修正**

**例**:Q30 *"唐三在斗罗大陆制造的第一件袖箭,材料是普通生铁吗?"*
- expected: 不是,是经过百日锤击后变成的**铁母**(原文明确反对)
- top-1: ch005#01(正确章节,信息齐全)
- LLM 答: **"是的,普通生铁。"** ← 唯一一次 hallucinate

**原因**:这不是检索错误,而是**语言理解错误**。原文写"由生铁锻造成铁母 → 用铁母制袖箭",LLM 把"由生铁起头"误解成"材料是生铁"。检索给得再好,LLM 自己理解错就完蛋。

**A5 怎么解**:agentic RAG 的 `grade` 节点可以让 LLM 自己判断"答案是否被原文支持",出现矛盾时改写 query 再查;但语言理解错误只有更强的 LLM 或 self-consistency(多次采样投票)能根治。

---

## A1 关键数字一句话总结

> **朴素 RAG 在 30 道斗罗题上准答率 63.3%(15/30 全对 + 4/30 部分对),平均 17.9s/题。**
> 跨章节型(1/6 hit)和专有名词事实型(3/6 hit)是两个最大短板,接下来 A2-A4 会逐一拆解。

---

## A2 Hybrid RAG · 详细结果

### 按类型细分(A1 vs A2)

| 类型 | 总数 | A1 (hit/par/miss/hallu) | A2 (hit/par/miss/hallu) | 准答率 A1→A2 |
|---|---|---|---|---|
| 事实型 | 6 | 3/0/3/0 | 4/0/2/0 | 50% → **67%** ↑ |
| 关系型 | 6 | 2/2/2/0 | 2/1/2/1 | 67% → **50%** ↓(新增 hallu.) |
| 跨章节型 | 6 | 1/1/4/0 | 2/3/1/0 | 33% → **83%** ↑↑↑ |
| 模糊型 | 6 | 4/1/1/0 | 5/0/1/0 | 83% → **83%** = |
| 陷阱型 | 6 | 5/0/0/1 | 6/0/0/0 | 83% → **100%** ↑(消除 hallu.) |

### 检索层互补救场(章节级 top-1)

| 指标 | 数字 |
|---|---|
| 仅 dense top-1 章节命中 | 19/29 (65.5%) |
| 仅 sparse top-1 章节命中 | 20/29 (69.0%) |
| 融合后 top-1 章节命中 | 20/29 (69.0%) |
| dense 救场 sparse(sparse 漏 dense 中) | 4 题 |
| sparse 救场 dense(dense 漏 sparse 中) | 5 题 |

**反直觉点**:章节级 top-1,融合并没有显著超过 sparse 单路 —— 因为 RRF 把两路里错的也参与排序,有时反而把对的挤下来。但是**答案级准答率仍提升 13 pp**,因为 top-5 给了 LLM 更多机会找到正确答案 chunk,即使 top-1 不一定就是答案所在。**检索层提升 ≠ 答案层提升,这点很重要。**

### 延迟分布

A2 平均 20.9s,中位 19.3s,最大 53.5s,30 题总耗时 10.4 分钟。比 A1 多 3 秒,几乎全部来自 LLM 输入变长(top-3 → top-5 多 2 个 chunk)。BM25 检索 + jieba 分词 + RRF 加起来在 ms 级,几乎免费。

---

## A2 vs A1 翻盘表

### 7 题提升

| Q# | 类型 | A1 → A2 | 关键 |
|---|---|---|---|
| Q02 | 事实 | miss → hit | sparse 把"锤子武魂"相关 chunk 拉回 top |
| Q08 | 关系 | partial → hit | top-5 多召回让 LLM 答出"师徒"主关系 |
| Q15 | 跨章 | miss → partial | sparse 把"曼陀罗蛇""魂环"专有名词召回 |
| Q17 | 跨章 | miss → hit | 召回唐昊和大师都叮嘱锤子武魂的 chunk |
| Q18 | 跨章 | miss → partial | 召回王圣冲突相关 chunk(门房一条仍漏) |
| Q23 | 模糊 | partial → hit | top-5 完整覆盖大师 4 点蓝银草分析 |
| Q30 | 陷阱 | hallu → hit | 召回"铁母"原文,消除"生铁制袖箭"幻觉 |

### 1 题退步 —— 写进消融实验的关键案例

**Q11 关系型 hit → hallucinate**

- A1 (top-3) 答:**"罗三炮是大师的武魂。"** ✓
- A2 (top-5) 答:**"罗三炮是大师的魂兽,大师是它的魂师。"** ✗(把武魂说成魂兽,因果说反)

**根因剖析**:A2 召回的 top-5 里,4 个 chunk(ch010#01/07/08/09)讲的是"罗三炮在战斗中"——读起来罗三炮像独立生物;只有 1 个(ch009#14)明确写"罗三炮是变异武魂"。LLM 被 **4:1 的多数信号压垮**,推断成"罗三炮是魂兽"。A1 用 top-3 时反而没引入这么多战斗场景噪声。

**A3 预言**:cross-encoder reranker 看到 query "罗三炮和大师是什么关系",会主动给定义类 chunk(ch009#14)打高分,压制战斗场景。这道题应在 A3 修复。

---

## A2 关键数字一句话总结

> **Hybrid RAG (Dense + BM25 + RRF) 在 30 题上准答率 76.7%(19 hit + 4 partial),比 A1 提升 13 pp,延迟仅多 3s。最大赢家是跨章节型(33% → 83%),最大代价是 Q11 因 top-5 噪声出现 hallucinate —— 恰好预言了 A3 reranker 该解决的问题。**

---

## A3 Rerank RAG · 详细结果

### 流水线变化

A2 → A3 只在 RRF 之后多接一层 cross-encoder:

```
RRF 融合 top-5  →  改成 top-30 候选池
                   ↓
                   bge-reranker-v2-m3(CPU,XLM-RoBERTa cross-encoder)
                   对每个 (query, chunk) pair 实时打分
                   ↓
                   按 rerank_score 排序取 top-5 → LLM
```

**为什么需要 cross-encoder**:bi-encoder(dense/sparse)各自编码 query 和 chunk 后才比较,深度交互全丢;cross-encoder 把 (query, chunk) 拼起来同进模型,从第一层就互相 attention,所以能识别"罗三炮和大师是什么**关系**"里"关系"这个语义焦点,主动给定义类 chunk 加分、压住战斗场景。代价:不能预计算,30 个候选要实时跑 30 次前向。

### 按类型细分(A1 → A2 → A3)

| 类型 | 总数 | A1 (h/p/m/H) | A2 (h/p/m/H) | A3 (h/p/m/H) | 准答率轨迹 |
|---|---|---|---|---|---|
| 事实型 | 6 | 3/0/3/0 | 4/0/2/0 | 5/0/1/0 | 50% → 67% → **83%** ↑ |
| 关系型 | 6 | 2/2/2/0 | 2/1/2/1 | 4/0/2/0 | 67% → 50% → **67%** ↑(消除 hallu) |
| 跨章节型 | 6 | 1/1/4/0 | 2/3/1/0 | 3/2/0/1 | 33% → 83% → **83%** =(新增 1 hallu) |
| 模糊型 | 6 | 4/1/1/0 | 5/0/1/0 | 6/0/0/0 | 83% → 83% → **100%** ↑ |
| 陷阱型 | 6 | 5/0/0/1 | 6/0/0/0 | 6/0/0/0 | 83% → 100% → 100% = |

### 检索层指标(章节级 top-1 命中)

| 指标 | A2 | A3 | 备注 |
|---|---|---|---|
| RRF top-1 章节命中 | 20/29 (69.0%) | 20/29 (69.0%) | RRF 没动,数字一致 |
| **Rerank top-1 章节命中** | — | **19/29 (65.5%)** | **反而降 3.4 pp** |
| rerank 改动 top-1 题数 | — | 18/30 | 60% 的题被换 top-1 |

**反直觉点**:章节级 top-1 反而下降,但**答案准答率提升 10 pp**。原因和 A2 同源 —— 检索层命中 ≠ 答案层命中。rerank 真正在做的是**清理 top-5 整体的噪声**(把"罗三炮战斗场景"压出 top-5),哪怕 top-1 不一定是最优答案 chunk,LLM 看到的 5 个候选整体相关性更高,推理更准。这是把 reranker 当"top-1 提升器"会失望、当"top-k 降噪器"会满意的典型例子。

### 延迟分布

| 阶段 | 平均 | 中位 | 最大 | 占比 |
|---|---|---|---|---|
| **rerank(CPU,30 pair)** | **22.9s** | 22.8s | 24.95s | **51% ★** |
| LLM(Qwen3-8B,top-5 prompt) | 21.7s | 18.4s | 50.1s | 49% |
| total / 题 | **44.6s** | 41.0s | 73.0s | — |
| 30 题总耗时 | **22.3 分钟** | — | — | — |

A3 平均延迟是 A2 的 2.1 倍,几乎全部代价来自 reranker(rank-bm25 + jieba 加起来仍是 ms 级,被吃在精排上)。**这就是为什么生产 RAG 一定是两阶段**:粗排越多越好(便宜),精排越少越好(贵但准),让 reranker 只跑 ~30 个 pair。要想再快只能上 GPU(fp16 + batch_size 大概能压到 1-3s)或换更小的 `bge-reranker-base`。

---

## A3 vs A2 翻盘表

### 6 题提升(A2 → A3)

| Q# | 类型 | A2 → A3 | reranker 关键动作 |
|---|---|---|---|
| Q01 | 事实 | miss → hit | 把 ch002#15(蓝银草武魂段)从 RRF 中段顶到 top |
| Q06 | 事实 | miss → hit | top-1 从 ch002#17(对话噪声)换成 ch003#04(素云涛执事介绍) |
| Q10 | 关系 | partial → hit | top-1 换成 ch007#15,LLM 答出"同宿舍"=七舍舍友核心 |
| **Q11** | **关系** | **hallu → hit** | **本次实验的核心修复:ch009#14(罗三炮是变异武魂)RRF#5 → rerank#1,分数 0.990,4 个战斗场景被压出 top-5,LLM 不再被"4:1 多数信号"压垮** |
| Q15 | 跨章 | partial → hit | top-5 整体降噪,LLM 完整答出"曼陀罗蛇 + 四百年修为 + 不冲突" |
| Q24 | 模糊 | miss → hit | top-1 从 ch005#13 换成 ch010#17,直接命中"半级魂力 / 三十级" |

### 1 题翻车 —— 写进消融实验的关键反面案例

**Q18 跨章节型 partial → hallucinate**

- Query:"唐三初到诺丁学院第一天遇到了哪几次冲突?"
- expected:门房刁难(被大师阻止)+ 王圣给下马威(被唐三放倒)
- A2 (top-5 含 ch005#15):partial(说出王圣冲突,漏门房)
- A3 rerank 把 top-1 从 ch005#15 换成 ch008#19,**ch005#15(门房 + 王圣冲突所在 chunk)被压出 top-5**
- A3 答:"1. 王圣冲突;**2. 食堂高年级嘲讽;3. 与小舞切磋**"——后两点是 LLM 编出来的(原文无)

**根因**:cross-encoder 对"诺丁学院冲突"这个 query 给"宿舍 / 学院日常场景"chunk 普遍打高分,真正包含门房冲突的 ch005#15 反而被压下去。**reranker 不是免费午餐**——候选池里有正确 chunk 时它能精排,候选池本身偏了或 reranker 自身偏向"看起来相关"的 chunk 时,它会"自信地精排到错的地方",比 RRF 单路更难诊断。

**A4 怎么解**:这正是 contextual retrieval 的靶子。如果 ch005#15 embed 之前加了上下文前缀("本段讲唐三第一天进入诺丁学院遇到门房刁难和宿舍王圣冲突"),它在 dense / sparse 双路里都会排得更靠前,候选池就不会偏。

### Q3 静默退步(rerank top-1 没换但答案错)

Q3 "唐三在武魂觉醒时的魂力等级是多少?" A2 hit → A3 miss。RRF 和 rerank top-1 都是 ch003#15,但 top-2~5 被 reranker 重排打乱了 A2 时正好包含答案 chunk 的位置组合,**LLM 看到 5 个不同的上下文 chunk 后选择保守回答"原文未提及"**。提醒:top-1 不变不代表 prompt 不变,top-5 整体每动一次都可能改变 LLM 行为。

---

## A3 关键数字一句话总结

> **A3(Hybrid + bge-reranker-v2-m3 精排)在 30 题上准答率 86.7%(24 hit + 2 partial),比 A2 再提升 10 pp,核心战果是把 A2 的 Q11 hallucinate 彻底修复,模糊型 6/6 满分;代价是延迟从 20.9s 翻到 44.6s/题(reranker CPU 占 51%)。Q18 则成了关键反面案例 —— reranker 在候选池偏掉时会"自信地精排到错处",这正是 A4 contextual retrieval 要解决的下一个问题。**

---

## A4 Contextual RAG · 详细结果

### 流水线变化(相对 A3 的唯一区别)

```
                    A3:    chunk.text → embed / BM25 / reranker
A4:  ctx_prefix + chunk.text → embed / BM25 / reranker
     ↑
     DeepSeek-V4-Flash 离线生成
     (185 chunk × ~1.75s ≈ 5 分钟,¥0.25,缓存命中率 99.3%)

         LLM prompt 仍然只用 chunk.text 原文
         (避免"LLM 看自己写的总结再答题"的回声链路)
```

### 上下文前缀长什么样(实例)

| chunk | 原文开头 | LLM 生成的 ctx_prefix |
|---|---|---|
| `ch010#17` | "从大师眼中,唐三看到了狂热..." | "第九章末尾,唐三在猎魂森林中与大师合力击杀百年曼陀罗蛇后,大师鼓励唐三吸收这条蛇的魂环作为**第一魂环**..." |
| `ch005#15` | "唐三人虽然不大,可力量绝对不小..." | "第五章末尾,唐三与村长杰克抵达诺丁城,前往诺丁初级魂师学院报到..." |
| `ch006#02` | "说道这里,他的话听了下来..." | "第六章,大师在诺丁学院门口收唐三为徒的关键场景。大师通过武魂殿证明上的先天满魂力与蓝银草武魂的矛盾,**推断出唐三拥有双生武魂**..." |

prefix 把"第一魂环""双生武魂""诺丁学院报到"这种**概念性关键词**显式写进可检索文本 —— query 命中靠的不再是 chunk 自身的字面词,而是 LLM 提炼的语义标签。

### 检索层指标(章节级 top-1 命中,非陷阱题分母 29)

| 指标 | A2 | A3 | **A4** | A3→A4 |
|---|---:|---:|---:|---:|
| RRF top-1 | 20/29 (69.0%) | 20/29 (69.0%) | **23/29 (79.3%)** | **+10.3 pp** ★ |
| Rerank top-1 | — | 19/29 (65.5%) | **22/29 (75.9%)** | +10.4 pp |
| **Recall@5** | 28/29 (96.6%) | 28/29 (96.6%) | 28/29 (96.6%) | = |

**关键观察**:Recall@5 三代不变 —— 答案 chunk 几乎一直都在 top-30 候选池里。**Contextual prefix 真正做的是"把对的 chunk 从 top-30 顶到 top-1"**,不是"召回更多"。这跟 Anthropic 论文的 ablation 一致(prefix 的增益在精排层,不在召回层)。

### A3 → A4 RRF top-1 翻盘(纯检索层收益)

| Q# | 类型 | A3 RRF top-1 | A4 RRF top-1 | 解释 |
|---|---|---|---|---|
| Q06 | 事实 | ch002#17 ✗ | **ch003#04 ✓** | "素云涛是诺丁城武魂分殿执事" 这种身份信息,A3 chunk 自身只有人名,A4 prefix 显式写"素云涛执事介绍" |
| Q24 | 模糊 | ch005#13 ✗ | **ch009#14 ✓** | "大师为什么不传授魂力修炼" 这种解释类问题,A4 prefix 把"大师变异武魂导致终身无法突破三十级"作为关键概念抓出来 |
| **Q18** | **跨章** | **ch005#15 ✗**(被压下) | **ch005#15 ✓ (RRF top-1)** | **A3 关键反面案例的修复 ★** —— prefix 把 ch005 chunk 的"诺丁学院"地点信息显式标记,dense/sparse 双路都拉回来 |

### A3 → A4 Rerank top-1 翻盘

| Q# | 类型 | A3 Rerank | A4 Rerank | 备注 |
|---|---|---|---|---|
| Q02 | 事实 | ✗→✓ | ch005#05 → ch004#06 | |
| Q08 | 关系 | ✓→✗ | ch006#05 → ch007#14 | **唯一一题精排退步**:prefix 让 ch007 chunk 整体相关度上升,把对的 ch006 压下去 |
| Q09 | 关系 | ✗→✓ | ch008#19 → ch007#02 | |
| Q17 | 跨章 | ✗→✓ | ch007#18 → ch005#05 | |

净 +3 题(4 ✗→✓ 减 1 ✓→✗)。Rerank 翻盘比 RRF 翻盘多,因为 reranker 也看了 prefix(给 cross-encoder 多了 50-100 字语义信号)。

### Q18 修复详情(A3 的关键反面案例)

A3 时:rerank 把 ch005#15(门房+王圣 chunk)压出 top-5,LLM 编出"食堂高年级嘲讽 / 与小舞切磋"两个不存在的冲突 → **hallucinate**

A4 时:`ch005#15` 升到 RRF top-1,rerank 也保住 top-5(实际 rerank_top1 是 ch007#12,但 top-5 完整包含 ch005#15、ch006#11、ch006#09 等关键 chunk),LLM 答出:
> 1. 王圣挑衅并扑向唐三...(片段2)
> 2. **与门房冲突,唐三推倒门房后被大师制止**(片段3)
> 3. 王圣嘲笑大师实力,唐三严厉反驳(片段5)

LLM 准确答出 **门房 + 王圣 + 王圣嘲讽老师**三个冲突 → 大概率 **hit**(待用户手工确认),A3 的 hallucinate 被 contextual retrieval 彻底修复。

### 延迟分布

| 阶段 | A3 | **A4** | 增量 |
|---|---:|---:|---:|
| rerank(30 pair,CPU) | 22.9s | **26.3s** | +3.4s |
| LLM(Qwen3-8B,top-5 prompt) | 21.7s | 22.0s | +0.3s |
| **total / 题** | **44.6s** | **48.3s** | **+3.7s(+8%)** |

reranker 多 3.4s 是因为现在看的是 `prefix + text`(平均 ~600 字 vs A3 的 ~500 字),输入长 20%。LLM prompt 不动(还是看原文),所以 LLM 延迟几乎不变。

### A4 离线 ingestion 一次性成本

| 项 | 数值 |
|---|---|
| 模型 | DeepSeek-V4-Flash(¥1/M input,¥0.02/M cached,¥2/M output) |
| chunk 总数 | 185 |
| 总 prompt tokens | 8.92M(98.6% 命中隐式缓存) |
| 总 output tokens | 11K |
| **总成本** | **¥0.25** |
| 总耗时 | 5.3 min |
| 平均延迟 | 1.75s/chunk |

没有 prompt caching 的话同等工作量大约 ¥8.9,**实际省了 96.4%** —— 这就是为什么 Anthropic 当年发 contextual retrieval 时一定要配套 prompt caching,没缓存这套方法工程上不可行。

### 按类型细分(A3 vs A4)

| 类型 | 总数 | A3 (h/p/m/H) | A4 (h/p/m/H) | 准答率 A3→A4 |
|---|---|---|---|---|
| 事实型 | 6 | 5/0/1/0 | 5/0/1/0 | 83% → 83% = |
| 关系型 | 6 | 4/0/2/0 | 4/0/2/0 | 67% → 67% = |
| **跨章节型** | 6 | 3/2/0/**1** | 4/2/0/**0** | **83% → 100%** ↑↑(消除 hallu) |
| 模糊型 | 6 | 6/0/0/0 | 6/0/0/0 | 100% → 100% = |
| 陷阱型 | 6 | 6/0/0/0 | 6/0/0/0 | 100% → 100% = |

**核心观察**:A4 vs A3 的全部增量集中在**跨章节型**(+17 pp,Q18 hallu→hit),其他四类纹丝不动 —— 这跟 contextual retrieval 的设计意图完全吻合,prefix 的价值正是在"跨 chunk 关联"这一类问题上,事实型 / 关系型 / 模糊型 / 陷阱型 chunk 内自洽时 prefix 是过度工程。**这就是为什么在选 RAG 上下文增强方案时要先看 query 类型分布**,如果你的语料里很少跨章节关联(比如 API 文档、产品手册),contextual retrieval 的增量收益会显著低于这次实验。

---

### A4 退化点 —— 写进消融实验的小反面案例

**Q30 陷阱型 hit(A3 准答出"铁母")→ hit(A4 只说"原文未提及")**

- Query:"唐三在斗罗大陆制造的第一件袖箭,材料是普通生铁吗?"
- expected:不是,是经过百日锤击后变成的铁母
- A3 答:**"不是,是经过百日锤击后变成的铁母。"** ✓ 完整否定 + 给出正确答案
- A4 答:**"原文未提及"** ⚠️ 没掉坑(陷阱口径仍判 hit),但比 A3 信息少

**根因**:A4 召回的 top-5 chunk 偏向 ch005 早期段落(村长杰克对话),没把"百日锤击 / 铁母"那个核心 chunk 召回到 top-5。**为什么 prefix 反而帮倒忙了**:contextualize 阶段 LLM 给"百日锤击"chunk 写的 prefix 强调的是"唐昊指导唐三锻造铁块的过程",没把"铁母"这个最终产物的名词列为关键词;query 里"普通生铁"在 prefix 里也对不上(prefix 关注"锻造过程"不关注"材料分类")。

**启示**:contextual prefix 是 LLM 写的"语义索引",它的覆盖度取决于 LLM 选哪些关键词当语义锚点。如果 query 用的概念词没被 LLM prefix 抓到,反而会被压下去 —— A5 agentic 改写 query 是一种修法,Track B 用 PDF 章节标题/术语词典做硬约束 prompt 是另一种。

---

### A4 关键数字一句话总结

> **A4(Contextual prefix + Hybrid + Rerank)在 30 题上准答率 90.0%(25 hit + 2 partial),比 A3 再提升 3.3 pp,**核心战果是把 A3 的 Q18 hallucinate 彻底修复 ——** ch005 门房 chunk 从被 reranker 压出 top-5 变成 RRF top-1,LLM 准确答出门房 + 王圣 + 嘲讽老师三个冲突;hallucinate 数从 1 降到 0。检索层 RRF top-1 命中率从 69% 跃升到 79.3%(+10.3 pp),但 Recall@5 不变(96.6%)证明 prefix 真正在做"top 段排序优化"而非"召回更多"。延迟代价仅 +3.7s/题,离线 ingestion 一次性 ¥0.25。全部增量集中在跨章节型(83→100%),其他四类纹丝不动,印证 contextual retrieval 的设计意图。**

---

## A4.5 LLM 消融实验 · 验证瓶颈在生成层

### 实验设计

A4 实测 **Recall@5 = 96.6%**,但仍有 3 miss + 2 partial。手工分析 a4.jsonl 发现:正确 chunk 几乎全在 top-5 里,**Qwen3-8B 本地量化版没有合成出答案** —— 怀疑瓶颈在生成层而非检索层。

**唯一变量是 LLM,其他全部固定**:
- 检索结果 → 直接读 `runs/a4.jsonl` 的 `retrieved` 字段(top-5 一字不变)
- prompt 模板 → 复用 `contextual_rag.py` 的 `PROMPT_TEMPLATE`
- 关掉 DeepSeek 默认开启的 thinking(`extra_body={"thinking": {"type": "disabled"}}`)—— 简单 QA 不需要思考链,省 token 省钱

**入口脚本**:`a4_swap_llm.py`(108 行,核心就是读 a4.jsonl → `build_prompt` → 调 DeepSeek → 写 a4_swap.jsonl)。

### 结果(30 题逐题对照)

| 指标 | Qwen3-8B(本地) | DeepSeek-V4-Flash(云端) | Δ |
|---|---:|---:|---:|
| hit | 25 | **29** | +4 |
| partial | 2 | **1** | -1 |
| miss | 3 | **0** | -3 |
| hallu. | 0 | 0 | = |
| **准答率(hit + partial)** | 90.0% | **100.0%** | **+10.0 pp** |
| **严格准答率(hit only)** | 83.3% | **96.7%** | **+13.4 pp** |
| 平均延迟 / 题 | 22.0s | **1.09s** | **20× 加速** |
| 30 题总成本 | 本地免费(占 5GB 显存) | **¥0.0427** | — |

### 6 题判定翻动详情

| Q# | 类型 | Qwen3-8B 答 | DeepSeek-V4 答 | 判定变化 |
|---|---|---|---|---|
| Q03 | 事实 | "十级。" | "先天满魂力,即魂力达到先天最高的十级。" | **miss → hit ↑** |
| Q09 | 关系 | "原文未提及" | "诺丁初级魂师学院七舍的工读生,通过切磋确定排名" | **miss → hit ↑** |
| Q12 | 关系 | "原文未提及" | "唐三前一世的师门是唐门。" | **miss → hit ↑** |
| Q13 | 跨章 | "唐门跳崖明志,通过挤压过程穿越到陌生世界" | "**从唐门鬼见愁跳崖明志,穿越到斗罗大陆,以婴儿身份重生**" | **partial → hit ↑** |
| Q16 | 跨章 | "分析先天满魂力和蓝银草的特殊性,概率极低..." | "**调查过六百四十七个蓝银草武魂拥有者**,发现蓝银草无法产生先天满魂力" | **partial → hit ↑** |
| **Q18** | 跨章 | "王圣冲突 + 门房推倒被大师制止 + 嘲讽大师" | "王圣冲突 + 食堂高年级穷鬼之骂" | **hit → partial ↓** |

**净 +5 题翻盘 -1 题退步 = +4 题**,关键观察:

1. **三道 "原文未提及" 全部破解(Q3 / Q9 / Q12)** —— Qwen3-8B 看到 chunk 里"七舍工读生""唐门"这种关键名词却答"未提及",DeepSeek 同样的 top-5 直接抓出来。这就是本地 8B 量化版的归纳硬伤。
2. **Q16 的"六百四十七个"是答案的核心数字**,Qwen3-8B 看到 chunk 里"调查过六百四十七个武魂为蓝银草的人"原话,答出来的是"概率极低"这种二手抽象;DeepSeek 直接复述了原话。复述能力 ≠ 推理能力,Qwen3-8B 这里弱在前者。
3. **唯一退步 Q18**:DeepSeek 漏了"门房刁难"那次冲突,以"食堂高年级穷鬼之骂"代替。"门房"和"食堂"在 top-5 里都有原文,DeepSeek 选择列举哪两次是模型归纳风格的差异 —— 这不是 RAG 系统问题,是 prompt 没明确要求"列出全部冲突"导致的覆盖度不足。**A5 agentic decompose 能拆出子查询("唐三与门房的冲突?""唐三与王圣的冲突?")强制全覆盖**,这正好成了 A5 的设计目标。

### 按类型细分

| 类型 | 总数 | Qwen3-8B (h/p/m/H) | DeepSeek (h/p/m/H) | 准答率 Qwen3→DS |
|---|---|---|---|---|
| 事实型 | 6 | 5/0/1/0 | **6/0/0/0** | 83% → **100%** ↑ |
| 关系型 | 6 | 4/0/2/0 | **6/0/0/0** | 67% → **100%** ↑↑ |
| 跨章节型 | 6 | 4/2/0/0 | **5/1/0/0** | 100% → 100% =(内部 Q13/Q16 升 hit,Q18 退 partial) |
| 模糊型 | 6 | 6/0/0/0 | 6/0/0/0 | 100% → 100% = |
| 陷阱型 | 6 | 6/0/0/0 | 6/0/0/0 | 100% → 100% = |

关系型从 67% 跳到 100% 是本次实验最大增量 —— 关系类问题需要"在 chunk 里识别人物 + 抽出他们的角色绑定"这种 1-2 跳推理,8B 本地量化模型容易直接答"原文未提及"安全模式;DeepSeek 显然这个 size 之上做得更稳。

### 延迟 / 成本细账

| 项 | Qwen3-8B(本地) | DeepSeek-V4-Flash |
|---|---|---|
| 模型规模 | 8B(量化版约 5GB 显存) | 不公开,推测 30B+ |
| 调用方式 | WSL2 → Windows Ollama HTTP | OpenAI 兼容 SDK → DashScope-like 云 |
| 单题延迟 | 22.0s(generation 主导) | 1.09s(网络 + 推理) |
| 30 题总 token(本次实验) | — | input 45,099(hit 4,608 / miss 40,491),output 1,046 |
| 30 题总成本 | 本地免费 | ¥0.0427(input miss ¥0.0405 + output ¥0.0021 + hit ¥0.0001) |
| 30 题总耗时 | ≈ 11 分钟 | **32.7 秒** |

注:DeepSeek 隐式 KV 缓存命中率仅 10.2%,因为 30 题的 prompt 各不相同(只有 PROMPT_TEMPLATE 头部 + 部分重叠 chunk)。如果想刷高缓存命中,需要把 PROMPT_TEMPLATE + 通用片段提到 system message 固定,user message 只放 query。本次不优化,成本本来就足够便宜。

### 这次实验对 A5 的指导意义

按 PLAN.md A4.5 节预设的三分支,实测结果落在第一支:

> **4 题全 hit**:瓶颈确认在 LLM。A5 主要价值从"刷准答率"变成"练 LangGraph 工程模式"(Track B 主图必用),不追准答率提升

进一步具体化:
- **A5 不应该再用本地 Qwen3-8B**(没动力,生成天花板就在那),换用 DeepSeek-V4-Flash 作为节点 LLM;
- **A5 的 KPI 重心是"质量分布"而非"准答率"** —— 已经 100% 准答率没有上限刷的空间。可以盯三个其他指标:
  1. **平均答案完整度**(Q13/Q14/Q15 这类多要素问题,DeepSeek 当前漏的细节能不能补回来);
  2. **Q18 这种"列举遗漏"问题**(用 decompose 拆子查询测能否覆盖到 expected 列的所有冲突);
  3. **延迟 / token 成本** —— agentic 必然引入多轮调用,要量化 retry 次数和单题总 token,这才是 A5 真正的成本曲线。
- **检索层不必再动** —— Recall@5 = 96.6%,A5 即使把 query 改写一遍重检索,信息天花板也已经在 retriever 这里达到。LangGraph 在这个语料上是练工程,不是刷指标。

### A4.5 数据污染验证 · 闭卷对照实验

**为什么必须做这个**:《斗罗大陆》是中国头部网文 IP(2008-2014 连载,网络随处可见),DeepSeek 预训练几乎一定见过全本。A4.5 开卷实验里 DeepSeek 答对的题,**无法从答案表象上区分**是从 top-5 chunks 抓的还是凭训练记忆答的。

**实验设计**(脚本 `a4_closed_llm.py`):

- **不给任何 chunk**,直接把 30 道题丢给 DeepSeek,prompt 主动提示"基于你对这本小说的了解作答"(让它放心调用记忆,如果还说"不知道"就是真的不知道)
- 同一个模型(deepseek-v4-flash)、同样的温度(0.2)、同样关 thinking,确保唯一变量是"有没有 top-5 chunks"
- **联网搜索可以直接排除**:我们用的是 `https://api.deepseek.com/chat/completions` 普通对话端点,没有 `tools` 参数,不带 web search

**三方对比结果**:

| | A4 (Qwen3-8B 开卷) | A4.5 (DS 开卷) | **A4.5-closed (DS 闭卷)** |
|---|---:|---:|---:|
| hit | 25 | 29 | **12** |
| partial | 2 | 1 | **8** |
| miss | 3 | 0 | **10** |
| 准答率 (hit+partial) | 90.0% | 100.0% | **66.7%** |
| **严格 hit-only** | 83.3% | 96.7% | **40.0%** |
| 平均延迟 / 题 | 22.0s | 1.09s | 1.04s |
| 30 题成本 | 本地免费 | ¥0.0427 | ¥0.0038 |

**纯 RAG 真实贡献(开卷 - 闭卷)**:

- 准答率 100.0% - 66.7% = **+33.3 pp**
- 严格 hit-only 96.7% - 40.0% = **+56.7 pp** ← 关键数字

**即使排除预训练记忆,RAG 系统仍带来 57 pp 严格命中率提升 —— 污染存在但没有否定 A4.5 的核心结论。**

### 污染的三类直接证据

#### 一、铁证级污染(凭后文知识破解前 10 章陷阱)

| Q# | 闭卷答 | expected(基于前 10 章) | 暴露的问题 |
|---|---|---|---|
| Q26 | **"阿银"** | 前 10 章没给母亲名字,只称"三妹" | 阿银是后文揭示的母亲名,模型从全本知识答出 |
| Q28 | **"玉小刚"** | 前 10 章大师只以"大师"相称,没给真名 | 玉小刚是后文真名,前 10 章原文确认未提及 |
| Q8 / Q11 / Q14 / Q17 闲笔 | "...大师(玉小刚)..." | 不要求名字 | 答案里挂"玉小刚"= 模型每次答这类题都自动用后文信息补全 |

#### 二、记忆混乱(预训练记忆不等于 chunk 内容)

| Q# | 闭卷答 | 真实(前 10 章原文) | 性质 |
|---|---|---|---|
| Q2 | 蓝银皇 | 一柄通体乌黑的小锤子(=昊天锤) | **第二武魂完全答错**,把第一第二武魂搞混 |
| Q15 | 曼陀罗蛇,特性"增加魂力" | 黄色魂环,坚韧 + 毒性 | 蛇种对,特性全错 |
| Q16 | "左右手分别出现蓝银草和昊天锤" | 大师凭武魂证明上"先天满魂力 + 蓝银草"的矛盾推断 | 机制完全瞎编 |
| Q17 | "身体强度不足以同时修炼" | 唐昊和大师两次叮嘱"不要让人看到" | 原因瞎编 |
| Q18 | "门卫(小舞解围)+ 萧老大" | 门房(大师阻止)+ 王圣 | 角色错位:萧老大≠王圣,小舞当时未出场 |
| Q22 | "心脏 / 肺 / 肾" | 胸口心脏 + 两条小腿肌肉 | 三颗心脏指什么完全瞎编 |
| Q23 | "生命力顽强 / 易吸收能量 / 可进化 / 植物亲和" | 魂力消耗小 / 迷惑性大 / 方向丰富 / 不排斥魂环 | 4 点全错 |
| Q27 | "是的,战狮" | 不是,是战虎 | **反向证据**:模型反而不知道战虎 |

模型的预训练知识**部分准确 + 大量混乱**,这恰好说明 RAG 的不可替代性:**LLM 知道"罗三炮""唐门""父子"这种基础名词(任何中文 LLM 都知道),但具体情节细节、机制原因、数字、列举,记忆都不可靠**。

#### 三、引用片段编号扫描(弱证据)

`a4_swap.jsonl` 开卷答案里 **6/30 题(20%)** 出现"片段 X"编号引用(Q2 / Q4 / Q15 / Q18 / Q19 / Q24)。这只是弱证据 —— 我们的 prompt 没强制要求引用,大部分模型默认不会主动写。不能反推"没引用 = 凭记忆答",更不能反推"有引用 = 必从 chunks 抓"。这一项仅作信号参考。

### 按类型看 RAG 增量

| 类型 | Qwen3 开 → DS 开 → DS 闭 | RAG 增量(开 - 闭,hit+par) |
|---|---|---|
| 事实型 | 83% → 100% → 67% | +33 pp |
| 关系型 | 67% → 100% → 100%(但 hit 仅 3/6,3 题 partial) | +0 pp 表面 / **hit-only +50 pp** 内层 |
| **跨章节型** | 100% → 100% → **50%** | **+50 pp**(最依赖 RAG,多要素合成记忆最不可靠) |
| 模糊型 | 100% → 100% → 67% | +33 pp |
| 陷阱型 | 100% → 100% → 50% | **+50 pp**(闭卷凭后文知识破解了前 10 章边界,正是 RAG 给模型设边界的核心价值) |

**模糊型 / 基础设定型(Q20 十级魂师 / Q21 五级魂环颜色)闭卷也对** —— 这是斗罗大陆的世界观,任何中文 LLM 都背得熟。但这不影响 RAG 的价值评估:**RAG 的核心价值不在通用设定,在前 10 章这种"小颗粒度"事实和"前 N 章"这种边界控制**。

### 修正后的 A4.5 归因

A4.5 看到的 "Qwen3-8B 90% → DeepSeek 100%" 这 10 pp 提升,实际来自两个来源:

1. **真正的 RAG 阅读能力差距**(主导):同样的 top-5 chunks,DeepSeek 能复述出"六百四十七个蓝银草武魂者"(Q16)、"七舍工读生"(Q9)、"鬼见愁跳崖+婴儿身份重生"(Q13)这种 chunk 里的具体细节,Qwen3-8B 本地量化版直接答"原文未提及"。**这一层 RAG 不可替代,闭卷会答错**(Q9 闭卷"同学(后兄妹)"= partial,Q13 闭卷漏圣魂村/唐昊 = partial)。
2. **预训练知识的部分补全**(次要):Q12 "唐三前世师门 → 唐门" 这种闭卷也答对,严格说不需要 RAG。但同类题模型也会答错(Q2 第二武魂闭卷错成"蓝银皇")—— 不能反推 Q12 是凭记忆 vs 凭 chunk,只能说"凭记忆也能答对"。

### A4.5 一句话总结(污染修正版)

> **同一份 A4 top-5,LLM 从本地 Qwen3-8B 换成云端 DeepSeek-V4-Flash 后,30 题准答率 90% → 100%,严格 hit-only 83.3% → 96.7%,延迟 22s → 1.09s(20× 加速),成本 ¥0.0427。3 道"原文未提及"miss(Q3/Q9/Q12)和 2 道列举不全 partial(Q13/Q16)被 DeepSeek 全部翻 hit,唯一退步 Q18 漏门房改列食堂。**为排除"DeepSeek 见过《斗罗大陆》训练数据"的污染干扰,补做闭卷对照(`a4_closed.jsonl`):同模型不给 chunks 凭记忆答,严格 hit-only 仅 40%,准答率 66.7%,**纯 RAG 贡献 +56.7 pp**。陷阱题 Q26→"阿银"、Q28→"玉小刚" 暴露后文知识渗透,Q2/Q16/Q17/Q18/Q22/Q23 闭卷瞎编机制证明记忆不可靠;Q12 "唐门" 等基础名词闭卷也对,说明预训练对这类题有部分贡献。验证 PLAN.md A4.5 假设:A4 剩余瓶颈在生成层不在检索层,且 RAG 系统的主导贡献不被污染解释。A5 不刷准答率,改聚焦"答案完整度 / 列举覆盖度 / 延迟成本",节点 LLM 直接用 DeepSeek。**

---

## A5 Self-Reflective / Agentic RAG · 详细结果

### 流水线变化(相对 A4 / A4.5 的全部增量)

```
                          A4:  query → contextual retrieve(top-5)→ Qwen3-8B 一次性 generate
A4.5(LLM 消融):    query → 直接复用 a4.jsonl top-5      → DeepSeek-V4-Flash generate

A5(本节):
   query
     ↓
   [decompose]  ─── DeepSeek 判要不要拆 → sub_queries: list[str](简单题不拆 = [原 query])
     ↓
   [retrieve]  ←──┐ 复用 A4 contextual + hybrid + rerank,
     ↓            │ 对 sub_queries 中每条跑一遍 → 累加 top-5,按 (source, chunk_idx) 去重
   [grade]    ───┤ DeepSeek 严判 chunks 够不够 → {sufficient: bool, reason: str}
     │           │
     │ 够 / retries≥2 → [generate]  DeepSeek 用累计 chunks 合成最终答案 → END
     │           │
     │ 不够 → [rewrite]  本地 Qwen3-8B 改写 query(参考 grade 的 reason)
     │           ↓
     │           ┘ 回 retrieve(retries += 1,直到 MAX_RETRIES=2 强退)
```

**A5 vs A4.5 的设计区别**:A4.5 只换 LLM;A5 多了反思循环(grade + rewrite + 多轮 retrieve)和 decompose 拆解。混合 LLM 路由对齐 IEQ-Ops dissertation 的 specialist agent 模式,Track B 主图直接复用这个结构。

### 实测结果(三方对比)

| | A4(Qwen3-8B 开卷) | A4.5(DS 开卷) | **A5(LangGraph + DS + 本地)** |
|---|---:|---:|---:|
| hit | 25 | **29** | 26 |
| partial | 2 | 1 | **4** |
| miss | 3 | **0** | **0** |
| hallu. | 0 | **0** | **0** |
| 准答率 (hit + partial) | 90.0% | **100.0%** | **100.0%** |
| **严格 hit-only** | 83.3% | **96.7%** | **86.7%** |
| 平均延迟 / 题 | 48.3s | **1.09s** | 57.5s |
| 30 题总耗时 | ≈ 24 分钟 | **32.7 秒** | **28.8 分钟** |
| 30 题 LLM 成本 | 本地免费 | ¥0.0427(全 DS) | ¥0.0960(DS) + 本地免费 |
| 引入循环 / 拆解 | ✗ | ✗ | ✓(retries≥1 共 6 题) |

**核心数字**:A5 hit-only 86.7% **比 A4 高 +3.4 pp**(LLM 升级的收益),**但比 A4.5 低 -10 pp**(rewrite + decompose 引入的副作用)。详见下文翻盘表。

### A4 → A5 翻盘表(逐题级)

| Q# | 类型 | A4 → A5 | A5 路径 | 关键 |
|---|---|---|---|---|
| Q03 | 事实 | **miss → hit** | ret=0 | LLM 升级:Qwen3-8B "原文未提及" 被 DS 抓出"先天满魂力即十级" |
| Q09 | 关系 | **miss → hit** | ret=0 | LLM 升级:抓出"七舍工读生,切磋后小舞成新老大" |
| Q12 | 关系 | **miss → hit** | ret=0 | LLM 升级:直接答"唐门" |
| Q13 | 跨章 | **partial → hit** | ret=0 | LLM 升级:补全"跳鬼见愁 → 婴儿身份重生" |
| Q16 | 跨章 | **partial → hit** | ret=0 | LLM 升级:复述"六百四十七个蓝银草武魂者"原文数字 |
| Q10 | 关系 | **hit → partial** ↓ | **ret=2** | rewrite 两轮拉到 7 chunks,合成时漏"被唐三和小舞先后击败"关键关系 |
| Q14 | 跨章 | **hit → partial** ↓ | ret=0 **n_sub=2** | decompose 拆 2 个 sub_queries → 7 chunks,形态/玉石全对,漏"大师送给唐三的见面礼" |
| Q28 | 陷阱 | **hit → partial** ↓ | **ret=2** | "原文未提及" 主答案对,但漏关键反驳"大师自称已忘记自己名字" |
| Q30 | 陷阱 | **hit → partial** ↓ | ret=0 **n_sub=2** | 否定陷阱对,但材料细节"普通生铁锻造成铁母"略不准(应是"含铁母的生铁") |

**净 +1 题翻盘**:5 题升 hit(全部来自 LLM 升级,跟 A4.5 完全一致)- 4 题退 partial(全部 A5 独有,rewrite 或 decompose 副作用)= +1。

### 4 题 A5 独有的 partial 退步 —— 这是 A5 最有价值的反面案例

把 A4 / A4.5 都 hit、唯独 A5 退 partial 的 4 题摊开看,**两个独立的副作用机制**显形:

**机制 A:rewrite 循环让 chunks 累积,LLM 合成时被噪声稀释**

- **Q10 王圣和唐三是什么关系?** ret=2,chunks 累计 7
  - grade 第一次判 "缺王圣的具体身份和与唐三的互动详情" → rewrite 改写 → 又拉 2 个新 chunks → grade 再判不够 → 又 rewrite → 拉 2 个新 chunks → MAX_RETRIES 强退到 generate
  - generate 看到 7 个 chunks(混入"小舞切磋""老杰克"等无关人物的 chunk),**合成时优先答"舍友 / 头儿 / 同宿舍" 这种安全描述**,把核心"被唐三和小舞先后击败"漏掉
  - **对比 A4.5**:只看 A4 时 top-5 的精排前 5(已经选出最相关 5 条),DeepSeek 直接抓出击败叙事
  - **结论**:多轮 retrieve 的 chunks 累加,虽然 grade 觉得"信息更全",但 generate 反而陷入"信号噪声比恶化"

- **Q28 大师告诉唐三他的真名是什么?** ret=2,陷阱题
  - grade 严判帮助识别"找不到" → rewrite 两次都没能"问出"原文不存在的内容
  - generate 看到 chunks(包括大师自称"已经忘记了自己的名字"那段),**只答 "原文未提及" 略弱**,没有像 A4.5 那样答出"大师自称已经忘记"这个关键反驳
  - **原因**:多轮 chunks 累积后,generate prompt 太长,LLM 走简化路径不挖反驳细节

**机制 B:decompose 拆 sub_queries,并行检索引入"看似相关"的 chunks**

- **Q14 二十四桥明月夜是什么?是从哪里得到的?** ret=0,n_sub=2(decompose 拆成 "二十四桥明月夜是什么" + "如何得到")
  - 两个 sub_queries 独立跑检索,**并行拉来 7 个 chunks**(单 query 只会拉 5 个)
  - 多出来的 2 个 chunks 偏向"大师讲探险经历",**把"大师送给唐三的见面礼"这个关键事实的 chunk 排名压下去**
  - generate 合成时只答得出形态(腰带)、容量(每块 1m³)和来源(大师探险得到),漏掉"赠予"环节

- **Q30 袖箭材料是普通生铁吗?** ret=0,n_sub=2(decompose 拆成 "材料是什么" + "是不是普通生铁")
  - 两个 sub_queries 并行检索,top-5 偏向"锻造过程"chunks
  - LLM 看到的 chunks 主要讲"锻造过程 → 铁母 → 袖箭",**没强调"原本是含铁母的特殊生铁"这个材料来源**
  - generate 答 "普通生铁锻造成铁母" —— 措辞内部矛盾(否定的是普通生铁却又说普通生铁锻造),材料细节不准

**两个机制的共性**:**多 chunks 不等于答得更准**。chunks 越多,LLM 越倾向走安全 / 概括的合成路径,反而漏掉单 chunk 时能抓住的关键细节。这就是 PLAN.md A4.5 提到的"答案完整度问题",A5 不仅没解决,反而新增了。

### 按类型细分(A4 → A4.5 → A5)

| 类型 | 总数 | A4 (h/p/m/H) | A4.5 (h/p/m/H) | **A5 (h/p/m/H)** | hit-only 轨迹 |
|---|---|---|---|---|---|
| 事实型 | 6 | 5/0/1/0 | 6/0/0/0 | **6/0/0/0** | 83% → 100% → **100%** |
| 关系型 | 6 | 4/0/2/0 | 6/0/0/0 | **5/1/0/0** | 67% → 100% → **83%** ↓ (Q10) |
| 跨章节型 | 6 | 4/2/0/0 | 5/1/0/0 | **5/1/0/0** | 67% → 83% → **83%** (内部 Q13/Q16 升,Q14 退) |
| 模糊型 | 6 | 6/0/0/0 | 6/0/0/0 | **6/0/0/0** | 100% → 100% → **100%** |
| 陷阱型 | 6 | 6/0/0/0 | 6/0/0/0 | **4/2/0/0** | 100% → 100% → **67%** ↓↓ (Q28/Q30) |

**关系型 / 跨章节 / 陷阱型**是 A5 退步的三类,**全部因为引入额外 chunks 稀释合成**(机制 A 或 B)。事实型和模糊型 chunk 内自洽,A5 没干扰也没增益。

### Q18 vs A4.5 的对比(decompose 唯一成功的拆解收益)

A4.5 Q18 *"唐三初到诺丁学院第一天遇到了哪几次冲突?"* **partial**(漏门房改列食堂)—— 这是 A4.5 唯一的 partial,**正好是 PLAN.md A4.5 节预告 A5 要解决的"列举覆盖度"问题**。

A5 Q18 路径:**ret=1, n_sub=2**(decompose 拆成 "进门冲突" + "宿舍冲突"两个 sub_queries)
- 第一轮 retrieve 跑两个 sub_queries → 拉来 7 chunks
- grade 判不够 → rewrite "诺丁学院第一天遇到的所有冲突详细描述" → 第二轮 retrieve 再拉 chunks
- generate 看到累计 chunks,答出**门房刁难 + 王圣下马威 + 食堂高年级嘲讽** 三条,**门房和王圣两个核心都有,扩展到食堂(同当天的真实冲突)**
- **判定 hit** ✓

**这是 A5 设计意图唯一实现的题**:decompose 强制覆盖,rewrite 补丢失。但同样的机制在 Q10/Q14/Q28/Q30 上**适得其反**——多检索 = 多噪声。**这正是 agentic RAG 在简单 RAG 已 96.7% hit-only 时的边际困境:每多一道增量检索,引入的噪声比额外信息更多。**

### retries 分布

| retries | 题数 | 平均延迟 | hit | partial | miss |
|---:|---:|---:|---:|---:|---:|
| 0 | 24 | 33.9s | 22 | 2 | 0 |
| 1 | 1 (Q18) | 117.5s | 1 | 0 | 0 |
| 2 | 5 (Q10/Q25/Q27/Q28/Q29) | 159.0s | 3 | 2 | 0 |

**ret=2 的 5 题**:Q25/Q27/Q29 是陷阱题,grade 严判帮助识别"找不到"→ rewrite 改写两轮 → 最终 generate 诚实答(都 hit);Q10 是关系题,rewrite 副作用退 partial;Q28 是陷阱题,陷阱口径仍 hit 但漏反驳细节退 partial。**严格判口径下,rewrite 循环的命中率(3/5=60%)显著低于不触发 rewrite(22/24=91.7%)**。

### 节点 / 延迟 / 成本细账

| 节点 | LLM | 30 题累计调用 | 累计延迟 | 平均 / 次 | 占总延迟比例 |
|---|---|---:|---:|---:|---:|
| **retrieve** | 无 | **41 次**(30 题 + 11 次 rewrite 后) | **1215.5s** | 29.65s | **70.5%** ★ |
| **rewrite** | 本地 Qwen3-8B | 11 次 | **400.0s** | 36.36s | 23.2% ★ |
| grade | DeepSeek | 41 次 | 46.8s | 1.14s | 2.7% |
| generate | DeepSeek | 30 次 | 34.7s | 1.16s | 2.0% |
| decompose | DeepSeek | 30 次 | 28.0s | 0.93s | 1.6% |

**延迟两大头**:retrieve(reranker CPU 是大头)和 rewrite(本地 8B 慢),加起来 93.7%。**LLM 节点(decompose/grade/generate)只占 6.3%**,DeepSeek 推理几乎"免费"。

| Cloud LLM(DeepSeek) | 数值 |
|---|---:|
| 总调用 | 101 calls(30 decompose + 41 grade + 30 generate) |
| 总 input tokens | 131,503(其中 cache hit 43,136,**命中率 32.8%**) |
| 总 output tokens | 3,369 |
| **总成本** | **¥0.0960** |
| 平均成本 / 题 | ¥0.0032 |

| Local LLM(Qwen3-8B via Ollama)| 数值 |
|---|---:|
| 总调用 | 11 calls(rewrite 节点,触发 6 题 × 1-2 次) |
| 总延迟 | 400.0s |
| 平均 / 次 | 36.4s(本地一贯慢) |
| 成本 | 免费(本地显存) |

**A5 vs A4.5 成本对比**:A5 ¥0.0960 比 A4.5 单次 ¥0.0427 高 ~2.25 倍,主要来自:(1) grade 节点 41 次额外 LLM 调用;(2) 拼成 grade 的 prompt 含累积 chunks,token 量上去了。每题成本 ¥0.0032 仍极低,但延迟上 57.5s vs 1.09s 是 50× 退化(基本回到 A4 本地 Qwen3 的水平)。

### A5 的真实工程价值(刷不动指标,但落地了 LangGraph 模式)

**这一阶段已经不刷准答率**(A4.5 验证瓶颈不在检索)。A5 真正交付的是 4 个工程能力,Track B 主图会一一复用:

1. **LangGraph StateGraph + TypedDict + Annotated reducers**
   - `Annotated[list[dict], add]` 让 `chunks` 和 `trace` 在节点间自动 append 累加,不用手动 merge
   - 节点签名 `(state) → state_update` 强制每个节点只关心自己改动的字段
2. **静态边 + 条件边 + 循环 + 强退出**
   - `add_conditional_edges` + 路由函数实现 `grade → rewrite 或 generate` 分支
   - `MAX_RETRIES=2` 在路由函数里判,而不是节点内部判 —— 图的拓扑显式可见
3. **混合 LLM 路由(对齐 dissertation specialist 模式)**
   - 节点工厂 `make_*_node(llm)` 闭包注入不同 LLM,Cloud LLM 处理"判断 / 合成"类任务,Local 处理"短改写"
   - `cost_delta` 字段在 jsonl 里记录单题本地 / 云端 token / latency / 成本,Track B 评测可直接复用
4. **trace 字段串起整条决策路径**
   - 每个节点 append 一条 `{node, latency_sec, ...节点特定字段}` 到 `state.trace`
   - 调试 "为什么这题 rewrite 了 2 轮还失败" 时,jsonl 里直接看 trace 就行,不用插日志

**反过来,A5 暴露的 agentic RAG 工程坑**(进 Track B 前先记下):
- **不要无差别 retrieve 累加 chunks**。后续可以加 "rewrite 后只保留 top-K 个最相关 chunks" 或 "新 chunks 必须超过相似度阈值才纳入" 策略,避免噪声稀释
- **decompose 的 sub_queries 拆分要克制**。当前 prompt 让 DeepSeek 自由判断,实测 6 题拆了(20%),其中 Q14/Q30 拆解反而退步,只有 Q18 拆解收益。建议在 Track B 收紧 prompt 或加 sub_query 数量上限
- **grade 严判 + MAX_RETRIES=2 是必备组合**。grade 宽松会死循环,grade 严但无强退也会死循环 —— A5 实测陷阱题 5/5 都触发了 retries=2 强退,这道护栏证明必要

### A5 一句话总结

> **A5(LangGraph StateGraph + 混合 LLM 路由 + 多轮 retrieve + 反思循环)在 30 题上 hit=26 / partial=4 / miss=0 / hallu=0,严格 hit-only 86.7%,比 A4 高 +3.4 pp(LLM 升级红利),比 A4.5 低 -10 pp(rewrite 和 decompose 引入的合成稀释副作用)。5 题(Q3/Q9/Q12/Q13/Q16)A4→A5 翻 hit 全部来自 LLM 升级(跟 A4.5 完全一致),但 4 题(Q10/Q14/Q28/Q30)A4→A5 退 partial 揭示了两个独立的副作用机制:rewrite 多轮累积 chunks 让 generate 走安全合成路径(Q10/Q28),decompose 拆 sub_queries 并行检索引入"看似相关"的 chunks 压低关键事实排名(Q14/Q30)。唯一成功的 agentic 收益是 Q18(A4.5 partial → A5 hit),decompose 拆"进门"+"宿舍"两个 sub_queries 强制覆盖列举,验证了 PLAN.md 预告的"列举覆盖度"目标。延迟 57.5s/题(retrieve 70.5% + 本地 rewrite 23.2%),成本 ¥0.0960(101 次 Cloud 调用 + 11 次 Local 调用)。陷阱题 5/5 全拒答正确,Q27 完美纠正"战狮 → 战虎",grade 严判 + MAX_RETRIES=2 强退的护栏组合证明必要。A5 的工程交付价值(StateGraph + 混合路由 + trace 调试)直接接 Track B 主图,但 agentic 增检索的副作用要在 Track B 加约束(rewrite 后只保留 top-K、decompose 限制 sub_query 上限)。**

---

## 当前进度

- [x] A0 语料切分 + 30 题金标集
- [x] A1 朴素 RAG
- [x] A2 Hybrid Retrieval(Dense + BM25)
- [x] A3 Reranker 精排
- [x] A4 Contextual Retrieval
- [x] A4.5 LLM 消融实验(30 题 / Qwen3-8B vs DeepSeek-V4-Flash,准答率 90% → 100%,延迟 22s → 1.1s)
- [x] A4.5 数据污染验证(DeepSeek 闭卷对照,严格 hit-only 40%,纯 RAG 贡献 +56.7 pp)
- [x] **A5 Agentic RAG**(LangGraph + 混合 LLM 路由,hit-only 86.7%,延迟 57.5s/题,成本 ¥0.096)

---

## 附:复现这次实验

```bash
cd /home/xms/projects/rag/douluo

# A1 单题 verbose(7 步详细打印) / 批量评测
uv run python naive_rag.py --query "鬼见愁悬崖扔下一块石头要数几秒?"
uv run python naive_rag.py --eval --output runs/a1.jsonl

# A2 单题 verbose(8 步,可视化 dense/sparse/RRF 三阶段) / 批量评测
uv run python hybrid_rag.py --query "唐三的第一武魂是什么?"
uv run python hybrid_rag.py --eval

# A3 单题 verbose(9 步,Step 7 可视化 RRF→rerank 翻牌) / 批量评测
uv run python rerank_rag.py --query "罗三炮和大师是什么关系?"
uv run python rerank_rag.py --eval --output runs/a3.jsonl

# A4 离线生成 contextual prefix(DeepSeek-V4-Flash, ~5 min, ¥0.25)
uv run python contextual_ingest.py                          # 全量 185 chunk
uv run python contextual_ingest.py --inspect 10             # 从 SQLite 抽 10 条检查质量

# A4 单题 verbose(10 步,Step 0 加载 prefix,Step 6.5 看 prefix 内容) / 批量评测
uv run python contextual_rag.py --query "唐三初到诺丁学院第一天遇到了哪几次冲突?"
uv run python contextual_rag.py --eval --output runs/a4.jsonl

# A4.5 LLM 消融(复用 a4.jsonl 的 top-5,只换 LLM 为 DeepSeek-V4-Flash,~30s,¥0.04)
uv run python a4_swap_llm.py                                # 全量 30 题
uv run python a4_swap_llm.py --limit 3                      # 调试用

# A4.5 数据污染对照(DeepSeek 闭卷不给 chunks 凭记忆答,~30s,¥0.004)
uv run python a4_closed_llm.py                              # 全量 30 题

# A5 单题 verbose(LangGraph 流式打印每节点决策 + 最终答案 + 成本统计)
uv run python agentic_rag.py --query "唐三的第一武魂是什么"          # 简单题 ~28s
uv run python agentic_rag.py --query "海神九考都有哪些"             # 越界题触发 rewrite 循环 ~174s

# A5 批量评测(30 题,~29 分钟,¥0.096 Cloud + 本地 ~400s)
uv run python agentic_rag.py --eval --output runs/a5.jsonl

# A5 节点 mock(3a 验收,4 个 LLM 节点各调一次,看 schema)
uv run python agentic_rag.py --mock --llm-only              # 跳过检索栈,~30s
uv run python agentic_rag.py --mock                          # 加 retrieve 节点,~1 min
```

实际产物:
- `runs/a1.jsonl` — A1 每题 top-3 / answer / judgment / latency
- `runs/a2.jsonl` — A2 每题 top-5 / dense_top1 / sparse_top1 / fused_top1 / dense_rank / sparse_rank / fusion_source / judgment / latency
- `runs/a3.jsonl` — A3 每题 top-5(带 rerank_score / rrf_rank)/ rrf_top1 / rerank_top1 / promoted_top1 / latency_rerank_sec / latency_llm_sec / judgment
- `runs/a4.jsonl` — A4 每题 top-5(每个 chunk 多带 ctx_prefix)/ 其他字段同 a3 / judgment(待标注)
- `runs/a4_swap.jsonl` — A4.5 LLM 消融每题 answer_qwen3 / answer_deepseek / judgment_qwen3 / judgment_deepseek / latency_*_sec / usage_deepseek
- `runs/a4_closed.jsonl` — A4.5 闭卷对照每题 answer_deepseek_closed / judgment_closed / latency_sec / usage
- `runs/a5.jsonl` — A5 每题 final_answer / sub_queries / retries / retrieved(带 retrieve_round / from_subquery)/ trace(完整决策路径)/ cost_delta(本地+云端 token/latency/¥)/ judgment
- `cache/ctx.sqlite` — A4 离线产物:185 条 (source, chunk_idx, ctx_prefix, usage, latency)
