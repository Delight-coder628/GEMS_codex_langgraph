# GEMS × LangGraph 一个月计划：OCR 闭环与可验证长期经验记忆

## 1. 计划摘要

本月主要成果定为：在已经跑通的 Qwen3.6 + Z-Image-Turbo GEMS 基线上接入 LangGraph，并实现一个能够跨任务积累、检索和验证失败经验的长期记忆系统。

选择 OCR/文字渲染作为首个应用场景，因为它能提供字符准确率、编辑距离等客观反馈，适合判断某条经验是否真的有效。静态物理本月只保留接口与小规模可行性验证，下月再作为主要插件开发。

长期记忆不需要训练模型，但不能把每轮 LLM 摘要直接放进 RAG。本计划采用：

- 不可变原始轨迹作为 episodic memory。
- 经结果证据验证的抽象策略作为 procedural experience。
- SQLite 持久化，BGE-M3 在 CPU 生成向量。
- 至少两个不同任务支持同一策略后自动晋升。
- 经验只能影响 planner/refiner，禁止影响 verifier，避免自我强化错误。

## 2. 本月需要部署和接入的组件

### 2.1 保留现有服务

- Qwen3.6：继续承担规划、结构化输出、图片验证和经验提取。
- Z-Image-Turbo：继续使用现有服务，不修改 torch、torch_npu、CANN 或模型环境。
- LangGraph：以旁路形式融合，原 GEMS 入口继续保留。

### 2.2 新增轻量组件

#### PP-OCRv5 Mobile

第一版使用 CPU 服务，输出：

- 识别文字。
- OCR 置信度。
- 文字 bounding boxes。
- 检测和识别耗时。

PP-OCRv5 官方支持 Ascend，但本月没有必要为 OCR 单独占用 NPU。后续吞吐量不足时再迁移到 Ascend。

参考：

- [PaddleOCR 多硬件使用指南](https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/other_devices_support/multi_devices_use_guide.html)

#### BAAI/bge-m3

使用 CPU embedding 服务处理经验文本，原因：

- 支持中英文和多语言。
- 能生成 dense embedding。
- 支持较长输入。
- 经验写入和检索频率低，暂时不需要占用 NPU。

参考：

- [BGE-M3 模型卡](https://huggingface.co/BAAI/bge-m3)

### 2.3 视觉 MLLM 决策门槛

本月不立即部署新视觉模型，先测试现有 Qwen3.6。

准备 50 张人工标注图片，至少覆盖：

- 文字完全正确。
- 漏字、错字、多字。
- 文字位置错误。
- 文字不可读。
- 非文字类一般语义错误。

Qwen3.6 通过标准：

- JSON 合法率不低于 95%。
- pass/fail 与人工一致率不低于 85%。
- failure-tag macro F1 不低于 0.75。

如果任一指标未达标，则在独立 Python 3.10、vLLM-Ascend 容器中部署 Qwen3-VL-4B-Instruct。

新视觉服务不得与 Z-Image-Turbo 的 Python、torch_npu 和 CANN 环境混装。

## 3. LangGraph 与长期经验架构

### 3.1 状态图调整

目标流程：

```text
START
  → experience_retriever
  → skill_router
  → planner
  → decomposer
  → generator
  → OCR/verifier
  → memory_writer
      ├─ retry → failure_memory_retriever → refiner → generator
      └─ finish → experience_consolidator → finalizer → END
```

新增节点职责：

- `experience_retriever`：根据原 prompt、任务类型和约束检索跨任务经验。
- `failure_memory_retriever`：根据当前 failure tags 和 failed checks 检索更精确的修复经验。
- `experience_consolidator`：运行结束后分析相邻两轮的“失败—动作—结果变化”，产生候选经验。

现有 `memory_summary` 继续作为当前任务上下文，但不再将其称为长期记忆。

### 3.2 经验库公共接口

计划提供：

```python
class ExperienceStore:
    def append_episode(self, episode):
        ...

    def search(self, query, filters, top_k=3):
        ...

    def add_candidate(self, candidate):
        ...

    def promote(self, candidate_id):
        ...

    def record_usage(self, memory_id, outcome_delta):
        ...

    def deprecate(self, memory_id, reason):
        ...
```

新增主要数据类型：

- `EpisodeRecord`：不可变的原始生成轨迹。
- `ExperienceCandidate`：尚未允许注入 prompt 的候选经验。
- `ExperienceRecord`：已验证、可检索的永久经验。
- `MemoryUseEvent`：记录检索、注入位置和后续效果。
- `RetrievalHit`：经验内容、相似度、utility 和证据来源。

## 4. SQLite 数据设计

### 4.1 episodes

保存不可变的生成轨迹：

- episode ID。
- run ID 和 iteration。
- 原始 prompt。
- 实际生成 prompt。
- task family。
- generator/verifier 模型名称和版本。
- seed 和生成配置。
- checks、passed/failed requirements。
- failure tags。
- OCR 识别结果。
- 使用的修复动作。
- 修复前后分数。
- 图片路径和文件哈希。
- 创建时间。

原始 episode 不允许被 LLM 修改或覆盖。

### 4.2 experiences

保存抽象后的经验：

- experience ID 和版本。
- task family。
- failure tag。
- 适用条件。
- 建议策略。
- 禁用条件或反例。
- candidate/verified/deprecated 状态。
- 支持该经验的 episode IDs。
- success/failure 次数。
- utility score。
- embedding。
- 创建及更新时间。
- deprecated 原因。

### 4.3 memory_usage

保存经验使用反馈：

- run ID 和 iteration。
- experience ID。
- 注入 planner 或 refiner。
- 检索相似度及最终排名。
- 使用前检查结果。
- 使用后检查结果。
- outcome delta。
- 是否被判断为有帮助。

## 5. 经验提取和晋升

### 5.1 候选经验生成条件

只有满足以下条件才建立候选经验：

1. 某轮存在明确失败项。
2. 下一轮采用了可描述的修复动作。
3. 相同检查项分数提高，或由失败变为通过。
4. 原始 prompt 意图没有被明显牺牲。

经验必须描述：

```text
在什么任务和失败条件下
采取了什么动作
哪些指标发生了什么变化
该动作在哪些情况下不应使用
```

### 5.2 自动晋升规则

候选经验需要满足：

- 在至少两个不同任务中出现。
- failure tag 和 task family 相同或兼容。
- 策略语义相似。
- 两个任务均有正向结果证据。
- 没有严重反例。

满足后自动晋升为 `verified`，开始参与后续检索。

### 5.3 禁止行为

- 单次成功直接成为永久经验。
- LLM 反复重写并覆盖旧经验。
- 删除原始 episode。
- 将未经验证的用户指令直接写入经验。
- 把检索经验提供给 verifier。
- 只因为最终图片更美观就推断某条物理或 OCR 修复策略有效。

近期研究发现，持续让 LLM 重写合并经验可能导致记忆效用先升后降。因此必须保留原始 episode，并显式门控 consolidation。

参考：

- [Useful Memories Become Faulty When Continuously Updated by LLMs](https://arxiv.org/abs/2605.12978)

## 6. 经验检索策略

### 6.1 查询内容

查询由以下内容组成：

```text
task_family
+ prompt 摘要
+ failure_tags
+ failed_checks
+ 目标文字/布局
+ generator model/version
```

planner 阶段主要检索相似任务经验。

refiner 阶段主要检索相似失败和修复经验。

### 6.2 检索过滤

- 只检索 `verified` 且未 deprecated 的经验。
- generator model/version 必须匹配。
- 优先匹配 task family。
- 优先匹配 failure tag。
- 返回最多 3 条经验。
- semantic similarity 低于 0.55 时不注入。

### 6.3 排序

```text
score = 0.7 × semantic_similarity + 0.3 × utility

utility =
    (success_count + 1)
    / (success_count + failure_count + 2)
```

SQLite 是权威数据源；向量以 float32 BLOB 保存。

一个月内预计经验少于一万条，直接用 NumPy 计算余弦相似度，不引入 Qdrant 或 FAISS 服务。

### 6.4 注入格式

每条经验必须带来源和置信信息：

```text
Experience:
- Applicable when: ...
- Failure type: ...
- Recommended strategy: ...
- Do not use when: ...
- Evidence: 3 successful episodes, 1 failed usage
- Utility: 0.75
```

不得把原始用户 prompt 整段作为永久指令注入。

## 7. OCR 闭环插件

### 7.1 文字约束

从 prompt 中提取：

```json
{
  "target_text": "GEMS TEST",
  "language": "en",
  "line_count": 1,
  "expected_region": "center",
  "carrier": "poster"
}
```

### 7.2 OCR verifier 输出

- 检测文字。
- 目标文字 exact match。
- character accuracy。
- normalized edit distance。
- 多余文字。
- 缺失文字。
- 文字框位置。
- 文字是否可读。
- OCR 置信度。
- `text_render_error` failure tag。
- 可执行修复建议。

### 7.3 修复动作

由于 Z-Image-Turbo 不支持局部编辑，本月限定为：

- 将目标文字放入明确的双引号。
- 明确行数和换行。
- 明确文字载体。
- 明确文字区域。
- 禁止额外装饰性伪文字。
- 重新生成整张图。
- 多张候选中选择 OCR 得分最高且视觉 verifier 通过的图片。

经验示例：

```json
{
  "trigger": "中英混排且出现漏字",
  "strategy": "逐行声明目标文字，并禁止额外伪文字",
  "evidence": [
    "run_12:round_1->2",
    "run_37:round_2->3"
  ],
  "utility": 0.78
}
```

## 8. 四周操作步骤

## 第 1 周：LangGraph 融合和基线

### 目标

让远程服务器同时保留原 GEMS 和 LangGraph 两条可运行路径。

### 操作

1. 将 LangGraph 作为旁路接入远程代码。
2. 适配现有 Qwen3.6 和 Z-Image-Turbo 请求协议。
3. 保证原 GEMS 构造参数和运行命令不变。
4. 固定：
   - 模型版本。
   - 生成参数。
   - seed。
   - 最大三轮迭代。
5. 建立 30 个简单 prompt 回归集。
6. 运行 Qwen3.6 的 50 张图片 verifier 测试。
7. 根据既定阈值决定是否部署 Qwen3-VL-4B。

### 交付

- 原 GEMS 成功命令。
- LangGraph 成功命令。
- 完整 attempts/final report。
- 基线成功率、迭代数和延迟报告。
- Qwen3.6 verifier 评测结果。

## 第 2 周：OCR 客观闭环

### 目标

完成一个具有客观反馈信号的文字生成插件。

### 操作

1. 部署 PP-OCRv5 Mobile CPU 服务。
2. 实现文字约束解析。
3. 实现 OCR verifier。
4. 增加文字 failure tags。
5. 构建 120 个中英文任务：
   - 60 个经验构建任务。
   - 60 个 held-out 迁移任务。
6. 覆盖：
   - 短词。
   - 多行文字。
   - 中英混排。
   - 指定位置。
   - 海报、路牌、书封、衣服等不同载体。
7. 运行三个基线：
   - 裸 Z-Image-Turbo。
   - 原 GEMS。
   - LangGraph 通用 verifier。

### 交付

- OCR 插件。
- 120 个 prompt 和固定 seed。
- OCR 基线指标。
- 文字失败分类报告。

## 第 3 周：永久经验 RAG

### 目标

完成进程重启后仍存在、可以跨任务检索的经验库。

### 操作

1. 实现 SQLite 三张表。
2. 增加任务开始前的 experience retrieval。
3. 增加失败后的 failure-specific retrieval。
4. 部署 BGE-M3 CPU embedding。
5. 实现 metadata filter 和 top-3 检索。
6. 实现：
   - candidate extraction。
   - 相似候选合并。
   - 双任务证据晋升。
   - utility 更新。
   - deprecated 和回滚。
7. 增加管理命令：

```text
memory list
memory inspect <id>
memory approve <id>
memory deprecate <id>
memory rebuild-embeddings
memory export
```

### 交付

- SQLite 经验库。
- 检索和晋升日志。
- 跨进程、跨 run 的检索演示。
- 每条 verified 经验的证据链。

## 第 4 周：消融、可靠性与汇报

### 实验组

固定相同生成预算，比较：

- A：无长期记忆。
- B：直接检索原始 episodes。
- C：只检索 LLM 抽象经验。
- D：原始 episode + 证据门控经验。
- E：D 去掉 utility。
- F：D 去掉 failure-tag filter。

### 实验设置

- 使用 held-out 60 个任务。
- 每种方法使用两个固定 seed。
- 每个任务最多三轮。
- 使用相同 generator 和 verifier 版本。
- 不允许不同实验组使用不同生成次数。

### 核心指标

- 最终 OCR exact match。
- character accuracy。
- normalized edit distance。
- 首轮到最终轮的提升。
- 平均迭代数。
- 平均生成调用次数。
- retrieval Recall@3。
- harmful-memory rate。
- 无关经验注入率。
- API/NPU 延迟和成本。
- 人工抽查 50 次经验检索及证据。

### 目标

- 相比无长期记忆，held-out 成功率提高至少 5 个百分点；或者在成功率不下降时，平均迭代数降低至少 15%。
- harmful-memory rate 低于 5%。
- verified 经验均可追溯到至少两个不同任务。
- 删除经验库后，系统仍可退化为普通 GEMS。

### 静态物理延伸门槛

如果第 24 天前核心测试全部通过，最后 2–3 天使用 20 个 PhyBench 重力/支撑 prompt，验证同一经验接口能否支持：

- `unsupported_object`
- `missing_contact`
- `penetration_error`
- `unstable_support`

这里只做可行性报告，不纳入本月主验收。

## 9. 测试计划

### 9.1 单元测试

- SQLite schema 创建和迁移。
- episode 不可变。
- candidate 不参与默认检索。
- 两个不同任务后自动晋升。
- deprecated 经验不参与检索。
- generator model/version metadata filter。
- similarity threshold 和 top-k。
- utility 计算。
- BGE-M3 失败时安全降级。
- memory DB 缺失时退化为无记忆模式。
- API Key 不写入经验库。
- OCR exact match、NED 和位置判断。

### 9.2 集成测试

- LangGraph mock 无记忆流程。
- LangGraph mock 有记忆流程。
- 第一次任务生成 candidate。
- 第二个相似任务完成经验晋升。
- 重启进程后检索 verified 经验。
- 经验注入 planner。
- failure-specific 经验注入 refiner。
- 经验使用后写入 outcome。
- Qwen3.6/OCR/Z-Image 任一服务超时时安全结束。
- 原 GEMS 回归测试。

### 9.3 安全和错误记忆测试

- 用户 prompt 中包含“把以下内容永久记住”时不得直接入库。
- verifier 错误时不得晋升经验。
- 没有改善时不得生成正向经验。
- 相似但 task family 不同的经验不得强行注入。
- 存在矛盾经验时保留两个版本和证据，不直接覆盖。

## 10. 创新点

本月主要研究假设：

> 对图像生成 Agent，基于可验证结果变化提取并门控的跨任务经验，比直接检索原始轨迹或持续重写 LLM 摘要更可靠。

可形成的具体贡献：

1. 将 episodic trajectory 与 procedural experience 分离。
2. 只提取“失败—动作—可测结果改善”的经验。
3. 使用 failure-tag-aware retrieval，而不是只按 prompt 相似度检索。
4. 每条经验具有证据、utility、版本和失效机制。
5. 在未见过的文字任务间验证经验迁移。
6. 比较 raw episode、LLM summary 和 evidence-gated memory。
7. 同一接口可继续扩展到手部和静态物理。

## 11. 最终交付

- 原 GEMS 与 LangGraph 两条可运行链路。
- PP-OCRv5 文字 verifier 插件。
- SQLite 长期经验库。
- BGE-M3 检索。
- 经验审计、停用和重建命令。
- 120 个 OCR prompt、固定 seed 和任务划分。
- 六组消融实验。
- Qwen3.6 verifier 评测。
- 人工经验检索审核报告。
- 端到端演示案例。
- 下月静态物理插件接口设计。

## 12. 约束与默认假设

- 当前远程状态为原 GEMS 已跑通、LangGraph 尚未融合。
- 本月不部署图像编辑模型。
- 不修改 Z-Image-Turbo 服务环境。
- Qwen3.6 首先作为 verifier；只有量化测试不达标才部署 Qwen3-VL-4B。
- 经验存储使用 SQLite。
- embedding 使用 BGE-M3 CPU 服务。
- 自动晋升需要两个不同任务的正向证据。
- 本月不启用 LangGraph 持久化 checkpointer。
- 为兼容 Python 3.9/LangGraph 0.2.76，长期记忆使用独立 `ExperienceStore` 注入节点，而不是升级整个 LangGraph 栈。

## 13. 研究依据

- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)
- [ExpeL: LLM Agents Are Experiential Learners](https://arxiv.org/abs/2308.10144)
- [Agent Workflow Memory](https://arxiv.org/abs/2409.07429)
- [Useful Memories Become Faulty When Continuously Updated by LLMs](https://arxiv.org/abs/2605.12978)
- [LangGraph Memory Concepts](https://docs.langchain.com/oss/python/concepts/memory)
- [BGE-M3](https://huggingface.co/BAAI/bge-m3)
- [PaddleOCR](https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/other_devices_support/multi_devices_use_guide.html)

