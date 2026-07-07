# 当前窗口交接摘要：GEMS × LangGraph 图像生成 Agent

本文档用于在新 Codex 窗口中快速恢复上下文，只保留关键背景、已完成产物和下一步建议。

## 1. 项目背景

- 当前仓库是 GEMS 相关代码，工作目录为 `D:\.A_Coding\GEMS\GEMS_codex_langgraph`。
- 用户已在公司远程服务器跑通原始 GEMS 基本流程：
  - 主 MLLM：公司内部 Qwen3.6 API。
  - 生图模型：Z-Image-Turbo。
  - 硬件：华为昇腾 910B NPU，至少 8 张卡，单卡约 32GB，ModelArts 平台。
- 实习课题方向：多模态 Agent 框架设计，聚焦图像生成/编辑领域；目标是证明“Agent 框架包裹裸模型后，生成/编辑效果有可观测提升”。
- 短期不追求大而全，建议先从可量化的小点切入：文字渲染 / OCR 闭环；手部和静态物理逻辑作为后续插件方向。

## 2. 已确定的技术路线

- 保留原始 GEMS 流程作为 baseline。
- 新增 LangGraph 旁路入口，与 GEMS 的 Agent Loop、Memory、Skill 思路对齐。
- 第一阶段只做文生图，不做真实图像编辑。
- 生成端只使用 Z-Image-Turbo。
- 主 MLLM 的 `base_url`、`api-key`、`model name` 需留空并从环境变量读取，避免硬编码公司配置。
- LangGraph 第一版建议使用：
  - Python 3.9.x
  - `langgraph==0.2.76`
  - 不启用持久化 checkpointer
  - 每次运行轨迹保存到本地 JSONL / JSON 文件
- 910B 使用方式：
  - 运行时指定一张或两张 NPU 卡。
  - 双卡时每张卡加载一份完整 Z-Image-Turbo 模型副本，并行处理不同请求。
  - 不做单请求跨卡模型并行。

## 3. 已生成的文档

当前仓库中与本次工作相关的文档：

- `GEMS_LANGGRAPH_GUIDE.md`
  - LangGraph 版本 GEMS 框架的整体运行说明。
- `GEMS_LANGGRAPH_MERGE_GUIDE_FOR_AGENT.md`
  - 写给公司内部 Agent 的融合指导文档，用于把 LangGraph 修改合入已经跑通的 GEMS 代码。
- `GEMS_ONE_MONTH_OCR_EXPERIENCE_MEMORY_PLAN.md`
  - 最近一个月的具体计划：以 OCR / 文字渲染闭环和长期经验记忆为主线。
- `CURRENT_WINDOW_HANDOFF_SUMMARY.md`
  - 本文件，用于新窗口快速接续。

## 4. 一个月计划的核心建议

推荐下个月主线是：

> 先实现“文字渲染质量提升”的可验证闭环，再接入长期经验记忆 RAG。

原因：

- 文字渲染比“审美质量”更容易客观评估。
- OCR 可提供明确指标，例如 exact match、编辑距离、OCR confidence。
- 适合展示 Agent 相对裸模型的提升：规划 prompt、生成、OCR 验证、失败归因、refine、再生成。
- 成熟模块较多，但把它们整合为可复用、可追踪、可长期积累经验的图像生成 Agent，仍有工程和课题价值。

建议接入：

- OCR：PP-OCRv5 Mobile，第一版可 CPU 部署。
- Embedding / RAG：BGE-M3，可 CPU 部署。
- 视觉验证：
  - 先继续使用 Qwen3.6 vision API。
  - 若公司 API 视觉能力不足，再考虑本地部署 Qwen3-VL-4B 或同级别 VLM。

## 5. 长期经验记忆设计

mentor 提到的“让 Agent 记住每次生成图片的失败经验，并转化为永久经验”是可行的，但建议不要直接把所有失败写进 prompt。

推荐分两层：

1. Episodic Memory
   - 保存每次运行的原始轨迹。
   - 包括输入 prompt、生成图、OCR / MLLM 检查结果、失败标签、refine prompt、耗时等。
   - 不直接作为经验使用。

2. Procedural Experience
   - 从多次 episode 中提炼出的抽象经验。
   - 只有在有量化证据证明有效时才晋升。
   - 示例：当目标文字较长时，应把文字拆短、放大、提高对比度、减少背景干扰。

关键原则：

- 经验只影响 planner / refiner，不影响 verifier，避免自我强化。
- 经验需要证据门控：至少在不同任务上多次有效，才写入长期经验库。
- 建议使用 SQLite 存结构化记录，向量检索只存经验摘要 embedding。

## 6. 建议的 LangGraph 节点演进

基础流程：

```text
START
→ skill_router
→ planner
→ decomposer
→ generator
→ verifier
→ memory_writer
→ conditional route
→ refiner / finalizer
→ END
```

下个月建议新增：

```text
experience_retriever
failure_memory_retriever
experience_consolidator
```

推荐新流程：

```text
START
→ skill_router
→ experience_retriever
→ planner
→ decomposer
→ generator
→ verifier
→ memory_writer
→ conditional route
→ failure_memory_retriever
→ refiner
→ generator
...
→ experience_consolidator
→ finalizer
→ END
```

## 7. 四周推进节奏

### 第 1 周：LangGraph 融合

- 把 LangGraph 旁路入口接入已跑通的 GEMS 代码。
- 保留原 `infer.py` 或原始入口作为 baseline。
- 统一输出目录，例如 `outputs/langgraph_runs/{run_id}/`。
- 确保每轮保存：
  - 输入 prompt
  - 实际生成 prompt
  - 图片
  - 检查结果
  - refine 记录
  - final report

### 第 2 周：OCR 闭环

- 部署 PP-OCRv5 Mobile。
- 增加 text-rendering skill。
- verifier 中加入 OCR 检查：
  - 是否识别到目标文字
  - exact match
  - 编辑距离
  - OCR confidence
- 让 refiner 根据 OCR 失败原因改 prompt。

### 第 3 周：长期经验记忆

- 建 SQLite 经验库。
- 接入 BGE-M3 embedding。
- 实现经验检索和经验晋升逻辑。
- 对比：
  - 无经验
  - 仅当前上下文经验
  - 长期经验 RAG

### 第 4 周：实验和汇报

- 做小规模 benchmark。
- 记录成功率、平均轮次、耗时、NPU 占用。
- 做消融实验。
- 整理实习阶段性汇报。

## 8. 当前不建议优先做的事情

- 不建议一开始就做完整图像编辑。
- 不建议立刻重训模型或 LoRA。
- 不建议把静态物理逻辑作为第一个主任务。
- 不建议同时上手部、文字、物理三个方向，否则很容易变成“模块拼接但没有清晰指标”。

## 9. 后续可扩展方向

- 手部质量：
  - 可接 MediaPipe / Hand detector / VLM 检查。
  - 更适合作为第二个 skill。
- 静态物理逻辑：
  - 可从光影、重力、支撑关系、镜面反射等局部可验证规则切入。
  - 第一阶段可做“物理约束检查器 + prompt 修正器”，先不训练。
- 更强 MLLM：
  - 如果 Qwen3.6 / GLM4.7 API 对图像细节验证不稳定，再考虑本地部署轻量 VLM。
  - 但本月优先把 OCR 这个客观验证链路跑通。

## 10. 新窗口建议第一步

在新窗口中可以直接说：

> 请先阅读 `CURRENT_WINDOW_HANDOFF_SUMMARY.md`、`GEMS_LANGGRAPH_MERGE_GUIDE_FOR_AGENT.md` 和 `GEMS_ONE_MONTH_OCR_EXPERIENCE_MEMORY_PLAN.md`，然后帮我检查当前仓库，制定下周把 LangGraph 融合进已跑通 GEMS 代码的具体文件级改造计划，先不要改代码。

