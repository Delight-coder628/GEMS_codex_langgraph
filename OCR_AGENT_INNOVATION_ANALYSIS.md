# 图像生成 Agent 的 OCR 创新点分析

> 面向中英海报与自然场景文字的三个月论文预研路线
> 更新时间：2026-07-11
> 结论性质：本地参考代码审计 + 公开论文/官方仓库调研 + 待实验验证的研究假设

## 摘要

当前本地 GEMS × LangGraph 参考实现已经具备一个可运行的文字验证闭环：从提示词提取目标文字，调用 OCR，对目标字符串与全图 OCR 拼接文本计算 exact match、substring match、编辑距离、归一化 similarity 和平均置信度；如果失败，则增加统一的 `text_render_error` 标签和修改建议，再进入 prompt refiner 与重新生成。这个实现适合作为工程起点和实验基线，但它仍然是一个“结果门禁”，还不是能定位、解释并修复错误的主动 OCR critic。

公开研究表明，**OCR 打分、字形/位置条件、文字局部编辑、OCR 验证本身都已不是空白**。TextDiffuser、GlyphControl、AnyText/AnyText2 已系统研究字形、布局、多语言生成与 OCR 评价；TextCtrl、FLUX-Text 研究场景文字编辑与风格保持；尤其 CVPR 2025 Highlight **Type-R** 已实现“发现错字—擦除错误文字—补全缺失文本框—修复文字—OCR 验证”的自动后处理。因此，如果本课题仅把 similarity 换成更多指标，或仅增加 OCR 框选与 Qwen-Image-Edit 局部修字，较难形成独立论文贡献。

本文建议将论文主线收紧为：

> **面向多文字、多属性约束的、不确定性感知且成本可控的生成—编辑 Agent：用结构化 OCR critic 建立目标文字与检测实例的对应关系，联合判断内容、布局、顺序、属性和可读性；再根据错误类型、OCR 不确定性、历史动作收益与剩余预算，在重新生成、重写提示词、局部编辑、复核或停止之间自适应路由，同时约束非目标区域不被破坏。**

最小可发表单元不是一个新 OCR 模型，而应当是：一套明确的数据契约与诊断体系、一个可解释的动作策略、一个覆盖“修对文字且保住原图”的修复 benchmark，以及对 Agent 相对裸模型和固定流水线的完整消融。

---

## 1. 项目背景与边界

### 1.1 已知实际条件

- 实习剩余约三个月，目标尽量形成学术论文，而不是只做功能展示。
- 公司内网不能调用外部 API，但可以离线部署开源模型。
- 当前已经部署 Z-Image-Turbo，正在部署 Qwen-Image-Edit。
- 计算资源为 8 张华为昇腾 NPU，每张约 32 GB；必要时可增加资源。
- 第一阶段坚持 training-free：不修改 Z-Image-Turbo 或 Qwen-Image-Edit 权重。
- 任务限定为中英海报与自然场景文字，不扩展到文档 OCR、表格、公式或通用多语言文档理解。

### 1.2 本文对“已有工作”的口径

本文审计的是当前文件夹中的**本地参考实现**。用户已说明公司内部代码是在此基础上修改和适配，逻辑与思想一致，但本地文件不能代表公司环境中每个接口、版本和运行结果都完全相同。因此：

- “已经实现”仅指本地参考仓库中可见的代码和文档；
- “公司已部署”仅采用用户明确提供的信息；
- “建议实现”均作为未来工作，不写成已经完成；
- 本地 mock 运行只证明工作流和日志链路能跑，不证明真实 OCR 或真实生成质量。

---

## 2. 本地已有工作的技术审计

### 2.1 整体框架

本地仓库保留原始 GEMS 作为 baseline，并增加 LangGraph 旁路。当前图结构为：

```text
START
  → skill_router
  → planner
  → decomposer
  → generator (Z-Image-Turbo service)
  → verifier (MLLM)
  → ocr_verifier
  → memory_writer
      ├─ success / max_iter / error → finalizer → END
      └─ retry → refiner → generator
```

文字任务由 `agent/skills/text_rendering/SKILL.md` 触发。该 skill 已包含位置描述、行级拆分、载体、字体/材质、显式文字信号和 exact text preservation 等 prompt 规则。OCR 默认关闭，可使用本地 PaddleOCR 或内网 HTTP OCR sidecar。

### 2.2 当前 OCR 输入、输出与判定

`langgraph_harness/ocr.py` 定义了三类核心对象：

- `OCRLine`：`text`、`confidence`、可选 `bbox`；
- `OCRResult`：多行结果，以及拼接文字、平均置信度和规范化文字；
- `OCRScore`：目标文字、识别文字、exact/substring、编辑距离、similarity、confidence、pass/fail、失败原因和建议。

当前评分可概括为：

```text
normalized_target     = 去大小写、空白和常见标点
normalized_recognized = 将全部 OCR line 以空格拼接后再规范化
distance              = Levenshtein distance
similarity            = 1 - distance / max(length(target), length(recognized))
text_pass              = exact OR substring OR similarity ≥ threshold
final_pass             = text_pass AND average_confidence ≥ threshold
```

`ocr_verifier` 只在 OCR 开启、`text_rendering` skill 被触发且存在 OCR client 时运行。它会从原始 prompt 提取文字；若没有，再从当前 prompt 提取。尽管提取函数可以返回多个目标，实际 verifier 只使用 `targets[0]`。OCR 失败后统一追加 `text_render_error`，修改建议分为“无文字”“文字不匹配”“低置信度”三种。

### 2.3 已有优点

- 保持 GEMS baseline，不破坏原入口，适合做裸模型/Agent 对照。
- OCR client 与图逻辑解耦，可切换本地 PaddleOCR、内网 HTTP 服务或 mock。
- 已记录每轮图片、prompt、检查、失败标签、OCR score、延迟和 final report。
- OCR 的 pass/fail 可以覆盖 MLLM 对精确文字的主观判断，符合“客观 verifier 独立于 planner”的原则。
- 有确定性单元测试覆盖目标提取、规范化、编辑距离和基本 pass/fail。

### 2.4 关键局限

| 层面 | 当前行为 | 对研究和产品的影响 |
|---|---|---|
| 目标解析 | 可以提取多段文字，但只验证首段 | 多标题、多行海报无法完整评价 |
| OCR 表示 | 保存 bbox，但评分不使用 | 无法定位错误或生成 edit mask |
| 实例匹配 | 全部检测文字拼接后和单目标比较 | 阅读顺序、冗余文字和一对多关系被混淆 |
| 错误类型 | 仅无文字、mismatch、低置信度 | 无法区分漏字、错字、重复、错位、错序、样式错误 |
| 置信度 | 对所有正置信度 line 求简单平均 | 可能被无关文字或 OCR 校准偏差误导 |
| 布局/属性 | 不评价位置、行关系、字体、颜色、载体 | 与 text-rendering skill 的规划目标不闭环 |
| 修复动作 | 只有 prompt refiner + 整图重生成 | 成本高，且容易破坏已正确的构图与对象 |
| 记忆 | 每轮 MLLM 摘要，仅当前 run 使用 | 还不是跨任务、证据门控的 procedural memory |
| 最佳图片 | 主要按通过检查数量选择 | 不敏感于文字错误严重度和非目标区域保持 |
| 实验依据 | 可见产物是简单 mock 非文字任务 | 尚无真实 OCR、真实生成和 Agent 增益证据 |

另一个工程风险是 PaddleOCR 版本接口漂移。当前 client 使用旧式 `ocr(image_path, cls=True)` 解析格式，而 PaddleOCR 3.x 官方示例主要使用 `predict()` 和新的结果结构。公司环境部署时需固定版本或做 adapter contract test，不能把 OCR API 兼容问题误认为算法问题。[PaddleOCR 官方使用文档](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html)

---

## 3. 相关工作与文献矩阵

### 3.1 视觉文字生成与编辑

| 工作 | 主要任务与方法 | 是否训练 | OCR/文字控制用途 | 编辑 | 常用评价 | 对本课题的启示与重叠风险 |
|---|---|---:|---|---:|---|---|
| [TextDiffuser](https://arxiv.org/abs/2305.10855) | Transformer 先规划关键词布局，扩散模型再按布局生成；发布 MARIO-10M/MARIO-Eval | 是 | OCR 标注、检测与字符级分割作为数据/监督 | 支持文字 inpainting | OCR、CLIP、人评 | “布局规划 + 文字生成”已有成熟先例；仅增加 layout planner 不新 |
| [GlyphControl](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8951bbdcf234132bcce680825e7cb354-Abstract-Conference.html) | 以 glyph image 控制内容、位置和大小；LAION-Glyph | 是 | glyph 条件与 OCR 评价 | 非主线 | OCR、CLIP、FID | 位置/大小显式控制已有工作；可作训练型上界或公开基线 |
| [TextDiffuser-2](https://arxiv.org/abs/2311.16465) | 微调 LLM 做行级布局规划，并在扩散模型中编码行级位置与文字 | 是 | 文字与位置联合编码 | 可交互改布局 | OCR、布局、质量、人评 | LLM planner 本身不是 Agent 创新，需强调闭环诊断和动作选择 |
| [AnyText](https://proceedings.iclr.cc/paper_files/paper/2024/hash/fb8e5f198c7a5dcd48860354e38c0edc-Abstract-Conference.html) | 辅助 latent module + OCR 字形 embedding；AnyWord-3M；多语言生成/编辑 | 是 | OCR encoder、text perceptual loss、benchmark evaluator | 是 | Sentence Accuracy、NED、FID | 多语言文字生成和 OCR 指标已成熟；适合中英文强基线 |
| [AnyText2](https://arxiv.org/abs/2411.15245) | WriteNet+AttnX，分行控制字体与颜色，提高速度和文字准确性 | 是 | 字形与文字属性条件 | 是 | 中英文字准确性、图像质量 | 字体/颜色控制本身不新；本项目可评价“属性约束是否满足” |
| [TextCtrl](https://papers.nips.cc/paper_files/paper/2024/hash/fa31574791443e8e7f38045b98584aa9-Abstract-Conference.html) | 结构—风格引导和 glyph-adaptive mutual self-attention；ScenePair | 是 | glyph 结构与识别评价 | 是 | 文字准确、风格/背景保持 | 强调文字编辑必须同时衡量内容正确与风格保持 |
| [Type-R](https://openaccess.thecvf.com/content/CVPR2025/papers/Shimoda_Type-R_Automatically_Retouching_Typos_for_Text-to-Image_Generation_CVPR_2025_paper.pdf) | 生成后检测排版错误，擦除错字，为缺词补框，再用文字编辑模型修复并 OCR 验证 | 组件训练/自动后处理 | OCR 定位、错误发现和验证 | 是 | OCR 准确与图像质量平衡 | 与“检测 + mask + 局部修字”高度重叠，是最重要的新颖性边界 |
| [FLUX-Text](https://arxiv.org/abs/2505.03329) | 基于 FLUX-Fill 的多语言场景文字编辑，加入轻量 glyph/text embedding | 是，约 100K 数据 | glyph 条件与文本准确评价 | 是 | 文字 fidelity、图像质量 | 表明强编辑模型仍需要字形条件；可作为训练型编辑上界 |
| [Qwen-Image](https://arxiv.org/abs/2508.02324) / [Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit) | 20B 基础模型，支持中英复杂文字生成和精确文字编辑；官方模型卡声称可保留字体、大小与风格 | 是，直接使用权重 | 内生文字渲染能力 | 是 | 官方 benchmark + 人评 | 是本项目最合适的通用编辑工具，但“调用它修字”不是创新 |
| [VTP survey/VTPBench/VTPScore](https://arxiv.org/abs/2504.21682) | 统一梳理 visual text processing，并以 VTPBench/VTPScore 评价多类任务 | — | 统一多维评价 | 覆盖 | VTPScore 等 | 可补充 OCR 字符指标看不到的视觉质量、风格和可读性 |

### 3.2 图像 Agent 与迭代反馈

| 工作 | 关键思想 | 与本课题关系 |
|---|---|---|
| [GenArtist, NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/hash/e7c786024ca718f2487712bfe9f51030-Abstract-Conference.html) | MLLM 统一选择生成/编辑工具，以树结构分解任务，并逐步验证和自纠 | 证明“多工具 + 逐步验证”已有 agentic 先例；本课题必须在文字专用 critic、动作依据和严格实验上差异化 |
| [Iterative Refinement Improves Compositional Image Generation](https://iterative-img-gen.github.io/) | generator、VLM critic、editor、verifier 的测试时闭环，对比 best-of-N | 与推荐闭环结构高度相关；应加入等预算 best-of-N 作为对照，证明编辑而非更多采样带来提升 |
| [ImAgent](https://arxiv.org/abs/2511.11483) | training-free policy controller，在多种生成动作间进行测试时扩展 | 支持成本/预算感知动作路由方向，也提高了仅“动态路由”的新颖性门槛 |
| [OCR-Agent](https://arxiv.org/abs/2602.21053) | capability reflection + memory reflection，改善 OCR/VLM 多轮自纠 | 其目标是 OCR 理解而非生成修复，但说明“反思 + 记忆”不能只作为口号，需量化避免重复失败的能力 |

### 3.3 OCR 后端与评价工具

- PP-OCRv5 适合作为首个轻量 OCR 后端。官方资料覆盖中、英及多种语言，并提供检测 polygon、识别内容和置信度；第一阶段可使用 CPU 或独立服务，避免占用生成 NPU。[PP-OCRv5 官方文档](https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.html)
- PaddleOCR 当前已继续演进，后续版本和 API 可能变化，因此论文应报告固定版本、检测模型、识别模型、输入缩放和阈值，而不是只写“使用 PaddleOCR”。
- OCR verifier 不是无误差真值。艺术字、弯曲文字、低对比度和复杂背景中，生成文字可能被人读对但 OCR 读错；反之，OCR 可能因语言先验猜对。论文需要 OCR 不确定性处理和人工抽检，而不能把单 OCR 输出当绝对 ground truth。
- OCRBench v2 是面向 MLLM 的双语文字定位与推理 benchmark，并不是本项目的生成质量主 benchmark，但其定位、场景覆盖和人工核验方法可借鉴。[OCRBench v2](https://papers.nips.cc/paper_files/paper/2025/hash/8c2e6bb15be1894b8fb4e0f9bcad1739-Abstract-Datasets_and_Benchmarks_Track.html)

---

## 4. 哪些内容不能单独宣称为创新

以下内容都可以实现，但不能单独作为论文主贡献：

1. **把 similarity 改成 CER/WER/NED/F1。** AnyText 等已经使用 Sentence Accuracy、NED 和 OCR 指标；这属于必要评价建设。
2. **保存 OCR bbox 并画框。** 文字检测天然输出 polygon；这是工程完善。
3. **根据 bbox 做 mask，再调用通用编辑模型修字。** Type-R 已系统完成相似流程。
4. **用 LLM 改写文字 prompt。** TextDiffuser-2 已用 LLM 做文字和布局规划；大量图像 Agent 也已做 prompt refinement。
5. **增加长期记忆/RAG。** 如果没有明确记忆结构、晋升规则和跨任务收益，只是通用 Agent 功能拼接。
6. **用 OCR 作为 reward/verifier。** OCR 被广泛用于文字生成训练和评价。
7. **部署 PP-OCR 到 Ascend。** 这是部署优化而不是算法创新，除非论文主题是系统吞吐或端云协同。

可将上述内容作为系统组件，但论文必须回答新的研究问题。

---

## 5. 推荐论文主线：USAR-Agent

可暂用名称：**USAR-Agent: Uncertainty-aware Structured OCR Critic and Adaptive Repair for Visual Text Generation**，中文为“面向视觉文字生成的不确定性感知结构化评价与自适应修复 Agent”。

### 5.1 研究问题

- **RQ1：** 相比全图字符串相似度，实例级、多约束 OCR critic 能否更准确地诊断“哪里错、错什么、严重程度如何”？
- **RQ2：** 相比固定的“失败就重生成”或“失败就局部编辑”，基于错误类型、不确定性、预期收益和预算的动作路由能否提高成功率/成本比？
- **RQ3：** 局部修复能否在提高文字正确性的同时，显著降低非目标区域漂移？
- **RQ4：** 证据门控的跨任务经验是否能减少无效重复动作，并对未见文字长度、布局和场景泛化？

### 5.2 贡献点草案

1. **结构化视觉文字约束与诊断。** 将自然语言请求解析为多个文字实例及内容、位置、顺序、行关系、颜色、字体类别和载体约束；通过目标—检测匹配输出可执行错误图，而不是单一分数。
2. **不确定性感知、成本可控的修复策略。** 联合 OCR 置信度、多个 OCR/增强视图的一致性、错误严重度、动作历史收益和剩余预算，在重生成、prompt refine、局部编辑、复核与停止之间选择。
3. **文字修复与内容保持的双目标闭环。** 每次编辑同时衡量目标文字收益和 mask 外漂移；若文字变好但整体图像被破坏，不视为成功。
4. **面向 Agent 修复的测试协议。** 构建包含初始错误图、结构化文字约束、错误标注和修复区域的中英测试集，报告等预算下成功率、成本、非目标保持与错误转移。

前三点是最小论文核心；长期记忆是增强项，不应在前五周拖慢主闭环。

### 5.3 与 Type-R 的明确差异

Type-R 已覆盖自动 typo retouch，因此本文不能把“自动修错字”写成首创。建议差异落在：

- Type-R 更像专用自动后处理 pipeline；本项目研究多种可替代动作的**决策问题**；
- 从 typo/content 扩展到多实例的内容、漏项、冗余、阅读顺序、布局和属性约束；
- 显式建模 OCR 不确定性，允许 `recheck`，避免 verifier 错误驱动破坏性编辑；
- 在固定调用预算下比较 regenerate、best-of-N、prompt refine、edit 和混合策略；
- 将 mask 外保持作为动作接受条件，而不是只追求 OCR 正确；
- 用跨任务历史估计“某类错误 × 某类动作”的收益，但 verifier 仍保持独立。

是否足以投稿最终仍取决于实验结果和投稿窗口；开始写论文前必须精读 Type-R 正文、附录和代码，逐项做 related-work claim audit。

---

## 6. 系统设计

### 6.1 推荐数据契约

```python
TextConstraint = {
    "id": str,
    "text": str,
    "language": "zh" | "en" | "mixed",
    "line_index": int | None,
    "reading_order": int | None,
    "expected_region": [x1, y1, x2, y2] | None,   # 归一化坐标
    "relation": {"type": str, "target_id": str} | None,
    "font_family_class": str | None,
    "font_weight": str | None,
    "color": str | None,
    "carrier": str | None,
    "required": bool,
}

DetectedTextInstance = {
    "id": str,
    "text": str,
    "normalized_text": str,
    "polygon": [[x, y], ...],
    "confidence": float,
    "reading_order": int,
    "crop_path": str | None,
    "attributes": {"color": str | None, "font_class": str | None},
    "recognizer_votes": list,
}

ConstraintMatch = {
    "constraint_id": str,
    "detected_id": str | None,
    "char_alignment": list,
    "content_score": float,
    "location_score": float | None,
    "order_score": float | None,
    "attribute_score": float | None,
    "uncertainty": float,
}

OCRDiagnosis = {
    "matches": list[ConstraintMatch],
    "missing": list[str],
    "extra": list[str],
    "error_types": list[str],
    "severity": float,
    "critic_confidence": float,
    "repair_regions": list,
    "recommended_actions": list[str],
}

ActionDecision = {
    "action": "accept" | "recheck" | "regenerate" | "refine_prompt" | "local_edit" | "stop",
    "target_constraints": list[str],
    "expected_gain": float,
    "estimated_cost": float,
    "reason": str,
}
```

### 6.2 目标—检测实例匹配

不要再将所有 OCR 文字简单拼接。推荐构造二分图，边代价由以下部分组成：

```text
cost(i, j) =
    w_text   × normalized_edit_distance(target_i, detection_j)
  + w_pos    × region_distance(expected_i, detected_j)
  + w_order  × reading_order_violation(i, j)
  + w_attr   × attribute_distance(i, j)
  + w_conf   × uncertainty_penalty(j)
```

使用 Hungarian matching 或最小费用匹配建立一对一对应；对可能跨多个 OCR box 的目标，可在匹配前生成相邻 box 合并候选。未匹配目标为 missing，未匹配检测为 extra。权重在 validation split 上确定并固定，不能按测试集调参。

### 6.3 错误分类

建议第一版只保留可可靠判定的七类：

- `missing_text`：目标实例未找到；
- `content_substitution`：替换或相似形错字；
- `content_insertion_deletion`：多字、少字；
- `extra_text`：出现未要求文字；
- `low_legibility`：多增强视图/识别器不一致或置信度低；
- `layout_order_error`：位置、行关系或阅读顺序错误；
- `attribute_error`：颜色或粗细等可稳定检测的属性不满足。

字体精确识别很难可靠完成，初期只做粗类别或交给 VLM 辅助，并在论文中报告人工一致性。不要用不可靠的自动标签堆出“多维 critic”。

### 6.4 OCR 不确定性

至少实现无需训练的三种信号：

1. 原图、放大图、对比度增强图的 OCR 一致性；
2. 中英不同识别配置或两个轻量 recognizer 的结果一致性；
3. 检测置信度、识别置信度与字符对齐稳定性。

当 critic 不确定性高时先 `recheck`，不立即编辑。人工标注一小批样本，画 reliability diagram 或计算 ECE/Brier score，验证 critic confidence 是否可用。若单 OCR 与 ensemble 差异很小，可在最终系统中退回单 OCR 以节省成本。

### 6.5 动作路由

首版使用可解释规则，不训练 policy：

| 诊断 | 默认动作 | 原因 |
|---|---|---|
| OCR 高不确定性 | `recheck` | 避免 verifier 错误导致破坏性修复 |
| 多数文字缺失、整体布局严重错误 | `regenerate` 或 `refine_prompt` | 局部编辑难以重建全局排版 |
| 单个实例少量错字、bbox 可靠 | `local_edit` | 修复范围小，背景保持更好 |
| 文字内容正确但位置错误 | `local_edit`；失败后 regenerate | 先尝试低成本局部修复 |
| 连续两次同动作无提升 | 切换动作 | 防止循环 |
| 预测收益低于阈值或预算耗尽 | `stop` | 控制测试时成本 |

决策目标可写成：

```text
utility(action) = expected_text_gain
                - λ1 × expected_outside_mask_drift
                - λ2 × latency
                - λ3 × NPU/model_call_cost
```

前期用离线统计估计 expected gain；第 9–10 周再比较 LLM router 和经验统计 router。为保证可复现，LLM router 必须输出结构化理由，且最终论文的主结果不能只依赖不可审计的自由文本决策。

### 6.6 局部编辑与 mask

- 对已有错误实例：将 OCR polygon 转为 mask，按文字高度自适应膨胀，保留周围少量纹理上下文。
- 对 missing text：使用目标期望区域；若 prompt 没有区域，先由 layout planner 给候选框，再由 VLM 检查是否遮挡主体。
- Qwen-Image-Edit 的指令同时包含：精确目标文字、只修改 mask 区域、保留字体/大小/颜色/透视/材质、保持其他像素和对象不变。
- 若部署接口不原生接受 mask，可用带框/裁剪提示、局部 crop 编辑再融合，或将 mask 作为额外参考图；必须在实验中记录具体方式，不能笼统称为 inpainting。

### 6.7 双目标接受门

候选编辑只有同时满足以下条件才替换当前最佳图：

```text
text_score(new) - text_score(old) ≥ Δ_text_min
outside_mask_drift(new, old)      ≤ drift_max
global_semantic_score(new)        ≥ semantic_floor
```

mask 外保持可用 LPIPS、SSIM、DINO/CLIP feature similarity 组合；若环境依赖受限，至少使用 SSIM + 感知特征。最终阈值在 validation set 固定。

### 6.8 LangGraph 演进

```text
START
  → constraint_parser
  → skill_router / planner
  → generator
  → visual_verifier
  → structured_ocr_critic
  → uncertainty_gate
      ├─ recheck → ocr_ensemble → structured_ocr_critic
      └─ reliable → action_router
          ├─ accept → experience_writer → finalizer
          ├─ refine_prompt → refiner → generator
          ├─ regenerate → generator
          ├─ local_edit → mask_builder → qwen_image_edit
          │               → preservation_verifier → structured_ocr_critic
          └─ stop → finalizer
```

`experience_retriever` 只进入 planner/action_router；不得进入 OCR critic 或 preservation verifier。

---

## 7. 经验记忆设计

### 7.1 两层存储

**Episodic memory** 保存不可变原始轨迹：任务约束、初图、OCR 实例、诊断、动作、编辑图、指标变化、延迟和资源成本。它用于审计和离线统计，不直接注入 prompt。

**Procedural experience** 保存抽象策略，例如：

```text
context: zh, 8–14 chars, two-line poster, one missing line
action: regenerate with explicit line boxes
evidence: 7/10 tasks improved, mean NED +0.21, mean cost 1.3 calls
status: candidate / active / retired
```

### 7.2 晋升与淘汰

- 至少覆盖 3 个不同 prompt、5 次独立尝试后才允许从 candidate 晋升；具体门槛可根据实际数据量调整，但必须预注册。
- 同时保存成功和失败证据，使用 Wilson interval 或 bootstrap confidence interval，不只记录平均提升。
- 经验只能改变动作先验或 prompt 策略，不能改变 verifier 的通过标准。
- 在新数据上持续无收益则降级或淘汰。
- 第一版可以 SQLite + 结构化过滤；数据规模不足时无需引入 embedding/RAG。只有当结构化检索召回不足时再加 BGE-M3。

### 7.3 记忆是否值得成为贡献

只有当实验显示它在未见任务上显著减少调用数或失败循环，且效果超过简单的错误类型统计表时，才写成论文贡献；否则作为系统功能放入附录。

---

## 8. Benchmark 与数据设计

### 8.1 两类测试集

1. **Generation set**：从文字 prompt 直接生成，评价裸模型和不同 Agent 的最终成功率。
2. **Repair set**：固定一批带已知文字错误的初始图，使不同修复策略从完全相同的状态开始，降低生成随机性对路由实验的干扰。

Repair set 是论文差异化的关键，因为公开文字 benchmark 多评价单次生成，而 Agent 需要评价“识别错误—选择动作—修复—保持内容”的轨迹。

### 8.2 建议规模

- 开发集：40–60 prompts，每个 prompt 2–4 seeds，用于调试，不进入主结果。
- 正式 generation test：至少 200 prompts，中文/英文各半，每个方法固定相同 seeds。
- repair test：至少 150 个初始错误样本，按七类错误分层；每个样本有人审目标文字、错误区域和非编辑区域。
- 人工评价子集：60–100 个样本，每个至少 3 名标注者，盲评且随机方法顺序。

如果三个月内标注资源不足，优先保证 repair set 的质量和分层，缩小规模也不要使用大量弱标签冒充 ground truth。

### 8.3 Prompt 分层

| 维度 | 建议分桶 |
|---|---|
| 语言 | 中文、英文、中英混排 |
| 长度 | 1–4、5–10、11–20、20+ 字符 |
| 实例数 | 1、2、3+ |
| 行数 | 单行、双行、多行 |
| 场景 | 平面海报、招牌/包装、衣物/曲面、霓虹/反光/低对比背景 |
| 属性 | 无属性、颜色、粗细/类别、材质/载体 |
| 布局 | 无指定、绝对区域、相对关系、阅读顺序 |

### 8.4 公开数据与基线

- MARIO-Eval：英文文字图像与布局评价参考；来源于 TextDiffuser。
- AnyText-benchmark：中文 Wukong-word 与英文 LAION-word，包含官方生成和评价脚本。[官方仓库](https://github.com/tyxsspa/AnyText)
- LAION-Glyph：英文 glyph-controlled generation 参考。
- ScenePair：真实场景文字编辑与风格保持参考，来源于 TextCtrl。
- LenCom-Eval：长且复杂文字，适合压力测试；该工作报告 training-free glyph enhancement 对 TextDiffuser 的 OCR word F1 有明显提升。[论文](https://arxiv.org/abs/2403.16422)

公开 benchmark 的 prompt/输入格式与 Z-Image-Turbo/Qwen-Image-Edit 不完全一致，必须写 conversion protocol，并避免给某一方法额外的 glyph/layout 信息。

---

## 9. 指标体系

### 9.1 文字内容

- **Sentence Accuracy**：所有必需文字实例完全正确的样本比例；作为最直观主指标。
- **CER / Character Accuracy**：中文优先字符级。
- **WER**：英文单词级辅助指标。
- **NED**：`1 - edit_distance / max_length`，需明确 normalization 规则。
- **Character/word precision、recall、F1**：区分漏字与额外文字。
- **Instance Recall/Precision**：目标文字实例是否一一找到，防止全图拼接掩盖错误。

### 9.2 布局和属性

- bbox/polygon IoU 或中心点距离；仅对存在期望区域的样本计算。
- 阅读顺序准确率或 Kendall's tau。
- 颜色差异可在文字 mask 内用 Lab/Delta E；字体只做粗粒度分类或人工评价。
- 多行关系准确率：上下、左右、居中、对齐。

### 9.3 图像保持和整体质量

- mask 外 SSIM、LPIPS、DINO/CLIP feature similarity；
- 全图 CLIP/VQAScore 或公司 VLM 的语义约束检查；
- 人工评价：文字正确、文字自然融合、非目标区域保持、整体审美四个维度；
- 可参考 VTPScore，但必须报告其模型依赖和人工相关性，不把 MLLM judge 当唯一真值。

### 9.4 Agent 效率

- 等预算最终成功率；
- 平均/中位生成调用、编辑调用、OCR 调用和总轮数；
- wall-clock latency、NPU-seconds、峰值显存；
- Area Under Success-vs-Budget Curve；
- 无收益动作率、重复动作率、错误转移率；
- `success / model-call` 或 utility 指标。

所有方法必须在相同最大生成/编辑调用预算下比较。否则 Agent 多调用模型带来的提升无法与简单 best-of-N 区分。

---

## 10. 基线与消融

### 10.1 必须基线

1. Z-Image-Turbo 单次裸生成；
2. Z-Image-Turbo best-of-N + 相同 verifier；
3. 仅 LLM prompt rewrite 后重生成；
4. 当前 similarity OCR 闭环；
5. 固定规则：OCR 失败全部重生成；
6. 固定规则：OCR 失败全部 Qwen-Image-Edit；
7. 结构化 critic + 固定动作；
8. 完整结构化 critic + uncertainty + adaptive router；
9. 完整系统 + procedural experience；
10. 可部署时加入 AnyText2、Type-R 或其可复现实验设置作为公开强基线。

Type-R 若无法在 Ascend 直接部署，可在独立 GPU/公开结果上做协议对齐，或明确列为 related work 而不是假装做了公平复现。任何不可公平复现的数字不应放进主表横向比较。

### 10.2 关键消融

- merged-text similarity vs instance matching；
- 去掉 layout/order/attribute 分量；
- 单 OCR vs 多增强 OCR vs 多识别器；
- 无 uncertainty gate；
- regenerate-only vs edit-only vs adaptive routing；
- 无 outside-mask preservation gate；
- 无成本项；
- 无记忆、当前 run 记忆、跨任务经验；
- 规则 router vs LLM router vs 经验统计 router。

### 10.3 统计要求

- 同 prompt、同 seed 配对比较；
- 对成功率使用 bootstrap confidence interval 或 McNemar test；
- 对连续指标报告均值、中位数、95% CI 和 effect size；
- 多次随机运行的 router 报告方差；
- 在实验前固定 primary metric、最大预算和接受阈值，降低选择性报告风险。

---

## 11. 测试与失败场景

实现阶段至少覆盖以下自动测试：

- 无目标文字时 OCR 节点应无副作用退出；
- 多个引号目标全部保留，不能只验证首个；
- 中英文标点和大小写 normalization 可配置；
- OCR 无结果、低置信、重复 box、box 顺序错乱；
- 一个目标跨多个 box、多个目标被合并为一个 box；
- missing、extra、substitution、insertion/deletion 的匹配正确；
- polygon 膨胀不越界，mask 与图片尺寸一致；
- 高不确定性进入 recheck 而非编辑；
- 连续两次同动作无收益会切换或停止；
- 达到调用、延迟或轮数预算必然终止；
- 编辑后文字提高但 mask 外漂移超阈值时拒绝候选；
- OCR client 新旧 PaddleOCR 响应格式 contract test；
- 日志中不写 API key，且每轮可从 artifact 完整回放。

真实系统还要做一组人工 adversarial case：艺术字、镜像字、曲面字、竖排、文字遮挡、相似中文字符、`O/0`、`I/l/1`、低对比、OCR 猜对但视觉不可读、文字正确但位置错误。

---

## 12. 三个月执行路线

### 第 1–2 周：Evaluator 与新颖性核查

- 精读并复现 AnyText 评价脚本、Type-R 论文/代码关键流程、TextCtrl 的保持指标。
- 固定 PaddleOCR 版本和服务契约，跑通真实中英文 OCR。
- 实现 `TextConstraint`、`DetectedTextInstance`、instance matching 和七类诊断。
- 创建 40–60 prompt 开发集与第一批人工错误样本。
- 交付：structured critic Demo、单元测试、baseline 指标表。

**Go/No-Go 1：** 结构化 critic 对人工错误类型的 macro-F1 至少明显优于当前三类错误；若 OCR 自身不可靠，先解决 critic 校准，暂不接编辑。

### 第 3–5 周：局部编辑闭环

- 完成 Qwen-Image-Edit 内网服务接口。
- 实现 polygon→mask、missing-text 区域规划、编辑 prompt 模板。
- 实现文字增益 + mask 外保持双目标接受门。
- 比较 regenerate-only、edit-only、best-of-N。
- 交付：从错误图到局部修复的可视化 Demo 和 repair set v1。

**Go/No-Go 2：** 局部编辑应在相同调用预算下优于重生成，并显著降低非目标漂移；否则论文主线转向 evaluator/benchmark 或更换编辑策略。

### 第 6–8 周：路由与完整实验

- 实现 uncertainty gate、预算模型和规则 router。
- 完成 200 prompt generation set 与至少 150 样本 repair set。
- 运行主要基线、消融、失败类型和预算曲线。
- 冻结主实验协议，停止随结果修改阈值。
- 交付：主结果表、预算曲线、定性案例和错误分析。

### 第 9–10 周：经验与泛化

- 从已完成轨迹建立 action outcome table。
- 实现证据门控 procedural experience。
- 与简单频率表、LLM router 对比，测试未见长度/布局/场景。
- 若无显著收益，将记忆降为附录，不影响主论文。

### 第 11–12 周：论文与汇报

- 完成人工盲评、统计检验和复现实验。
- 精简贡献点，逐条检查是否被 Type-R、GenArtist、ImAgent 等覆盖。
- 完成论文初稿、系统图、失败案例、Demo 视频与 mentor 汇报。
- 整理代码配置、模型版本、数据 split、随机种子和复现说明。

---

## 13. 算力与部署建议

### 13.1 推荐分配

- Z-Image-Turbo：1–2 卡加载独立副本，批量生成时扩大并发；
- Qwen-Image-Edit：根据实际显存和 Ascend 适配单独占 2–4 卡；20B 模型是否可单卡需以部署精度、量化和框架为准，本文不预设；
- OCR：初期 CPU/独立轻量容器；仅在 profiling 证明其为瓶颈时迁移 NPU；
- 其余 NPU：公开基线、批量实验或并行 seeds。

单次请求不必强行做跨卡模型并行；优先通过多模型副本提高 benchmark 吞吐。所有服务分环境部署，避免 Z-Image、Qwen-Image-Edit、PaddleOCR 的 PyTorch/Paddle/CANN 依赖互相污染。

### 13.2 实验日志

每轮至少保存：

```text
original prompt / parsed constraints
actual generation or edit prompt
input and output image hashes
OCR raw output and normalized instances
diagnosis and uncertainty
chosen action and rejected alternatives
mask / crop / expected region
text and preservation metrics before/after
model version, seed, latency, device and cost
termination reason
```

这些字段既用于复现，也用于后续 experience learning。

---

## 14. 风险、备选路线与优先级

| 方向 | 创新潜力 | 三个月可行性 | 主要风险 | 建议 |
|---|---:|---:|---|---|
| 多实例结构化 critic + repair benchmark | 高 | 高 | 需要高质量人工标注 | **必须做，最稳主线** |
| 不确定性与成本感知动作路由 | 中高 | 中高 | 容易被认为规则工程；需强等预算实验 | **核心增强** |
| Qwen 局部修字 + 保持门 | 中 | 高 | 与 Type-R 高度重叠 | 作为系统组件，不单独 claim |
| 跨任务 procedural experience | 中 | 中 | 三个月轨迹量可能不足 | 第 9 周后再决定是否主贡献 |
| 训练 OCR reward/router | 中高 | 低到中 | 数据和时间不足，偏离零训练路线 | 只作为后续工作 |
| 新 glyph-control 生成模型 | 高 | 低 | 训练/适配成本大，已有强工作 | 本期不做 |
| OCR Ascend 加速 | 低（算法论文） | 中 | 工程工作吞噬时间 | profiling 后再决定 |

### 备选论文路线

若 Qwen-Image-Edit 无法稳定局部修复，转向：

1. **Evaluator/benchmark 论文：** 聚焦多实例、多约束、修复轨迹和 OCR 不确定性的评测集；
2. **决策分析论文：** 在多个生成/编辑模型上研究不同错误适合 regenerate 还是 edit，并给出等预算经验结论；
3. **系统论文：** 强调 Ascend 内网、多服务隔离、可追踪闭环和吞吐，但投稿目标需转向系统/应用类 venue。

---

## 15. 最小 Demo 定义

输入：

> 生成一张未来科技会议海报，顶部写“智启未来”，底部写“AI FOR INDUSTRY”，两行居中，中文为白色粗体，英文为蓝色无衬线体。

Demo 应展示：

1. Z-Image-Turbo 初图；
2. prompt 被解析成两个 `TextConstraint`；
3. OCR polygon、实例匹配和字符级差异可视化；
4. critic 指出具体错误，如英文第 5 个字符替换、中文位置正确；
5. router 选择只编辑英文区域，并显示原因、预计成本和剩余预算；
6. Qwen-Image-Edit 输出修复图；
7. OCR 内容指标提高，同时 mask 外 SSIM/LPIPS 通过保持门；
8. 最终报告列出动作轨迹、耗时和相对裸模型的收益。

这个 Demo 比“生成—OCR 分数—重新生成”更能体现 Agent 的必要性，也与论文实验的数据结构一致。

---

## 16. 可与 mentor 讨论的题目与贡献表述

### 推荐题目

- **USAR-Agent: Uncertainty-aware Structured OCR Critic and Adaptive Repair for Visual Text Generation**
- **Beyond OCR Similarity: Cost-aware Agentic Repair for Multilingual Text-in-Image Generation**
- **Generate, Diagnose, or Edit? Budgeted Agentic Refinement for Visual Text Rendering**

### 三句话贡献版本

1. 我们提出实例级结构化 OCR critic，将多行中英文视觉文字请求表示为可匹配的内容、布局、顺序和属性约束，并输出带不确定性的可执行诊断。
2. 我们提出预算感知的生成—编辑路由，在重生成、prompt refine、局部编辑和复核之间自适应选择，并使用文字收益与非编辑区保持双目标接受候选。
3. 我们建立面向修复轨迹的中英测试协议，在等模型调用预算下系统比较裸模型、best-of-N、固定闭环和自适应 Agent，并分析各模块的真实收益。

需要避免的表述：

- “首次使用 OCR 改进图像文字”；
- “首次自动修复生成图中的错字”；
- “首次使用 Agent 做图像生成/编辑”；
- “提出新的 OCR 模型”（本项目并没有）；
- 在没有真实实验前写“显著提升”。

---

## 17. 近期行动清单

1. 在公司代码中核对本地审计结论，尤其是是否仍只使用首个 target、是否保存 polygon、Qwen-Image-Edit 接口是否支持 mask。
2. 固定 PaddleOCR 版本并制作 50 张中英文人工核验集，先测 OCR 本身，不要直接测 Agent。
3. 精读 Type-R 正文、附录和代码，形成逐模块差异表；这是立项前的最高优先级。
4. 用当前输出 schema 向后兼容地设计三个新对象：`TextConstraint`、`DetectedTextInstance`、`OCRDiagnosis`。
5. 先完成 instance matching 和错误分类，再接 Qwen-Image-Edit；否则编辑动作没有可靠依据。
6. 从第一天就实现等预算 best-of-N baseline，避免最后无法证明 Agent 的调用效率价值。

---

## 参考文献与资源

### 论文与官方论文页

1. Chen et al. [TextDiffuser: Diffusion Models as Text Painters](https://arxiv.org/abs/2305.10855), 2023.
2. Yang et al. [GlyphControl: Glyph Conditional Control for Visual Text Generation](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8951bbdcf234132bcce680825e7cb354-Abstract-Conference.html), NeurIPS 2023.
3. Chen et al. [TextDiffuser-2: Unleashing the Power of Language Models for Text Rendering](https://arxiv.org/abs/2311.16465), 2023/2024.
4. Tuo et al. [AnyText: Multilingual Visual Text Generation and Editing](https://proceedings.iclr.cc/paper_files/paper/2024/hash/fb8e5f198c7a5dcd48860354e38c0edc-Abstract-Conference.html), ICLR 2024.
5. Tuo et al. [AnyText2: Visual Text Generation and Editing With Customizable Attributes](https://arxiv.org/abs/2411.15245), 2024.
6. Zeng et al. [TextCtrl: Diffusion-based Scene Text Editing with Prior Guidance Control](https://papers.nips.cc/paper_files/paper/2024/hash/fa31574791443e8e7f38045b98584aa9-Abstract-Conference.html), NeurIPS 2024.
7. Shimoda et al. [Type-R: Automatically Retouching Typos for Text-to-Image Generation](https://openaccess.thecvf.com/content/CVPR2025/papers/Shimoda_Type-R_Automatically_Retouching_Typos_for_Text-to-Image_Generation_CVPR_2025_paper.pdf), CVPR 2025 Highlight.
8. Lan et al. [FLUX-Text: A Simple and Advanced Diffusion Transformer Baseline for Scene Text Editing](https://arxiv.org/abs/2505.03329), 2025.
9. Lakhanpal et al. [Refining Text-to-Image Generation: Towards Accurate Training-Free Glyph-Enhanced Image Generation](https://arxiv.org/abs/2403.16422), 2024.
10. Shu et al. [Visual Text Processing: A Comprehensive Review and Unified Evaluation](https://arxiv.org/abs/2504.21682), 2025.
11. Wang et al. [GenArtist: Multimodal LLM as an Agent for Unified Image Generation and Editing](https://papers.nips.cc/paper_files/paper/2024/hash/e7c786024ca718f2487712bfe9f51030-Abstract-Conference.html), NeurIPS 2024.
12. Wang et al. [ImAgent: A Unified Multimodal Agent Framework for Test-Time Scalable Image Generation](https://arxiv.org/abs/2511.11483), 2025.
13. Wen et al. [OCR-Agent: Agentic OCR with Capability and Memory Reflection](https://arxiv.org/abs/2602.21053), 2026.
14. Wu et al. [Qwen-Image Technical Report](https://arxiv.org/abs/2508.02324), 2025.
15. Wei et al. [OCRBench v2](https://papers.nips.cc/paper_files/paper/2025/hash/8c2e6bb15be1894b8fb4e0f9bcad1739-Abstract-Datasets_and_Benchmarks_Track.html), NeurIPS 2025 Datasets and Benchmarks.

### 官方代码与模型

- [AnyText 官方仓库](https://github.com/tyxsspa/AnyText)
- [AnyText2 官方仓库](https://github.com/tyxsspa/AnyText2)
- [GlyphControl 官方仓库](https://github.com/AIGText/GlyphControl-release)
- [TextCtrl 官方仓库](https://github.com/weichaozeng/TextCtrl)
- [Type-R 官方仓库](https://github.com/CyberAgentAILab/Type-R)
- [FLUX-Text 官方仓库](https://github.com/AMAP-ML/FluxText)
- [Qwen-Image-Edit 官方模型卡](https://huggingface.co/Qwen/Qwen-Image-Edit)
- [Z-Image 官方仓库](https://github.com/Tongyi-MAI/Z-Image)
- [PaddleOCR 官方仓库](https://github.com/PaddlePaddle/PaddleOCR)
- [PP-OCRv5 多语言识别文档](https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.html)

### 本地参考依据

- `GEMS_LANGGRAPH_GUIDE.md`：GEMS × LangGraph 旁路、部署和运行说明；
- `CURRENT_WINDOW_HANDOFF_SUMMARY_0707.md`：项目前序决策和公司环境背景；
- `GEMS_LANGGRAPH_OCR_MERGE_GUIDE_FOR_AGENT.md`：OCR 接入和失败重试说明；
- `langgraph_harness/ocr.py`：当前目标提取、OCR adapter 和 similarity 评分；
- `langgraph_harness/nodes.py`、`graph.py`：OCR verifier 在图中的位置与状态更新；
- `agent/skills/text_rendering/SKILL.md`：文字 prompt 规划规则；
- `tests/test_ocr.py`：当前 OCR 单元测试覆盖；
- `outputs/langgraph_runs/20260707_223745_7ce24cb3/`：现有 mock 运行产物，仅用于确认日志链路。
