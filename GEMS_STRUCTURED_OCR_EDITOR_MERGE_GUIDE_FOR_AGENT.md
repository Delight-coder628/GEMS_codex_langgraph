# GEMS + LangGraph 结构化 OCR Critic 与 Qwen-Image-Edit 合并指南

本文档供公司内部 coding agent 使用，用于把本地参考仓库的一周增量改造合并到公司已适配的 GEMS + LangGraph 代码。公司代码可能已修改模型服务、配置或目录结构，因此应按“接口和行为”合并，不要机械覆盖整个文件。

## 1. 本次目标与非目标

本次完成：

- 将单目标、全图 merged-text similarity OCR 升级为多目标、实例级结构化 Critic；
- 输出 missing、extra、content mismatch、low confidence 和 reading order error；
- 保留旧 `OCRScore`/`ocr_score`，避免已有日志消费者立即失效；
- 增加确定性动作建议；
- 增加可选 Qwen-Image-Edit HTTP client、polygon mask、LangGraph editor 节点和 mock；
- 编辑失败自动回落到已有 prompt refiner，不中断整次运行；
- 默认关闭 OCR/editor，非文字任务行为不变。

本次不完成：

- OCR ensemble、不确定性校准；
- 复杂位置、字体和颜色自然语言解析；
- LPIPS/DINO 等编辑保持评价；
- 长期经验、学习型 router；
- Qwen-Image-Edit 服务端部署代码。

## 2. 新旧流程

原流程：

```text
generator → MLLM verifier → OCR verifier → memory_writer
  ├─ success → finalizer
  └─ failure → refiner → generator
```

新流程：

```text
generator → MLLM verifier → structured OCR critic → memory_writer
  ├─ success → finalizer
  ├─ local_edit → editor → MLLM verifier → structured OCR critic
  └─ other failure → refiner → generator
```

只有以下条件全部满足才进入 editor：

1. `editor.enabled=true`；
2. 当前只有一个明确内容错误；
3. 错误检测实例具有 polygon；
4. 尚未达到 `max_edits_per_run`。

否则沿用原 refiner/re-generate 路径。

## 3. 建议合并顺序

### 第一步：结构化 OCR 数据层

将本地 `langgraph_harness/ocr.py` 中以下类型和函数移植到公司对应 OCR 模块：

- `TextConstraint`
- `DetectedTextInstance`
- `ConstraintMatch`
- `OCRDiagnosis`
- `build_text_constraints`
- `build_detected_instances`
- `diagnose_ocr_result`

保留公司已有 OCR client。如果公司 HTTP OCR 已返回 `polygon`，将其映射到本地 `OCRLine.bbox` 或直接映射到 `DetectedTextInstance.polygon`。

重要语义：

- similarity 只用于目标—检测实例配对；
- 结构化 Critic 的内容通过要求规范化后 exact match；
- 旧 `score_ocr_result` 仍保留原阈值行为，只用于兼容字段；
- 当前匹配使用确定性贪心算法，不需要 SciPy。

### 第二步：状态和日志

向公司 Agent state 增加：

```text
ocr_constraints: list
ocr_instances: list
ocr_diagnosis: dict
recommended_action: str
actual_action: str
edit_attempts: int
editor_latency_ms: int
pending_edit: dict
```

初始化必须提供空值，避免 TypedDict/LangGraph 节点访问缺失。

每轮 OCR artifact 建议采用：

```json
{
  "target_texts": [],
  "constraints": [],
  "instances": [],
  "diagnosis": {},
  "ocr_result": {},
  "legacy_ocr_score": {},
  "ocr_score": {}
}
```

初次 OCR 文件为 `ocr_round_01.json`；同一 generation round 的第一个编辑结果写为 `ocr_round_01_edit_01.json`，避免覆盖编辑前诊断。

### 第三步：替换 OCR verifier 业务逻辑

公司版本可能已经修改 `nodes.py`，不要整体覆盖。只替换以下行为：

1. 从原始 prompt 提取全部目标；
2. 构造 constraints 和 OCR instances；
3. 调用结构化诊断；
4. 为每个目标增加独立 verifier check；
5. 任一必需目标失败或存在 extra/order error 时，整体 OCR 失败；
6. 更新 `suggested_fix`、`recommended_action` 和 `pending_edit`；
7. 继续写 `text_render_error`，保持公司已有 failure tag 兼容。

如果公司 prompt parser 已有更强结构化输出，可直接生成 `TextConstraint`，不需要退回正则解析。

### 第四步：Editor client 与图路由

移植 `langgraph_harness/editor.py` 中：

- `EditorClient`
- `MockEditorClient`
- `build_polygon_mask`
- `build_text_edit_prompt`

向 Node dependencies 增加可空 `editor`，增加 `editor` 节点，并在 `memory_writer` 后的条件路由中加入 `local_edit`。Editor 成功后重新进入 MLLM verifier，而不是直接 OCR；这样可继续检查整图语义。

编辑失败时：

- 不写入全局 `errors`；
- `edit_attempts += 1`，防止对同一失败服务无限循环；
- 记录 `Editor fallback: ...`；
- 设置动作为 `refine_prompt`，回落到原路径。

## 4. 配置与环境变量

YAML 新增：

```yaml
editor:
  enabled: false
  url: "${EDITOR_URL}"
  timeout_seconds: 600
  max_retries: 1
  max_edits_per_run: 1
  mask_padding_ratio: 0.25
```

环境变量：

```dotenv
EDITOR_URL=http://127.0.0.1:8003/edit
```

公司第一次合并必须保持 `enabled: false`，确认 Qwen 服务正常后再打开。

配置校验：

- enabled 时 URL 必填；
- URL 必须为 `http://` 或 `https://`；
- `max_edits_per_run` 建议第一周固定为 1。

## 5. Qwen-Image-Edit HTTP contract

客户端请求：

```http
POST /edit
Content-Type: multipart/form-data
```

字段：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `image` | PNG file | 是 | 当前生成图或上一次编辑图 |
| `mask` | grayscale PNG file | 否 | 白色为编辑区域，黑色为保留区域 |
| `prompt` | string | 是 | 精确替换指令 |
| `seed` | integer string | 否 | 复现实验使用 |

响应必须为非空 PNG：

```http
HTTP/1.1 200 OK
Content-Type: image/png
```

本地 client 同时检查 Content-Type 和 PNG signature。若公司服务返回 JSON + base64，只修改 `EditorClient.edit` adapter，不修改节点和状态。

联调命令：

```bash
curl -X POST http://127.0.0.1:8003/edit \
  -F "image=@input.png;type=image/png" \
  -F "mask=@mask.png;type=image/png" \
  -F 'prompt=Edit only the masked region. Replace "AI AGEMT" with "AI AGENT" and preserve everything else.' \
  -F "seed=42" \
  --output edited.png
```

若 Qwen 推理接口不原生支持 mask，服务端仍可接收该字段并采用以下任一适配：

- 用 mask 裁出局部区域，编辑后融合；
- 将 mask 作为第二张参考图；
- 暂时忽略 mask，但保留严格区域指令。

论文实验前必须记录实际采用哪一种，不能统一称为原生 inpainting。

## 6. PaddleOCR 兼容注意事项

本地参考 client 使用旧式：

```python
PaddleOCR(...).ocr(image_path, cls=True)
```

PaddleOCR 3.x 常见接口为 `predict()`，结果结构也有变化。公司 agent 应先检查已固定版本：

```bash
python -c "import paddleocr; print(paddleocr.__version__)"
```

只要最终 adapter 能输出：

```json
{"lines": [{"text": "...", "confidence": 0.98, "bbox": [[x,y], ...]}]}
```

结构化 Critic 就不需要改。必须保留 bbox/polygon；没有位置时仍可评分，但不会触发 local edit。

## 7. 测试与 smoke test

安装依赖：

```bash
pip install -r requirements.txt
pip install -r requirements-ocr.txt
```

本地全量测试：

```bash
python -m pytest -q
```

本次参考仓库预期为 36 passed。

建议公司内真实 smoke test：

1. 准备至少 10 张图：中文 5 张、英文 5 张；
2. 覆盖完全正确、单字错误、漏文字、额外文字和双行顺序错误；
3. 检查 `ocr_round_NN.json` 中 constraints、instances 和 diagnosis；
4. editor 关闭时确认 mismatch 走 refiner；
5. editor 开启后用至少 3 张单个错字图检查 `/edit`；
6. 断开 editor 服务，确认流程不会无限循环且会回落 refiner。

Mock 测试不证明真实模型质量，只证明图路由、预算和 artifact 正确。

## 8. 冲突处理原则

- 公司已有更强 prompt parser：保留公司 parser，只转换成 `TextConstraint`。
- 公司已有 OCR service：保留 service，只统一输出 adapter。
- 公司已有 editor endpoint：只修改 `EditorClient` 请求/响应转换。
- 公司 state 使用 Pydantic 而不是 TypedDict：增加同名默认字段即可。
- 公司图中 memory_writer 位置不同：保证 OCR 诊断先落盘，再根据诊断路由。
- 公司已有持久化 checkpointer：新增字段必须有默认值，并验证旧 checkpoint 的 migration；本地参考版没有 checkpointer。

不要覆盖公司密钥、模型路径、NPU 配置、torch/torch_npu/CANN 版本或已验证的 Z-Image 服务代码。

## 9. 回滚方式

最安全的功能回滚：

```yaml
ocr:
  enabled: false
editor:
  enabled: false
```

只回滚编辑：关闭 editor 即可，结构化 OCR 会自动把 `local_edit` 降级为 `refine_prompt`。

代码级回滚时应按相反顺序：先删除 editor 图路由，再删 editor dependency/state，最后恢复旧 OCR verifier。不要先删 state 字段，否则运行中的节点可能访问缺失键。

## 10. 本周成果与后续边界

本周版本的价值是建立可测试的数据层和可插拔编辑闭环，不应直接宣称为论文最终创新。后续优先级：

1. 在人工标注集上验证细粒度诊断准确率；
2. 加入 OCR 不确定性/recheck，避免错误 OCR 驱动编辑；
3. 增加编辑前后非 mask 区域保持指标；
4. 在等调用预算下比较 regenerate-only、edit-only、best-of-N 和 adaptive route；
5. 最后再决定是否加入长期经验。
