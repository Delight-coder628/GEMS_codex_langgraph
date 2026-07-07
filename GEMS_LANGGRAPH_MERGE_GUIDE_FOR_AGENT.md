# 将 LangGraph 改造融合进已跑通 GEMS 的执行手册

> 目标读者：公司内部负责修改代码的 Coding Agent。  
> 目标环境：现有 GEMS 已经能够使用 Qwen3.6 作为主 MLLM、Z-Image-Turbo 作为生成模型完成完整推理。

## 1. 最重要的迁移原则

现有远程服务器代码已经跑通，它是本次迁移的唯一运行基线。不要用参考仓库整体覆盖它。

本次工作应采用“旁路接入”：

```text
保留原 GEMS.run()               新增 LangGraph Harness
        │                                │
        ├── 调用同一个 Qwen3.6 服务 ──────┤
        └── 调用同一个 Z-Image 服务 ──────┘
```

迁移后的验收标准：

1. 原 `infer.py` 和原 GEMS 流程仍然能够运行。
2. 新 `run_langgraph_gems.py` 能够独立运行。
3. 两条流程复用已经跑通的 Qwen3.6 和 Z-Image-Turbo 服务。
4. 不重新部署模型，不擅自修改 CANN、torch、torch_npu、transformers 或 diffusers。
5. 不把公司 API Key、内网地址或模型凭证写入 Git。
6. 第一版只支持文生图，不实现输入图片编辑。

## 2. 开始修改前必须调查的事实

Coding Agent 必须先只读检查远程服务器上的工作代码，记录下面的信息，不允许根据参考代码猜测。

### 2.1 Qwen3.6 接口

定位原 GEMS 调用主 MLLM 的代码，确认：

- OpenAI-compatible `base_url`。
- API Key 从哪里读取。
- 实际 `model` 字符串。
- 使用 `/chat/completions` 还是其他路径。
- 图片是通过 `image_url`、base64 还是公司自定义字段传入。
- 返回正文位于 `message.content` 还是其他字段。
- 是否存在 `reasoning_content`；该字段必须视为可选。
- 单次请求的超时、最大 token 和重试策略。

最常见的原调用位置是：

```text
agent/base_agent.py
agent/GEMS.py
```

### 2.2 Z-Image-Turbo 接口

定位已经跑通的生成请求代码和模型服务代码，确认：

- 服务 URL。
- `POST` 请求使用 JSON body、query 参数还是 form。
- prompt 字段名称。
- 是否支持 `seed`、`width`、`height`、`num_inference_steps`、`guidance_scale`。
- 返回值是原始 PNG、JSON 中的 base64、图片路径还是 URL。
- Content-Type 是否为 `image/png`。
- 服务超时和错误返回格式。

参考 LangGraph 客户端默认发送：

```json
{
  "prompt": "image prompt",
  "seed": 42,
  "width": 1024,
  "height": 1024,
  "num_inference_steps": 9,
  "guidance_scale": 0.0
}
```

但必须以服务器上已跑通的协议为准。

### 2.3 当前软件环境

修改前保存环境快照：

```bash
python --version
python -c "import torch; print(torch.__version__)"
python -c "import torch_npu; import torch; print(torch.npu.is_available(), torch.npu.device_count())"
python -m pip freeze > before_langgraph_merge_requirements.txt
npu-smi info
```

不要把环境快照中的公司私有地址或凭证提交到仓库。

## 3. 需要从 LangGraph 参考版本迁移的内容

参考版本是本仓库当前代码。建议按下面顺序复制或合并。

### 3.1 可直接新增的文件

这些文件与原 GEMS 主循环没有覆盖关系：

```text
langgraph_harness/
├── __init__.py
├── config.py
├── graph.py
├── mock_clients.py
├── nodes.py
├── parsing.py
├── prompts.py
├── routing.py
├── run_logger.py
├── schemas.py
└── states.py

run_langgraph_gems.py
configs/langgraph_gems.yaml
agent/skills/hand_quality/SKILL.md
```

还应合并：

```text
.env.example
.gitignore
```

`.gitignore` 至少保留：

```gitignore
.env
outputs/
__pycache__/
*.py[cod]
.pytest_cache/
```

### 3.2 必须人工适配的文件

`agent/clients.py` 是两套框架共享服务的适配层，不能未经检查直接复制。

它应向 LangGraph 暴露两个稳定接口：

```python
class MLLMClient:
    def think(self, prompt: str, images=None) -> str:
        ...

    def chat(self, prompt: str, images=None):
        # 返回 (content, reasoning)，reasoning 不存在时返回空字符串
        ...


class GeneratorClient:
    def generate(
        self,
        prompt: str,
        seed=None,
        width=1024,
        height=1024,
        num_inference_steps=9,
        guidance_scale=0.0,
    ) -> bytes:
        # 无论服务原始响应是什么，最终统一返回 PNG bytes
        ...
```

适配原则：

1. LangGraph 节点只依赖以上接口，不应知道公司 HTTP 细节。
2. Qwen3.6 的请求格式优先复用原 GEMS 中已经验证成功的实现。
3. Z-Image-Turbo 的请求格式优先复用原生成代码。
4. 如果生成服务返回 base64、路径或 URL，在 `GeneratorClient` 内转换成 PNG bytes。
5. 客户端异常中不得包含 API Key。

### 3.3 默认不应覆盖的文件

下面文件属于已跑通基线：

```text
infer.py
agent/base_agent.py
agent/GEMS.py
agent/skill_manager.py
agent/server/中已经跑通的模型服务
eval/
```

LangGraph Harness 不直接依赖 `GEMS.run()`，因此不需要为了接入状态图而重写 `agent/GEMS.py`。

只有希望让原 GEMS 与 LangGraph 共用 `agent/clients.py` 时，才对 `base_agent.py` 做最小合并：

```python
self.mllm_client = MLLMClient(...)
self.generator_client = GeneratorClient(...)
```

并保持原公开接口不变：

```python
GEMS(gen_url=..., mllm_url=..., max_iterations=...)
agent.run({"prompt": "..."})
```

## 4. Qwen3.6 的具体接入方式

### 4.1 使用环境变量

真实值只在远程服务器 `.env` 或 ModelArts 环境变量中填写：

```dotenv
MLLM_BASE_URL=
MLLM_API_KEY=
MLLM_MODEL=
GENERATOR_URL=
```

对于当前已跑通环境：

```dotenv
MLLM_MODEL=<服务器实际使用的Qwen3.6模型名>
```

不要仅凭“Qwen3.6”猜测服务注册名称；必须复制原 GEMS 成功请求中的 model 值。

加载变量：

```bash
set -a
source .env
set +a
```

### 4.2 图片消息格式

LangGraph 的 verifier 和 memory 节点需要 Qwen3.6 看图。若原 GEMS 已能验证生成图片，直接复用其视觉消息格式。

标准 OpenAI-compatible 格式为：

```python
[
    {"type": "text", "text": "Image: "},
    {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64,<BASE64>"
        },
    },
    {"type": "text", "text": "Please verify ..."},
]
```

不要要求 Qwen3.6 必须返回 `reasoning_content`：

```python
content = message.content or ""
reasoning = getattr(message, "reasoning_content", None) or ""
```

### 4.3 结构化输出

Qwen3.6 需要为以下节点返回 JSON：

- skill router：skill ID 数组。
- decomposer：yes/no 检查问题数组。
- verifier：完整 `VerifyResult` 对象。

不要假设公司接口支持 JSON mode 或 tool calling。当前 Harness 已提供：

- Markdown JSON fence 提取。
- 尾逗号清理。
- Python 字面量回退。
- verifier 最多两次 JSON 修复。

如果 Qwen3.6 已支持可靠的 `response_format`，可在适配层增加可选配置，但不要把它作为唯一运行方式。

## 5. Z-Image-Turbo 的具体接入方式

### 情况 A：现有服务接收 JSON 并返回 PNG

保持 `run_langgraph_gems.py` 中：

```python
generator = GeneratorClient(
    url=config.generator.url,
    timeout=config.generator.timeout_seconds,
    max_retries=config.generator.max_retries,
    request_style="json",
)
```

### 情况 B：现有服务使用 query 参数

原 GEMS 仓库常见协议是：

```python
requests.post(gen_url, params={"prompt": prompt})
```

此时将 LangGraph 客户端配置为：

```python
request_style="query"
```

如果服务只接受 prompt，其他生成参数暂时忽略，由服务端继续使用已验证的默认值。

### 情况 C：现有服务返回 JSON/base64

不要修改 generator node。只修改 `GeneratorClient.generate()`，使其最终返回：

```python
png_bytes: bytes
```

generator node 会负责保存：

```text
outputs/langgraph_runs/<run_id>/round_01.png
```

### 是否迁移 `agent/server/z_image_npu.py`

如果公司服务器上的 Z-Image-Turbo 服务已经稳定运行，不迁移、不替换该服务。

只有需要统一 API 或当前服务不支持单/双卡时，才考虑使用参考版本的 `z_image_npu.py`。替换前必须单独验证：

- ModelScope 权重目录结构。
- `ZImagePipeline` 在现有 torch/torch_npu/CANN 组合上可加载。
- 单卡生成结果。
- 双卡两个完整副本。
- 并发请求、超时和 NPU 内存占用。

不要为了 LangGraph 融合去重新部署已经跑通的生成模型。

## 6. LangGraph 各节点与原 GEMS 的对应关系

```text
原 GEMS                         LangGraph Harness
---------------------------------------------------------------
GEMS.plan()                     skill_router + planner
GEMS.decompose()                decomposer
BaseAgent.generate()            generator
GEMS.verify_image()             structured verifier
SUMMARIZE_EXPERIENCE_TEMPLATE   memory_writer
REFINE_PROMPT_TEMPLATE          refiner
GEMS.run() 中的 for 循环         graph + conditional routing
最佳图片局部变量                  AgentState.best_image_path
print 日志                       attempts.jsonl/final_report.json
```

当前状态流：

```text
START
  -> skill_router
  -> planner
  -> decomposer
  -> generator
  -> verifier
  -> memory_writer
       ├── 全部通过 -> finalizer -> END
       ├── 达到上限 -> finalizer -> END
       ├── 发生错误 -> finalizer -> END
       └── 继续优化 -> refiner -> generator
```

内部 Agent 不要把整个图重新压回一个 `for` 循环，否则会失去节点测试、显式状态和条件路由的价值。

## 7. 推荐迁移步骤

### 第 0 步：保护已跑通代码

```bash
git status
git switch -c codex/langgraph-integration
```

如果公司环境不允许建分支，至少复制工作目录并保存 `pip freeze`。

不得在存在未确认修改时执行：

```text
git reset --hard
git checkout -- .
```

### 第 1 步：只加入 Harness

加入：

```text
langgraph_harness/
run_langgraph_gems.py
configs/langgraph_gems.yaml
```

此时不要改原 GEMS 和两个模型服务。

### 第 2 步：适配共享客户端

以已跑通请求为依据完成 `agent/clients.py`：

1. 先单独测试 Qwen3.6 纯文本。
2. 再测试 Qwen3.6 输入一张 PNG。
3. 单独测试 Z-Image-Turbo 返回 PNG bytes。
4. 最后才连接 LangGraph。

### 第 3 步：先运行纯 Mock 图

```bash
python run_langgraph_gems.py \
  --config configs/langgraph_gems.yaml \
  --prompt "a red apple" \
  --mock
```

应生成：

```text
input.json
config.json
attempts.jsonl
round_01.png
final_report.json
```

Mock 失败说明是代码、依赖或路径问题，与 Qwen3.6/Z-Image 无关。

### 第 4 步：连接真实服务

先确认：

```bash
curl <Z-IMAGE-HEALTH-URL>
```

加载 `.env` 后运行简单提示：

```bash
python run_langgraph_gems.py \
  --config configs/langgraph_gems.yaml \
  --prompt "一只坐在木桌上的橘猫，写实摄影"
```

第一条真实测试不要包含复杂文字、多人手部或空间关系，先确认整条链路通畅。

### 第 5 步：验证失败重试

使用至少包含两个可验证要求的 prompt：

```text
一张蓝色海报，中央准确写着“GEMS TEST”，右下角有一只白猫
```

确认：

- decomposer 产生多个检查项。
- verifier 输出结构化 JSON。
- 存在失败项时进入 refiner。
- 达到成功或最大轮次时进入 finalizer。
- `best_image_path` 指向通过项最多的图片。

### 第 6 步：回归原 GEMS

重新运行原先已经成功的命令。原输出必须仍然一致，至少应满足：

- 原 `infer.py` 可启动。
- 原 GEMS 构造参数未改变。
- Qwen3.6 请求仍成功。
- Z-Image-Turbo 请求仍成功。

## 8. 依赖安装注意事项

远程服务器已经能运行模型，不要直接使用一个全新的 requirements 覆盖模型环境。

优先只安装 Agent 层依赖：

```bash
pip install \
  "langgraph==0.2.76" \
  "langchain-core>=0.3.40,<0.4" \
  "openai>=1.60,<2" \
  "pydantic>=2.7,<3" \
  "PyYAML>=6,<7" \
  "requests>=2.31,<3"
```

如果现有 Qwen3.6 客户端依赖特定 OpenAI SDK 版本，以已跑通版本为准，不强制升级。

禁止在没有完整 CANN 版本信息时执行：

```bash
pip install --upgrade torch
pip install --upgrade torch-npu
pip install --upgrade diffusers transformers
```

Python 3.9 基线使用 `langgraph==0.2.76`。本项目不启用持久化 checkpointer；如果以后升级到 Python 3.10 和 LangGraph 1.x，应作为单独迁移任务处理。

## 9. 测试清单

### 静态和单元测试

```bash
python -m compileall -q agent langgraph_harness run_langgraph_gems.py
pytest -q
```

至少覆盖：

- 空 MLLM 配置会明确报错。
- API Key 不进入 `config.json`。
- 图片正确编码为视觉消息。
- JSON fence、尾逗号和错误 JSON 修复。
- success、retry、max iteration、error 四种路由。
- generator 返回非 PNG、空内容、超时、4xx/5xx。
- best image 选择。
- attempts 和 final report 落盘。

### 集成测试

```text
[ ] 原 GEMS 简单 prompt 成功
[ ] LangGraph mock 成功
[ ] Qwen3.6 纯文本成功
[ ] Qwen3.6 看 PNG 成功
[ ] Z-Image-Turbo 单次生成成功
[ ] LangGraph 简单 prompt 成功
[ ] LangGraph 复杂 prompt 发生至少一次验证
[ ] 最大迭代数能够终止
[ ] outputs 中没有 API Key
[ ] 原 eval 文件未被修改
```

## 10. 常见整合问题

### 422 Unprocessable Entity

通常是 Z-Image 服务请求格式不一致。检查它接收 JSON 还是 query：

```python
request_style="json"
# 或
request_style="query"
```

### Generator returned non-PNG

服务可能返回 JSON/base64。将响应转换放到 `GeneratorClient`，不要让 node 解析公司协议。

### Qwen3.6 能回答文本但 verifier 失败

检查：

- 模型是否真的是视觉模型部署。
- `image_url` 是否支持 data URL。
- 图片 MIME 类型是否正确。
- 公司网关是否限制请求体大小。
- `max_tokens` 是否过小。

### `reasoning_content` AttributeError

不要直接访问该字段：

```python
reasoning = getattr(message, "reasoning_content", None) or ""
```

### GraphRecursionError

CLI 应按最大轮次设置足够的 recursion limit：

```python
recursion_limit = max(50, max_iterations * 10 + 20)
```

不要通过无限增大 recursion limit 掩盖路由错误。

### 原 GEMS 被改坏

优先恢复原 `infer.py`、`base_agent.py` 和 `GEMS.py`，让原流程重新成功。LangGraph Harness 可以仅依赖独立客户端运行，不需要侵入原循环。

### NPU 环境突然出现算子或版本错误

如果问题发生在安装 Agent 依赖之后：

1. 对比 `before_langgraph_merge_requirements.txt`。
2. 检查 torch、torch_npu、transformers、diffusers 是否被升级。
3. 恢复原模型环境。
4. 将 Agent 层放进独立 Python 环境，通过 HTTP 调用原模型服务。

不要在生产环境中连续试装多个 torch_npu 版本。

## 11. 给内部 Coding Agent 的最终约束

执行本迁移时必须遵守：

1. 先读现有运行代码，再编辑。
2. 已跑通的 Qwen3.6 和 Z-Image-Turbo 请求协议是事实来源。
3. 新增旁路，不整体重写原 GEMS。
4. 服务差异只在 adapter/client 层消化。
5. 每完成一个迁移阶段就运行对应测试。
6. 任何包升级必须说明原因和影响，禁止顺手升级模型栈。
7. 真实密钥和内网地址只放环境变量。
8. 不修改 `eval/` 和 ArtiMuse 内置模型代码。
9. 不实现图像编辑，不引入其他生图模型。
10. 最终交付必须同时提供原 GEMS 和 LangGraph 两条成功命令，以及一次真实运行的 `final_report.json`。

完成以上步骤后，LangGraph 是现有 GEMS 的可观测执行层，而不是另一个模型部署项目。两条路径应共享已经验证成功的 Qwen3.6 和 Z-Image-Turbo 服务，并能够随时相互对照。
