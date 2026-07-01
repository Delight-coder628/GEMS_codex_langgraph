# GEMS × LangGraph 文生图框架说明

## 1. 改动概览

本项目保留原 GEMS 推理和评测代码，新增一条旁路 LangGraph 工作流。新工作流只支持文生图，生成后由公司 OpenAI-compatible MLLM 看图验证，并在失败时总结经验、改写 prompt、再次调用 Z-Image-Turbo。

流程如下：

```text
skill_router -> planner -> decomposer -> generator -> verifier
     -> memory_writer -> success/finalizer
                      -> refiner -> generator
```

每次运行都会写入 `outputs/langgraph_runs/<run_id>/`：

```text
input.json          本次输入
config.json         已脱敏配置
attempts.jsonl      每轮 prompt、检查结果、失败标签和耗时
round_01.png        每轮图片
final_report.json   最终状态和最佳图片
```

## 2. 目录和文件职责

### 原 GEMS

- `infer.py`：原始最小推理入口，作为 baseline 保留。
- `agent/base_agent.py`：生成、MLLM 视觉调用和抽象 Agent 接口。
- `agent/GEMS.py`：原始 plan、decompose、generate、verify、memory、refine 大循环。
- `agent/skill_manager.py`：扫描 `agent/skills/*/SKILL.md` 并解析描述和指令。
- `agent/clients.py`：新增的共享 OpenAI-compatible MLLM 客户端和生成服务客户端。
- `agent/server/z_image.py`：原 CUDA Z-Image 服务，仅保留作原始参考。
- `agent/server/qwen_image.py`、`kimi.sh`：原仓库其他模型的参考启动代码，新框架不会调用。
- `agent/server/z_image_npu.py`：新增 ModelArts Ascend 单卡/双卡服务。
- `agent/skills/`：原 aesthetic、creative、spatial、text skills，以及新增 hand quality skill。
- `assets/`：README 展示图片。

### LangGraph Harness

- `run_langgraph_gems.py`：单 prompt、JSONL 批量和 mock 模式入口。
- `langgraph_harness/config.py`：YAML、环境变量展开、参数校验和密钥脱敏。
- `states.py`、`schemas.py`：图状态和 verifier/日志 Pydantic schema。
- `prompts.py`：路由、规划、拆解、验证、记忆和改写提示模板。
- `parsing.py`：JSON 代码块、尾逗号、Python 字面量和 verifier schema 容错。
- `nodes.py`：所有 LangGraph 业务节点。
- `routing.py`、`graph.py`：条件路由和图编译。
- `run_logger.py`：图片、JSON、JSONL 和最终报告持久化。
- `mock_clients.py`：无需 MLLM、生成模型或 NPU 的确定性测试客户端。

### 评测代码（未修改）

- `eval/GenEval2.py`：GenEval2 任务生成入口。
- `eval/CREA/CREA.py`：CREA 图片生成；`eval.py` 使用外部 MLLM judge 评分；`judge_prompt.txt` 是评分规则。
- `eval/ArtiMuse/gen_artimuse.py`：ArtiMuse 图片生成；`eval_artimuse.py` 调用审美模型评分。
- `eval/ArtiMuse/ArtiMuse/`：上游 ArtiMuse evaluator 副本。`src/eval/` 是数据集/单图评估入口，`src/artimuse/internvl/` 包含会话模板、分布式工具、InternViT、InternLM2、Phi3 和审美 token 实现；`test_datasets/` 是四套审美测试元数据。

## 3. 版本和安装

当前基线：

- Python 3.9.x
- ModelArts 已安装的 PyTorch 2.1.0 和配套 `torch_npu`
- `langgraph==0.2.76`
- `diffusers==0.36.0`
- `transformers>=4.51,<5`

不要通过本项目升级 `torch` 或 `torch_npu`。它们必须与 ModelArts 镜像的 CANN、驱动和固件匹配。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python --version
python -c "import torch, torch_npu; print(torch.__version__); print(torch.npu.is_available(), torch.npu.device_count())"
npu-smi info
pip install -r requirements.txt
```

`langgraph==0.2.76` 是为 Python 3.9 选择的兼容版本，本项目不启用其持久化 checkpointer。以后需要 checkpoint 时，应升级到 Python 3.10 和已修复安全问题的 LangGraph 1.x。

## 4. 配置

复制模板并填入公司参数：

```bash
cp .env.example .env
```

```dotenv
MLLM_BASE_URL=
MLLM_API_KEY=
MLLM_MODEL=
GENERATOR_URL=http://127.0.0.1:8001/generate
ZIMAGE_MODEL_PATH=/absolute/path/to/Z-Image-Turbo
ASCEND_RT_VISIBLE_DEVICES=0
```

加载变量：

```bash
set -a
source .env
set +a
```

MLLM 必须支持 OpenAI-compatible `/v1/chat/completions` 和 `image_url`。`reasoning_content` 是可选字段。

模型目录应至少包含：

```text
Z-Image-Turbo/
├── model_index.json
├── transformer/
├── text_encoder/
└── vae/
```

## 5. 启动 Z-Image-Turbo

单卡，例如物理卡 3：

```bash
python -m agent.server.z_image_npu \
  --devices 3 \
  --model-path ../Z-Image-Turbo \
  --port 8001
```

双卡，例如物理卡 3 和 6：

```bash
python -m agent.server.z_image_npu \
  --devices 3,6 \
  --model-path ../Z-Image-Turbo \
  --port 8001
```

双卡模式会加载两个完整模型副本，提高并发吞吐；单次请求不会跨卡。`ASCEND_RT_VISIBLE_DEVICES=3,6` 后，进程内对应逻辑设备是 `npu:0,npu:1`。

服务必须只启动一个 Uvicorn 主进程，不能再使用 `uvicorn --workers 2`，否则每个 Web worker 都会重复加载模型。

检查服务：

```bash
curl http://127.0.0.1:8001/health

curl -X POST http://127.0.0.1:8001/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"一只坐在窗边的橘猫","seed":42}' \
  --output smoke.png
```

## 6. 运行 Agent

单 prompt：

```bash
python run_langgraph_gems.py \
  --config configs/langgraph_gems.yaml \
  --prompt "一张写有“AI AGENT WEEK 3”的未来主义海报"
```

批量 JSONL：

```bash
python run_langgraph_gems.py \
  --config configs/langgraph_gems.yaml \
  --prompts prompts.jsonl
```

每行可以是字符串，或 `{"task_id":"001","prompt":"..."}`。

不连接任何服务的流程测试：

```bash
python run_langgraph_gems.py \
  --config configs/langgraph_gems.yaml \
  --prompt "a red apple" \
  --mock
```

运行测试：

```bash
pytest -q
```

## 7. 常见问题

- `Missing required MLLM settings`：环境变量没有加载，或仍为空。
- `torch.npu is unavailable`：没有加载 CANN 环境、`torch_npu` 与 torch/CANN 不匹配，或 ModelArts 没挂载 NPU。
- worker 启动失败：先检查 `/health`、模型目录结构、`npu-smi info` 和卡是否被其他进程占用。
- NPU unsupported operator：先记录完整算子报错；不要单独升级 torch，需选择一套与 CANN 匹配的 torch/torch_npu 镜像后再验证。
- 429：生成队列已满，可降低上游并发或调整 `ZIMAGE_QUEUE_SIZE`。
- 504：调大 `ZIMAGE_REQUEST_TIMEOUT`，并确认模型 worker 仍存活。
- verifier JSON 错误：框架会自动修复两次；仍失败时在 `final_report.json` 中以 error 结束，不会无限重试。
- 本期不支持输入图片编辑、局部重绘或 mask；Z-Image-Turbo 只作为文生图模型使用。
