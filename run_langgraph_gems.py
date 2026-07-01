import argparse
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

from agent.clients import GeneratorClient, MLLMClient
from agent.skill_manager import SkillManager
from langgraph_harness.config import AppConfig
from langgraph_harness.graph import build_graph
from langgraph_harness.mock_clients import MockGeneratorClient, MockMLLMClient
from langgraph_harness.nodes import NodeDependencies
from langgraph_harness.run_logger import RunLogger
from langgraph_harness.states import AgentState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the GEMS LangGraph text-to-image harness."
    )
    parser.add_argument(
        "--config",
        default="configs/langgraph_gems.yaml",
        help="YAML configuration path.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", help="One image-generation prompt.")
    source.add_argument(
        "--prompts",
        help="JSONL file; each line is a prompt string or an object with a prompt field.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic mock MLLM and generator clients.",
    )
    return parser.parse_args()


def load_prompts(args: argparse.Namespace) -> Iterable[Dict[str, str]]:
    if args.prompt is not None:
        yield {"task_id": "single", "prompt": args.prompt}
        return

    path = Path(args.prompts)
    if not path.is_file():
        raise FileNotFoundError("Prompt file not found: {}".format(path))
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, str):
                yield {"task_id": str(line_number), "prompt": value}
            elif isinstance(value, dict) and isinstance(value.get("prompt"), str):
                yield {
                    "task_id": str(value.get("task_id", line_number)),
                    "prompt": value["prompt"],
                }
            else:
                raise ValueError(
                    "Invalid prompt JSONL at line {}".format(line_number)
                )


def make_run_id() -> str:
    return "{}_{}".format(
        datetime.now().strftime("%Y%m%d_%H%M%S"),
        uuid.uuid4().hex[:8],
    )


def initial_state(
    run_id: str,
    task_id: str,
    prompt: str,
    artifact_dir: str,
    max_iterations: int,
) -> AgentState:
    return {
        "run_id": run_id,
        "task_id": task_id,
        "original_prompt": prompt,
        "current_prompt": prompt,
        "max_iterations": max_iterations,
        "iteration": 0,
        "artifact_dir": artifact_dir,
        "triggered_skills": [],
        "skill_instructions": "",
        "plan_text": "",
        "atomic_checks": [],
        "image_path": "",
        "best_image_path": "",
        "best_passed_count": -1,
        "verify_result": {},
        "passed_checks": [],
        "failed_checks": [],
        "failure_tags": [],
        "suggested_fix": "",
        "confidence": 0.0,
        "attempt_history": [],
        "memory_summary": "",
        "logs": [],
        "errors": [],
        "generation_latency_ms": 0,
    }


def run_item(
    config: AppConfig, item: Dict[str, str], mock: bool = False
) -> Dict[str, Any]:
    run_id = make_run_id()
    artifact_dir = str(Path(config.agent.artifact_root) / run_id)
    logger = RunLogger(artifact_dir)
    logger.write_json(
        "input.json",
        {"task_id": item["task_id"], "prompt": item["prompt"]},
    )
    logger.write_json("config.json", config.sanitized_dict())

    if mock:
        mllm = MockMLLMClient()
        generator = MockGeneratorClient()
    else:
        mllm = MLLMClient(
            base_url=config.mllm.base_url,
            api_key=config.mllm.api_key,
            model=config.mllm.model,
            timeout=config.mllm.timeout_seconds,
            max_retries=config.mllm.max_retries,
        )
        generator = GeneratorClient(
            url=config.generator.url,
            timeout=config.generator.timeout_seconds,
            max_retries=config.generator.max_retries,
            request_style="json",
        )

    dependencies = NodeDependencies(
        config=config,
        mllm=mllm,
        generator=generator,
        logger=logger,
        skill_manager=SkillManager(config.agent.skills_dir),
    )
    graph = build_graph(dependencies)
    state = initial_state(
        run_id=run_id,
        task_id=item["task_id"],
        prompt=item["prompt"],
        artifact_dir=artifact_dir,
        max_iterations=config.agent.max_iterations,
    )
    recursion_limit = max(50, config.agent.max_iterations * 10 + 20)
    return graph.invoke(state, {"recursion_limit": recursion_limit})


def main() -> int:
    args = parse_args()
    try:
        config = AppConfig.from_yaml(
            args.config,
            require_mllm=not args.mock,
            require_generator=not args.mock,
        )
        results = []
        for item in load_prompts(args):
            result = run_item(config, item, mock=args.mock)
            results.append(result)
            print(
                json.dumps(
                    {
                        "run_id": result["run_id"],
                        "status": result.get("final_status"),
                        "final_image_path": result.get("final_image_path"),
                        "report": result.get("final_report_path"),
                    },
                    ensure_ascii=False,
                )
            )
        return 0 if all(item.get("final_status") != "error" for item in results) else 1
    except Exception as exc:
        print("Error: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
