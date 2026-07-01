import json
from pathlib import Path

from langgraph_harness.config import AgentConfig, AppConfig, GeneratorConfig, MLLMConfig
from run_langgraph_gems import run_item


def test_mock_graph_writes_complete_artifacts(tmp_path: Path) -> None:
    config = AppConfig(
        mllm=MLLMConfig(),
        generator=GeneratorConfig(),
        agent=AgentConfig(
            max_iterations=2,
            artifact_root=str(tmp_path),
            skills_dir="agent/skills",
            verbose=False,
        ),
    )
    result = run_item(
        config,
        {"task_id": "test", "prompt": "A simple red apple"},
        mock=True,
    )

    assert result["final_status"] == "success"
    run_dir = Path(result["artifact_dir"])
    assert (run_dir / "input.json").is_file()
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "attempts.jsonl").is_file()
    assert (run_dir / "round_01.png").is_file()
    report = json.loads((run_dir / "final_report.json").read_text(encoding="utf-8"))
    assert report["final_status"] == "success"
    assert report["iterations"] == 1
