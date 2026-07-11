import json
import importlib.util
from pathlib import Path

import pytest

from langgraph_harness.config import (
    AgentConfig,
    AppConfig,
    EditorConfig,
    GeneratorConfig,
    MLLMConfig,
    OCRConfig,
)
from langgraph_harness.editor import MockEditorClient
from langgraph_harness.mock_clients import MOCK_PNG
from langgraph_harness.ocr import MockOCRClient, OCRLine, OCRResult
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


def test_mock_text_prompt_with_ocr_match_writes_ocr_artifact(tmp_path: Path) -> None:
    config = AppConfig(
        mllm=MLLMConfig(),
        generator=GeneratorConfig(),
        ocr=OCRConfig(enabled=True),
        agent=AgentConfig(
            max_iterations=2,
            artifact_root=str(tmp_path),
            skills_dir="agent/skills",
            verbose=False,
        ),
    )
    result = run_item(
        config,
        {"task_id": "test", "prompt": 'A poster saying "AI AGENT WEEK 3"'},
        mock=True,
        ocr_client=MockOCRClient(text="AI AGENT WEEK 3", confidence=0.98),
    )

    assert result["final_status"] == "success"
    run_dir = Path(result["artifact_dir"])
    ocr_result = json.loads((run_dir / "ocr_round_01.json").read_text(encoding="utf-8"))
    assert ocr_result["ocr_score"]["passed"] is True
    report = json.loads((run_dir / "final_report.json").read_text(encoding="utf-8"))
    assert report["ocr_score"]["exact_match"] is True


def test_mock_text_prompt_with_ocr_mismatch_retries(tmp_path: Path) -> None:
    config = AppConfig(
        mllm=MLLMConfig(),
        generator=GeneratorConfig(),
        ocr=OCRConfig(enabled=True, normalized_match_threshold=0.95),
        agent=AgentConfig(
            max_iterations=1,
            artifact_root=str(tmp_path),
            skills_dir="agent/skills",
            verbose=False,
        ),
    )
    result = run_item(
        config,
        {"task_id": "test", "prompt": 'A poster saying "AI AGENT WEEK 3"'},
        mock=True,
        ocr_client=MockOCRClient(text="WRONG TEXT", confidence=0.98),
    )

    assert result["final_status"] == "max_iter_reached"
    assert "text_render_error" in result["failure_tags"]
    run_dir = Path(result["artifact_dir"])
    ocr_result = json.loads((run_dir / "ocr_round_01.json").read_text(encoding="utf-8"))
    assert ocr_result["ocr_score"]["passed"] is False
    assert ocr_result["ocr_score"]["failure_reason"] == "ocr_text_mismatch"


@pytest.mark.skipif(
    importlib.util.find_spec("PIL") is None, reason="Pillow is required for edit masks"
)
def test_mock_mismatch_uses_editor_then_rechecks_ocr(tmp_path: Path) -> None:
    class SequenceOCRClient:
        def __init__(self):
            self.calls = 0

        def recognize(self, image_path: str) -> OCRResult:
            self.calls += 1
            text = "AI AGEMT" if self.calls == 1 else "AI AGENT"
            return OCRResult(
                lines=[
                    OCRLine(
                        text=text,
                        confidence=0.98,
                        bbox=[[0, 0], [1, 0], [1, 1], [0, 1]],
                    )
                ]
            )

    config = AppConfig(
        mllm=MLLMConfig(),
        generator=GeneratorConfig(),
        ocr=OCRConfig(enabled=True, normalized_match_threshold=0.95),
        editor=EditorConfig(enabled=True, url="http://editor/edit"),
        agent=AgentConfig(
            max_iterations=2,
            artifact_root=str(tmp_path),
            skills_dir="agent/skills",
            verbose=False,
        ),
    )
    result = run_item(
        config,
        {"task_id": "test", "prompt": 'A poster saying "AI AGENT"'},
        mock=True,
        ocr_client=SequenceOCRClient(),
        editor_client=MockEditorClient(MOCK_PNG),
    )

    assert result["final_status"] == "success"
    assert result["edit_attempts"] == 1
    assert result["ocr_diagnosis"]["passed"] is True
    run_dir = Path(result["artifact_dir"])
    assert (run_dir / "edit_round_01_01.png").is_file()
    assert (run_dir / "ocr_round_01_edit_01.json").is_file()


def test_editor_failure_falls_back_without_looping(tmp_path: Path) -> None:
    class PolygonOCRClient:
        def recognize(self, image_path: str) -> OCRResult:
            return OCRResult(
                lines=[
                    OCRLine(
                        text="AI AGEMT",
                        confidence=0.98,
                        bbox=[[0, 0], [1, 0], [1, 1], [0, 1]],
                    )
                ]
            )

    class FailingEditor:
        def edit(self, *args, **kwargs):
            raise RuntimeError("mock editor unavailable")

    config = AppConfig(
        mllm=MLLMConfig(),
        generator=GeneratorConfig(),
        ocr=OCRConfig(enabled=True, normalized_match_threshold=0.95),
        editor=EditorConfig(
            enabled=True, url="http://editor/edit", max_edits_per_run=1
        ),
        agent=AgentConfig(
            max_iterations=2,
            artifact_root=str(tmp_path),
            skills_dir="agent/skills",
            verbose=False,
        ),
    )
    result = run_item(
        config,
        {"task_id": "test", "prompt": 'A poster saying "AI AGENT"'},
        mock=True,
        ocr_client=PolygonOCRClient(),
        editor_client=FailingEditor(),
    )
    assert result["final_status"] == "max_iter_reached"
    assert result["edit_attempts"] == 1
    assert result["iteration"] == 2
    assert any("Editor fallback" in item for item in result["logs"])
