from pathlib import Path

import pytest

from langgraph_harness.config import AppConfig


CONFIG = """
mllm:
  base_url: "${MLLM_BASE_URL}"
  api_key: "${MLLM_API_KEY}"
  model: "${MLLM_MODEL}"
generator:
  url: "${GENERATOR_URL}"
agent: {}
"""


def test_config_expands_environment_and_redacts_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG, encoding="utf-8")
    monkeypatch.setenv("MLLM_BASE_URL", "http://mllm/v1")
    monkeypatch.setenv("MLLM_API_KEY", "secret")
    monkeypatch.setenv("MLLM_MODEL", "company-vlm")
    monkeypatch.setenv("GENERATOR_URL", "http://generator/generate")

    config = AppConfig.from_yaml(str(path))

    assert config.mllm.model == "company-vlm"
    assert config.sanitized_dict()["mllm"]["api_key"] == "***"
    assert "secret" not in config.sanitized_json()


def test_config_rejects_missing_mllm_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG, encoding="utf-8")
    for name in ("MLLM_BASE_URL", "MLLM_API_KEY", "MLLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GENERATOR_URL", "http://generator/generate")

    with pytest.raises(ValueError, match="Missing required MLLM"):
        AppConfig.from_yaml(str(path))
