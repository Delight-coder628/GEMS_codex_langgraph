import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


ENV_PATTERN = re.compile(r"^\$\{([A-Z0-9_]+)\}$")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        match = ENV_PATTERN.match(value)
        if match:
            return os.getenv(match.group(1), "")
        return value
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


class MLLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str = ""
    api_key: str = ""
    model: str = ""
    timeout_seconds: float = Field(default=600.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=10)

    def validate_required(self) -> None:
        missing = [
            name
            for name, value in (
                ("MLLM_BASE_URL", self.base_url),
                ("MLLM_API_KEY", self.api_key),
                ("MLLM_MODEL", self.model),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                "Missing required MLLM settings: {}".format(", ".join(missing))
            )


class GeneratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = ""
    timeout_seconds: float = Field(default=600.0, gt=0)
    max_retries: int = Field(default=1, ge=0, le=10)
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)
    num_inference_steps: int = Field(default=9, ge=1, le=100)
    guidance_scale: float = Field(default=0.0, ge=0.0, le=20.0)
    seed: Optional[int] = Field(default=None, ge=0)


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_iterations: int = Field(default=5, ge=1, le=20)
    verifier_repair_attempts: int = Field(default=2, ge=0, le=5)
    artifact_root: str = "outputs/langgraph_runs"
    skills_dir: str = "agent/skills"
    verbose: bool = True


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mllm: MLLMConfig
    generator: GeneratorConfig
    agent: AgentConfig

    @model_validator(mode="after")
    def validate_generator_url(self) -> "AppConfig":
        if self.generator.url and not self.generator.url.startswith(
            ("http://", "https://")
        ):
            raise ValueError("generator.url must start with http:// or https://")
        return self

    @classmethod
    def from_yaml(
        cls, path: str, require_mllm: bool = True, require_generator: bool = True
    ) -> "AppConfig":
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError("Config file not found: {}".format(config_path))
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        config = cls.model_validate(_expand_env(raw))
        if require_mllm:
            config.mllm.validate_required()
        if require_generator and not config.generator.url:
            raise ValueError("GENERATOR_URL is required.")
        return config

    def sanitized_dict(self) -> Dict[str, Any]:
        data = self.model_dump()
        data["mllm"]["api_key"] = "***" if self.mllm.api_key else ""
        return data

    def sanitized_json(self) -> str:
        return json.dumps(self.sanitized_dict(), ensure_ascii=False, indent=2)
