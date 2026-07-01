import base64
import os
from typing import Any, Dict, List, Optional, Tuple

import requests
from openai import OpenAI


class MLLMClient:
    """OpenAI-compatible multimodal chat client used by both GEMS paths."""

    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        model: str = "",
        timeout: float = 600.0,
        max_retries: int = 2,
    ):
        self.base_url = (base_url or os.getenv("MLLM_BASE_URL", "")).strip()
        self.api_key = api_key or os.getenv("MLLM_API_KEY", "")
        self.model = (model or os.getenv("MLLM_MODEL", "")).strip()
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = None

    def validate(self) -> None:
        missing = []
        if not self.base_url:
            missing.append("MLLM_BASE_URL")
        if not self.api_key:
            missing.append("MLLM_API_KEY")
        if not self.model:
            missing.append("MLLM_MODEL")
        if missing:
            raise ValueError(
                "Missing required MLLM configuration: {}. "
                "Fill the variables in your environment; do not commit secrets.".format(
                    ", ".join(missing)
                )
            )

    @property
    def client(self) -> OpenAI:
        self.validate()
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._client

    @staticmethod
    def build_content(prompt: str, images: Optional[List[bytes]] = None) -> List[Dict[str, Any]]:
        images = images or []
        segments = prompt.split("<image>")
        content: List[Dict[str, Any]] = []

        for index, segment in enumerate(segments):
            if segment:
                content.append({"type": "text", "text": segment})
            if index < len(images):
                encoded = base64.b64encode(images[index]).decode("utf-8")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,{}".format(encoded)},
                    }
                )

        for image in images[len(segments) :]:
            encoded = base64.b64encode(image).decode("utf-8")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,{}".format(encoded)},
                }
            )
        return content

    def chat(
        self,
        prompt: str,
        images: Optional[List[bytes]] = None,
        max_tokens: int = 16384,
        temperature: Optional[float] = None,
    ) -> Tuple[str, str]:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": self.build_content(prompt, images)}
            ],
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature

        response = self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        content = message.content or ""
        reasoning = getattr(message, "reasoning_content", None) or ""
        return content, reasoning

    def think(self, prompt: str, images: Optional[List[bytes]] = None) -> str:
        content, _ = self.chat(prompt, images=images)
        return content


class GeneratorClient:
    """HTTP client for the Z-Image-Turbo generation service."""

    def __init__(
        self,
        url: str = "",
        timeout: float = 600.0,
        max_retries: int = 1,
        request_style: str = "json",
    ):
        self.url = (url or os.getenv("GENERATOR_URL", "")).strip()
        self.timeout = timeout
        self.max_retries = max_retries
        self.request_style = request_style

    def validate(self) -> None:
        if not self.url:
            raise ValueError("GENERATOR_URL is required.")
        if self.request_style not in {"json", "query"}:
            raise ValueError("request_style must be 'json' or 'query'.")

    def generate(
        self,
        prompt: str,
        seed: Optional[int] = None,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 9,
        guidance_scale: float = 0.0,
    ) -> bytes:
        self.validate()
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        }
        if seed is not None:
            payload["seed"] = seed

        last_error: Optional[Exception] = None
        for _ in range(self.max_retries + 1):
            try:
                kwargs = (
                    {"json": payload}
                    if self.request_style == "json"
                    else {"params": {"prompt": prompt}}
                )
                response = requests.post(self.url, timeout=self.timeout, **kwargs)
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                if "image/png" not in content_type:
                    raise RuntimeError(
                        "Generator returned non-PNG content type: {}".format(
                            content_type or "<missing>"
                        )
                    )
                if not response.content:
                    raise RuntimeError("Generator returned an empty response.")
                return response.content
            except (requests.RequestException, RuntimeError) as exc:
                last_error = exc

        raise RuntimeError("Generator request failed: {}".format(last_error))
