from typing import Any, Dict

import pytest

from agent.clients import GeneratorClient, MLLMClient


def test_mllm_content_inserts_image_at_placeholder() -> None:
    content = MLLMClient.build_content("before <image> after", [b"image"])
    assert [item["type"] for item in content] == ["text", "image_url", "text"]
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


class FakeResponse:
    content = b"png"
    headers = {"content-type": "image/png"}

    def raise_for_status(self) -> None:
        return None


def test_generator_client_sends_json(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr("agent.clients.requests.post", fake_post)
    client = GeneratorClient("http://generator/generate", request_style="json")
    assert client.generate("a cat", seed=42) == b"png"
    assert captured["json"]["prompt"] == "a cat"
    assert captured["json"]["seed"] == 42


def test_generator_client_rejects_non_png(monkeypatch: pytest.MonkeyPatch) -> None:
    class JsonResponse(FakeResponse):
        headers = {"content-type": "application/json"}

    monkeypatch.setattr(
        "agent.clients.requests.post", lambda *args, **kwargs: JsonResponse()
    )
    client = GeneratorClient(
        "http://generator/generate", max_retries=0, request_style="json"
    )
    with pytest.raises(RuntimeError, match="non-PNG"):
        client.generate("a cat")
