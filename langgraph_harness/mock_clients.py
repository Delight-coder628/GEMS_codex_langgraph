import base64
from typing import List, Optional, Tuple


MOCK_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/x8AAusB9Y9Zl1sAAAAASUVORK5CYII="
)


class MockGeneratorClient:
    def generate(self, prompt: str, **kwargs) -> bytes:
        return MOCK_PNG


class MockMLLMClient:
    def think(self, prompt: str, images: Optional[List[bytes]] = None) -> str:
        lowered = prompt.lower()
        if "skill router" in lowered:
            return "[]"
        if "json array of strings" in lowered:
            return '["Does the image satisfy the requested scene?"]'
        if "strict image-generation verifier" in lowered:
            return (
                '{"passed": true, "checks": [{"question": "Does the image satisfy '
                'the requested scene?", "passed": true, "evidence": "mock", '
                '"failure_tags": [], "suggested_fix": "", "confidence": 1.0}], '
                '"failed_checks": [], "failure_tags": [], "suggested_fix": "", '
                '"confidence": 1.0}'
            )
        if "summarize this image-generation attempt" in lowered:
            return "Mock generation passed the requested visual check."
        if "repair the following verifier response" in lowered:
            return "{}"
        if "return only the rewritten prompt" in lowered:
            return prompt.split("Original request:", 1)[-1].split(
                "Skill instructions:", 1
            )[0].strip()
        return "Mock refined prompt"

    def chat(
        self, prompt: str, images: Optional[List[bytes]] = None
    ) -> Tuple[str, str]:
        return self.think(prompt, images=images), ""
