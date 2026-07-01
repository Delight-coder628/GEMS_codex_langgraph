import abc
from typing import List, Optional, Tuple

from agent.clients import GeneratorClient, MLLMClient
from agent.skill_manager import SkillManager


class BaseAgent(metaclass=abc.ABCMeta):
    def __init__(
        self,
        gen_url: str,
        mllm_url: str,
        mllm_api_key: str = "",
        mllm_model: str = "",
        request_timeout: float = 600.0,
    ):
        self.gen_url = gen_url
        self.mllm_url = mllm_url
        self.mllm_client = MLLMClient(
            base_url=mllm_url,
            api_key=mllm_api_key,
            model=mllm_model,
            timeout=request_timeout,
        )
        # Keep the original query-string protocol for the baseline CUDA servers.
        self.generator_client = GeneratorClient(
            url=gen_url,
            timeout=request_timeout,
            request_style="query",
        )
        self.skill_manager = SkillManager()

    def generate(self, prompt: str) -> bytes:
        return self.generator_client.generate(prompt)

    def edit(self, prompt: str, image: bytes):
        raise NotImplementedError(
            "Image editing is intentionally out of scope for Z-Image-Turbo."
        )

    def think(self, prompt: str, images: Optional[List[bytes]] = None) -> str:
        try:
            return self.mllm_client.think(prompt, images=images)
        except Exception as e:
            raise RuntimeError("MLLM: {}".format(e)) from e

    def think_with_thought(
        self, prompt: str, images: Optional[List[bytes]] = None
    ) -> Tuple[str, str]:
        try:
            return self.mllm_client.chat(prompt, images=images)
        except Exception as e:
            raise RuntimeError("MLLM: {}".format(e)) from e

    def run(self, item: dict) -> bytes:
        raise NotImplementedError
