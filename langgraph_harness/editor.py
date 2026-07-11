import io
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import requests


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class EditorClient:
    """HTTP client for an internal Qwen-Image-Edit compatible service."""

    def __init__(
        self,
        url: str,
        timeout: float = 600.0,
        max_retries: int = 1,
    ):
        self.url = url.strip()
        self.timeout = timeout
        self.max_retries = max_retries

    def edit(
        self,
        image_path: str,
        prompt: str,
        mask_bytes: Optional[bytes] = None,
        seed: Optional[int] = None,
    ) -> bytes:
        if not self.url:
            raise ValueError("EDITOR_URL is required when editor is enabled.")
        image_bytes = Path(image_path).read_bytes()
        files: Dict[str, Tuple[str, bytes, str]] = {
            "image": ("image.png", image_bytes, "image/png")
        }
        if mask_bytes is not None:
            files["mask"] = ("mask.png", mask_bytes, "image/png")
        data: Dict[str, str] = {"prompt": prompt}
        if seed is not None:
            data["seed"] = str(seed)

        last_error = None
        for _ in range(self.max_retries + 1):
            try:
                response = requests.post(
                    self.url,
                    files=files,
                    data=data,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                if "image/png" not in content_type:
                    raise RuntimeError(
                        "Editor returned non-PNG content type: {}".format(
                            content_type or "<missing>"
                        )
                    )
                if not response.content.startswith(PNG_SIGNATURE):
                    raise RuntimeError("Editor returned invalid or empty PNG content.")
                return response.content
            except (requests.RequestException, RuntimeError) as exc:
                last_error = exc
        raise RuntimeError("Editor request failed: {}".format(last_error))


class MockEditorClient:
    def __init__(self, content: bytes):
        self.content = content
        self.calls = []

    def edit(self, image_path: str, prompt: str, **kwargs: Any) -> bytes:
        self.calls.append(
            {
                "image_path": image_path,
                "prompt": prompt,
                "mask_bytes": kwargs.get("mask_bytes"),
                "seed": kwargs.get("seed"),
            }
        )
        return self.content


def build_polygon_mask(
    image_path: str,
    polygon: Sequence[Sequence[float]],
    padding_ratio: float = 0.25,
) -> bytes:
    from PIL import Image, ImageDraw

    with Image.open(image_path) as image:
        width, height = image.size
    points = [
        (float(point[0]), float(point[1]))
        for point in polygon
        if isinstance(point, (list, tuple)) and len(point) >= 2
    ]
    if not points:
        raise ValueError("A non-empty OCR polygon is required to build an edit mask.")
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    padding = max(1.0, (max_y - min_y) * padding_ratio)
    box = (
        max(0, int(min_x - padding)),
        max(0, int(min_y - padding)),
        min(width - 1, int(max_x + padding)),
        min(height - 1, int(max_y + padding)),
    )
    mask = Image.new("L", (width, height), color=0)
    ImageDraw.Draw(mask).rectangle(box, fill=255)
    output = io.BytesIO()
    mask.save(output, format="PNG")
    return output.getvalue()


def build_text_edit_prompt(target_text: str, detected_text: str) -> str:
    return (
        'Edit only the masked text region. Replace "{}" with the exact text "{}". '
        "Preserve the original font style, size, color, perspective, material, background, "
        "layout, people, and all content outside the mask. Do not add any other text."
    ).format(detected_text, target_text)
