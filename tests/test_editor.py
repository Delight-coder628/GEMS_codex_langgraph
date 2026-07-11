from io import BytesIO
from pathlib import Path

import pytest
Image = pytest.importorskip("PIL.Image")

from langgraph_harness.editor import EditorClient, build_polygon_mask


def _png(width=20, height=10) -> bytes:
    output = BytesIO()
    Image.new("RGB", (width, height), "white").save(output, format="PNG")
    return output.getvalue()


def test_polygon_mask_has_image_size_and_clipped_padding(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(_png())
    mask_bytes = build_polygon_mask(
        str(image_path), [[-2, 1], [8, 1], [8, 6], [-2, 6]], padding_ratio=0.5
    )
    mask = Image.open(BytesIO(mask_bytes))
    assert mask.size == (20, 10)
    assert mask.getpixel((0, 1)) == 255
    assert mask.getpixel((19, 9)) == 0


def test_editor_client_sends_multipart_and_accepts_png(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(_png())
    captured = {}

    class Response:
        headers = {"content-type": "image/png"}
        content = _png()

        def raise_for_status(self):
            return None

    def fake_post(url, files, data, timeout):
        captured.update(url=url, files=files, data=data, timeout=timeout)
        return Response()

    monkeypatch.setattr("langgraph_harness.editor.requests.post", fake_post)
    result = EditorClient("http://editor/edit").edit(
        str(image_path), "replace text", mask_bytes=_png(), seed=7
    )
    assert result.startswith(b"\x89PNG")
    assert set(captured["files"]) == {"image", "mask"}
    assert captured["data"]["seed"] == "7"


def test_editor_client_rejects_non_png(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(_png())

    class Response:
        headers = {"content-type": "application/json"}
        content = b"{}"

        def raise_for_status(self):
            return None

    monkeypatch.setattr(
        "langgraph_harness.editor.requests.post", lambda *args, **kwargs: Response()
    )
    with pytest.raises(RuntimeError, match="non-PNG"):
        EditorClient("http://editor/edit", max_retries=0).edit(
            str(image_path), "replace text"
        )
