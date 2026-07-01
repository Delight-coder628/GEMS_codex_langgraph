from pathlib import Path

import pytest

from agent.server.z_image_npu import (
    parse_visible_devices,
    validate_model_path,
)


@pytest.mark.parametrize(
    "value, expected",
    [("0", [0]), ("3,6", [3, 6]), (" 7, 1 ", [7, 1])],
)
def test_parse_visible_devices(value, expected) -> None:
    assert parse_visible_devices(value) == expected


@pytest.mark.parametrize("value", ["", "0,1,2", "8", "1,1", "a"])
def test_parse_visible_devices_rejects_invalid_values(value) -> None:
    with pytest.raises(ValueError):
        parse_visible_devices(value)


def test_validate_model_path(tmp_path: Path) -> None:
    (tmp_path / "model_index.json").write_text("{}", encoding="utf-8")
    for name in ("transformer", "text_encoder", "vae"):
        (tmp_path / name).mkdir()
    assert validate_model_path(str(tmp_path)) == tmp_path.resolve()
