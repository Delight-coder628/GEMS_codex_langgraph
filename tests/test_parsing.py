import pytest

from langgraph_harness.parsing import (
    parse_string_list,
    parse_verify_result,
)


def test_parse_string_list_accepts_fence_and_trailing_comma() -> None:
    assert parse_string_list('```json\n["one", "two",]\n```') == ["one", "two"]


def test_parse_verify_result_normalizes_summary_fields() -> None:
    result = parse_verify_result(
        """
        {
          "passed": true,
          "checks": [
            {
              "question": "Is there a cat?",
              "passed": false,
              "evidence": "No cat",
              "failure_tags": ["object_missing"],
              "suggested_fix": "Add a cat",
              "confidence": 0.9
            }
          ],
          "failed_checks": [],
          "failure_tags": [],
          "suggested_fix": "Add a cat",
          "confidence": 0.9
        }
        """
    )
    assert result.passed is False
    assert result.failed_checks == ["Is there a cat?"]
    assert result.failure_tags == ["object_missing"]


def test_parse_string_list_rejects_non_array() -> None:
    with pytest.raises(ValueError, match="Expected a JSON array"):
        parse_string_list('{"value": "not a list"}')
