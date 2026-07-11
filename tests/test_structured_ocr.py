from langgraph_harness.ocr import (
    OCRLine,
    OCRResult,
    build_detected_instances,
    build_text_constraints,
    diagnose_ocr_result,
)


def _diagnose(targets, lines, editor_enabled=False):
    constraints = build_text_constraints(targets)
    instances = build_detected_instances(OCRResult(lines=lines))
    return diagnose_ocr_result(
        constraints,
        instances,
        min_confidence=0.5,
        normalized_match_threshold=0.8,
        editor_enabled=editor_enabled,
    )


def test_multiple_targets_match_distinct_instances() -> None:
    diagnosis = _diagnose(
        ["智启未来", "AI FOR INDUSTRY"],
        [
            OCRLine("智启未来", 0.98, [[10, 10], [100, 10], [100, 30], [10, 30]]),
            OCRLine(
                "AI FOR INDUSTRY",
                0.96,
                [[10, 50], [180, 50], [180, 70], [10, 70]],
            ),
        ],
    )
    assert diagnosis.passed is True
    assert len({item.detection_id for item in diagnosis.matches}) == 2
    assert diagnosis.recommended_action == "accept"


def test_single_mismatch_with_polygon_recommends_local_edit() -> None:
    diagnosis = _diagnose(
        ["AI AGENT"],
        [OCRLine("AI AGEMT", 0.98, [[1, 2], [20, 2], [20, 8], [1, 8]])],
        editor_enabled=True,
    )
    assert diagnosis.passed is False
    assert "content_mismatch" in diagnosis.error_types
    assert diagnosis.recommended_action == "local_edit"
    assert diagnosis.matches[0].character_differences


def test_missing_extra_low_confidence_and_order_errors() -> None:
    missing = _diagnose(["ONE", "TWO"], [OCRLine("ONE", 0.99)])
    assert "missing_text" in missing.error_types

    extra = _diagnose(
        ["ONE"], [OCRLine("ONE", 0.99), OCRLine("UNWANTED", 0.99)]
    )
    assert "extra_text" in extra.error_types

    low = _diagnose(["ONE"], [OCRLine("ONE", 0.2)])
    assert "low_confidence" in low.error_types

    reversed_lines = [
        OCRLine("ONE", 0.99, [[0, 50], [10, 50], [10, 60], [0, 60]]),
        OCRLine("TWO", 0.99, [[0, 10], [10, 10], [10, 20], [0, 20]]),
    ]
    order = _diagnose(["ONE", "TWO"], reversed_lines)
    assert "reading_order_error" in order.error_types
    assert order.recommended_action == "regenerate"
