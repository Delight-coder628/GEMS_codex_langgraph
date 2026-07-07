from langgraph_harness.ocr import (
    OCRLine,
    OCRResult,
    edit_distance,
    extract_target_texts,
    normalize_text,
    score_ocr_result,
)


def test_extract_target_texts_from_quotes_and_cues() -> None:
    assert extract_target_texts('A poster saying "AI AGENT WEEK 3"') == [
        "AI AGENT WEEK 3"
    ]
    assert extract_target_texts("一张写着「未来工厂」的海报") == ["未来工厂"]
    assert extract_target_texts("logo with the word GEMS on glass") == ["GEMS on glass"]


def test_normalize_text_removes_case_space_and_punctuation() -> None:
    assert normalize_text("AI Agent, Week 3!") == "aiagentweek3"
    assert normalize_text("“未来 工厂！”") == "未来工厂"


def test_edit_distance() -> None:
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("same", "same") == 0
    assert edit_distance("", "abc") == 3


def test_score_ocr_result_passes_exact_match() -> None:
    result = OCRResult(lines=[OCRLine(text="AI AGENT WEEK 3", confidence=0.92)])

    score = score_ocr_result(
        "AI AGENT WEEK 3",
        result,
        min_confidence=0.5,
        normalized_match_threshold=0.8,
    )

    assert score.passed is True
    assert score.exact_match is True
    assert score.failure_reason == ""


def test_score_ocr_result_fails_mismatch() -> None:
    result = OCRResult(lines=[OCRLine(text="AI AGENT WEEK B", confidence=0.92)])

    score = score_ocr_result(
        "AI AGENT WEEK 3",
        result,
        min_confidence=0.5,
        normalized_match_threshold=0.95,
    )

    assert score.passed is False
    assert score.failure_reason == "ocr_text_mismatch"
    assert score.suggested_fix
