import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import requests


_QUOTED_PATTERNS = [
    re.compile(r'"([^"]+)"'),
    re.compile(r"'([^']+)'"),
    re.compile(r"“([^”]+)”"),
    re.compile(r"‘([^’]+)’"),
    re.compile(r"「([^」]+)」"),
    re.compile(r"『([^』]+)』"),
]

_TEXT_CUE_PATTERNS = [
    re.compile(r"(?:写着|文字为|标题为|标语为|标志为)\s*[:：]?\s*([^，。；;,.!?！？\n]+)"),
    re.compile(
        r"(?:saying|reads|read|with the words?|with the text|logo with the word)\s+"
        r"([^,.!?;\n]+)",
        re.IGNORECASE,
    ),
]


@dataclass
class OCRLine:
    text: str
    confidence: float = 0.0
    bbox: Optional[List[Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bbox": self.bbox,
        }


@dataclass
class OCRResult:
    lines: List[OCRLine] = field(default_factory=list)

    @property
    def merged_text(self) -> str:
        return " ".join(line.text for line in self.lines if line.text).strip()

    @property
    def confidence(self) -> float:
        scores = [line.confidence for line in self.lines if line.confidence > 0]
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    @property
    def normalized_text(self) -> str:
        return normalize_text(self.merged_text)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lines": [line.to_dict() for line in self.lines],
            "merged_text": self.merged_text,
            "normalized_text": self.normalized_text,
            "confidence": self.confidence,
        }


@dataclass
class OCRScore:
    target_text: str
    recognized_text: str
    normalized_target: str
    normalized_recognized: str
    exact_match: bool
    substring_match: bool
    edit_distance: int
    similarity: float
    confidence: float
    passed: bool
    failure_reason: str = ""
    suggested_fix: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_text": self.target_text,
            "recognized_text": self.recognized_text,
            "normalized_target": self.normalized_target,
            "normalized_recognized": self.normalized_recognized,
            "exact_match": self.exact_match,
            "substring_match": self.substring_match,
            "edit_distance": self.edit_distance,
            "similarity": self.similarity,
            "confidence": self.confidence,
            "passed": self.passed,
            "failure_reason": self.failure_reason,
            "suggested_fix": self.suggested_fix,
        }


def extract_target_texts(prompt: str) -> List[str]:
    targets: List[str] = []
    for pattern in _QUOTED_PATTERNS:
        targets.extend(match.group(1).strip() for match in pattern.finditer(prompt))
    if targets:
        return _dedupe([target for target in targets if target])
    for pattern in _TEXT_CUE_PATTERNS:
        targets.extend(
            _clean_target(match.group(1)) for match in pattern.finditer(prompt)
        )
    return _dedupe([target for target in targets if target])


def normalize_text(value: str) -> str:
    lowered = value.casefold()
    punctuation = string.punctuation + "，。！？；：“”‘’「」『』、·《》（）【】"
    table = str.maketrans("", "", punctuation)
    without_punctuation = lowered.translate(table)
    return re.sub(r"\s+", "", without_punctuation)


def edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def score_ocr_result(
    target_text: str,
    ocr_result: OCRResult,
    min_confidence: float,
    normalized_match_threshold: float,
) -> OCRScore:
    normalized_target = normalize_text(target_text)
    normalized_recognized = ocr_result.normalized_text
    distance = edit_distance(normalized_target, normalized_recognized)
    max_len = max(len(normalized_target), len(normalized_recognized), 1)
    similarity = 1.0 - (distance / max_len)
    exact_match = normalized_target == normalized_recognized
    substring_match = bool(
        normalized_target
        and normalized_recognized
        and normalized_target in normalized_recognized
    )
    confidence = ocr_result.confidence
    text_passed = exact_match or substring_match or similarity >= normalized_match_threshold
    confidence_passed = confidence >= min_confidence
    passed = text_passed and confidence_passed

    failure_reason = ""
    suggested_fix = ""
    if not normalized_recognized:
        failure_reason = "ocr_found_no_text"
        suggested_fix = (
            "Render the exact requested text larger, with higher contrast, on a clean flat surface."
        )
    elif not text_passed:
        failure_reason = "ocr_text_mismatch"
        suggested_fix = (
            'Render the exact text "{}" without paraphrasing, missing letters, or extra words.'
        ).format(target_text)
    elif not confidence_passed:
        failure_reason = "ocr_low_confidence"
        suggested_fix = (
            "Make the text sharper, less distorted, and separated from background clutter."
        )

    return OCRScore(
        target_text=target_text,
        recognized_text=ocr_result.merged_text,
        normalized_target=normalized_target,
        normalized_recognized=normalized_recognized,
        exact_match=exact_match,
        substring_match=substring_match,
        edit_distance=distance,
        similarity=max(0.0, min(1.0, similarity)),
        confidence=confidence,
        passed=passed,
        failure_reason=failure_reason,
        suggested_fix=suggested_fix,
    )


class PaddleOCRClient:
    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        self._ocr = None

    @property
    def ocr(self) -> Any:
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise RuntimeError(
                    "PaddleOCR is not installed. Install requirements-ocr.txt "
                    "or switch ocr.backend to an internal http service."
                ) from exc
            self._ocr = PaddleOCR(**self.kwargs)
        return self._ocr

    def recognize(self, image_path: str) -> OCRResult:
        raw = self.ocr.ocr(image_path, cls=True)
        return parse_paddleocr_result(raw)


class HttpOCRClient:
    def __init__(self, url: str, timeout: float = 60.0):
        self.url = url
        self.timeout = timeout

    def recognize(self, image_path: str) -> OCRResult:
        if not self.url:
            raise ValueError("ocr.url is required when ocr.backend is http.")
        with Path(image_path).open("rb") as handle:
            response = requests.post(
                self.url,
                files={"image": ("image.png", handle, "image/png")},
                timeout=self.timeout,
            )
        response.raise_for_status()
        return parse_http_ocr_result(response.json())


class MockOCRClient:
    def __init__(self, text: str = "", confidence: float = 1.0):
        self.text = text
        self.confidence = confidence

    def recognize(self, image_path: str) -> OCRResult:
        if not self.text:
            return OCRResult(lines=[])
        return OCRResult(lines=[OCRLine(text=self.text, confidence=self.confidence)])


def parse_paddleocr_result(raw: Any) -> OCRResult:
    lines: List[OCRLine] = []
    for item in _flatten_paddle_lines(raw):
        bbox = item[0] if len(item) > 0 else None
        payload = item[1] if len(item) > 1 else None
        if isinstance(payload, (list, tuple)) and payload:
            text = str(payload[0])
            confidence = _safe_float(payload[1] if len(payload) > 1 else 0.0)
            lines.append(OCRLine(text=text, confidence=confidence, bbox=bbox))
    return OCRResult(lines=lines)


def parse_http_ocr_result(raw: Dict[str, Any]) -> OCRResult:
    raw_lines = raw.get("lines", [])
    lines = []
    for item in raw_lines:
        if isinstance(item, dict):
            lines.append(
                OCRLine(
                    text=str(item.get("text", "")),
                    confidence=_safe_float(item.get("confidence", 0.0)),
                    bbox=item.get("bbox"),
                )
            )
    if not lines and raw.get("text"):
        lines.append(
            OCRLine(
                text=str(raw.get("text", "")),
                confidence=_safe_float(raw.get("confidence", 0.0)),
            )
        )
    return OCRResult(lines=lines)


def _flatten_paddle_lines(raw: Any) -> List[Any]:
    if raw is None:
        return []
    if isinstance(raw, list):
        if raw and _looks_like_paddle_line(raw[0]):
            return raw
        flattened: List[Any] = []
        for item in raw:
            flattened.extend(_flatten_paddle_lines(item))
        return flattened
    return []


def _looks_like_paddle_line(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and isinstance(value[1], (list, tuple))
    )


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _dedupe(values: Sequence[str]) -> List[str]:
    result: List[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _clean_target(value: str) -> str:
    return value.strip(" ：:，。,.!?！？;；\"'“”‘’「」『』")
