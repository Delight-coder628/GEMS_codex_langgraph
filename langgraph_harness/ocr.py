import re
import string
from difflib import SequenceMatcher
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


@dataclass
class TextConstraint:
    id: str
    text: str
    language: str
    reading_order: int
    expected_region: Optional[List[float]] = None
    font_family_class: Optional[str] = None
    color: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "language": self.language,
            "reading_order": self.reading_order,
            "expected_region": self.expected_region,
            "font_family_class": self.font_family_class,
            "color": self.color,
        }


@dataclass
class DetectedTextInstance:
    id: str
    text: str
    normalized_text: str
    confidence: float
    polygon: Optional[List[Any]]
    reading_order: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "normalized_text": self.normalized_text,
            "confidence": self.confidence,
            "polygon": self.polygon,
            "reading_order": self.reading_order,
        }


@dataclass
class ConstraintMatch:
    constraint_id: str
    detection_id: str
    target_text: str
    detected_text: str
    edit_distance: int
    similarity: float
    character_differences: List[Dict[str, str]]
    passed: bool
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "detection_id": self.detection_id,
            "target_text": self.target_text,
            "detected_text": self.detected_text,
            "edit_distance": self.edit_distance,
            "similarity": self.similarity,
            "character_differences": self.character_differences,
            "passed": self.passed,
            "confidence": self.confidence,
        }


@dataclass
class OCRDiagnosis:
    matches: List[ConstraintMatch]
    missing_constraint_ids: List[str]
    extra_detection_ids: List[str]
    error_types: List[str]
    passed: bool
    suggested_fix: str
    recommended_action: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matches": [item.to_dict() for item in self.matches],
            "missing_constraint_ids": self.missing_constraint_ids,
            "extra_detection_ids": self.extra_detection_ids,
            "error_types": self.error_types,
            "passed": self.passed,
            "suggested_fix": self.suggested_fix,
            "recommended_action": self.recommended_action,
            "passed_constraint_count": sum(item.passed for item in self.matches),
            "total_constraint_count": len(self.matches)
            + len(self.missing_constraint_ids),
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


def build_text_constraints(targets: Sequence[str]) -> List[TextConstraint]:
    return [
        TextConstraint(
            id="text_{:02d}".format(index),
            text=target,
            language=detect_language(target),
            reading_order=index - 1,
        )
        for index, target in enumerate(targets, start=1)
    ]


def detect_language(value: str) -> str:
    has_zh = bool(re.search(r"[\u3400-\u9fff]", value))
    has_en = bool(re.search(r"[A-Za-z]", value))
    if has_zh and has_en:
        return "mixed"
    if has_zh:
        return "zh"
    return "en"


def build_detected_instances(ocr_result: OCRResult) -> List[DetectedTextInstance]:
    indexed = list(enumerate(ocr_result.lines))
    indexed.sort(key=lambda item: _reading_order_key(item[1], item[0]))
    return [
        DetectedTextInstance(
            id="det_{:02d}".format(index),
            text=line.text,
            normalized_text=normalize_text(line.text),
            confidence=line.confidence,
            polygon=line.bbox,
            reading_order=index - 1,
        )
        for index, (_, line) in enumerate(indexed, start=1)
        if line.text
    ]


def diagnose_ocr_result(
    constraints: Sequence[TextConstraint],
    instances: Sequence[DetectedTextInstance],
    min_confidence: float,
    normalized_match_threshold: float,
    editor_enabled: bool = False,
) -> OCRDiagnosis:
    candidates = []
    for constraint in constraints:
        for instance in instances:
            similarity = _text_similarity(constraint.text, instance.text)
            candidates.append((similarity, constraint.id, instance.id))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    constraint_by_id = {item.id: item for item in constraints}
    instance_by_id = {item.id: item for item in instances}
    used_constraints = set()
    used_instances = set()
    pairs = []
    for similarity, constraint_id, instance_id in candidates:
        if constraint_id in used_constraints or instance_id in used_instances:
            continue
        used_constraints.add(constraint_id)
        used_instances.add(instance_id)
        pairs.append((constraint_id, instance_id, similarity))

    matches = []
    for constraint_id, instance_id, similarity in pairs:
        constraint = constraint_by_id[constraint_id]
        instance = instance_by_id[instance_id]
        normalized_target = normalize_text(constraint.text)
        normalized_detected = instance.normalized_text
        distance = edit_distance(normalized_target, normalized_detected)
        matches.append(
            ConstraintMatch(
                constraint_id=constraint_id,
                detection_id=instance_id,
                target_text=constraint.text,
                detected_text=instance.text,
                edit_distance=distance,
                similarity=similarity,
                character_differences=_character_differences(
                    normalized_target, normalized_detected
                ),
                passed=(
                    normalized_target == normalized_detected
                    and instance.confidence >= min_confidence
                ),
                confidence=instance.confidence,
            )
        )
    matches.sort(key=lambda item: constraint_by_id[item.constraint_id].reading_order)

    missing = [item.id for item in constraints if item.id not in used_constraints]
    extra = [item.id for item in instances if item.id not in used_instances]
    error_types = []
    if missing:
        error_types.append("missing_text")
    if extra:
        error_types.append("extra_text")
    if any(item.character_differences for item in matches):
        error_types.append("content_mismatch")
    if any(item.confidence < min_confidence for item in matches):
        error_types.append("low_confidence")
    matched_orders = [
        instance_by_id[item.detection_id].reading_order for item in matches
    ]
    if matched_orders != sorted(matched_orders):
        error_types.append("reading_order_error")

    passed = bool(constraints) and not error_types and all(item.passed for item in matches)
    suggested_fix = _diagnosis_fix(
        constraints, instance_by_id, matches, missing, extra, error_types
    )
    action = _recommend_action(
        matches, missing, error_types, instance_by_id, editor_enabled
    )
    return OCRDiagnosis(
        matches=matches,
        missing_constraint_ids=missing,
        extra_detection_ids=extra,
        error_types=error_types,
        passed=passed,
        suggested_fix=suggested_fix,
        recommended_action=action,
    )


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


def _reading_order_key(line: OCRLine, fallback: int) -> Any:
    polygon = line.bbox
    if not isinstance(polygon, list) or not polygon:
        return (1, fallback, fallback)
    points = [
        point
        for point in polygon
        if isinstance(point, (list, tuple)) and len(point) >= 2
    ]
    if not points:
        return (1, fallback, fallback)
    center_x = sum(_safe_float(point[0]) for point in points) / len(points)
    center_y = sum(_safe_float(point[1]) for point in points) / len(points)
    return (0, round(center_y / 10.0), center_x)


def _text_similarity(left: str, right: str) -> float:
    normalized_left = normalize_text(left)
    normalized_right = normalize_text(right)
    distance = edit_distance(normalized_left, normalized_right)
    max_len = max(len(normalized_left), len(normalized_right), 1)
    return max(0.0, min(1.0, 1.0 - distance / max_len))


def _character_differences(left: str, right: str) -> List[Dict[str, str]]:
    differences = []
    matcher = SequenceMatcher(a=left, b=right, autojunk=False)
    for tag, left_start, left_end, right_start, right_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        differences.append(
            {
                "operation": tag,
                "expected": left[left_start:left_end],
                "detected": right[right_start:right_end],
            }
        )
    return differences


def _diagnosis_fix(
    constraints: Sequence[TextConstraint],
    instance_by_id: Dict[str, DetectedTextInstance],
    matches: Sequence[ConstraintMatch],
    missing: Sequence[str],
    extra: Sequence[str],
    error_types: Sequence[str],
) -> str:
    if not error_types:
        return ""
    constraint_by_id = {item.id: item for item in constraints}
    parts = []
    if missing:
        texts = [constraint_by_id[item].text for item in missing]
        parts.append("Render the missing exact text: {}.".format(", ".join(texts)))
    mismatches = [item for item in matches if item.character_differences]
    for item in mismatches:
        parts.append(
            'Replace OCR text "{}" with the exact text "{}".'.format(
                item.detected_text, item.target_text
            )
        )
    if extra:
        texts = [instance_by_id[item].text for item in extra]
        parts.append("Remove unintended text: {}.".format(", ".join(texts)))
    if "low_confidence" in error_types:
        parts.append("Make the requested text larger, sharper, and higher contrast.")
    if "reading_order_error" in error_types:
        parts.append("Preserve the requested line and reading order.")
    return " ".join(parts)


def _recommend_action(
    matches: Sequence[ConstraintMatch],
    missing: Sequence[str],
    error_types: Sequence[str],
    instance_by_id: Dict[str, DetectedTextInstance],
    editor_enabled: bool,
) -> str:
    if not error_types:
        return "accept"
    if "reading_order_error" in error_types or len(missing) > 1:
        return "regenerate"
    mismatches = [
        item
        for item in matches
        if "content_mismatch" in error_types and item.character_differences
    ]
    if len(mismatches) == 1:
        instance = instance_by_id[mismatches[0].detection_id]
        if editor_enabled and instance.polygon:
            return "local_edit"
    return "refine_prompt"
