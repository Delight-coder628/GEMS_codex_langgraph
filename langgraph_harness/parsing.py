import ast
import json
import re
from typing import Any, List

from langgraph_harness.schemas import VerifyResult


def _candidate_text(text: str) -> str:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    starts = [index for index in (text.find("{"), text.find("[")) if index >= 0]
    if starts:
        start = min(starts)
        opening = text[start]
        closing = "}" if opening == "{" else "]"
        end = text.rfind(closing)
        if end >= start:
            text = text[start : end + 1]
    return re.sub(r",\s*([}\]])", r"\1", text)


def parse_json_value(text: str) -> Any:
    candidate = _candidate_text(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        try:
            value = ast.literal_eval(candidate)
        except (ValueError, SyntaxError) as exc:
            raise ValueError("Unable to parse JSON response: {}".format(exc)) from exc
        if not isinstance(value, (dict, list)):
            raise ValueError("Parsed response must be an object or array.")
        return value


def parse_string_list(text: str) -> List[str]:
    value = parse_json_value(text)
    if not isinstance(value, list):
        raise ValueError("Expected a JSON array.")
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def parse_verify_result(text: str) -> VerifyResult:
    value = parse_json_value(text)
    if not isinstance(value, dict):
        raise ValueError("Verifier response must be a JSON object.")
    result = VerifyResult.model_validate(value)

    failed = [check.question for check in result.checks if not check.passed]
    tags = []
    for check in result.checks:
        for tag in check.failure_tags:
            if tag not in tags:
                tags.append(tag)
    for tag in result.failure_tags:
        if tag not in tags:
            tags.append(tag)

    result.failed_checks = failed
    result.failure_tags = tags
    result.passed = bool(result.checks) and not failed
    return result
