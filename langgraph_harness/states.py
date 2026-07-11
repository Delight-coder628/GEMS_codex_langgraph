from typing import Any, Dict, List

from typing_extensions import NotRequired, TypedDict


class AgentState(TypedDict):
    run_id: str
    task_id: str
    original_prompt: str
    current_prompt: str
    max_iterations: int
    iteration: int
    artifact_dir: str
    triggered_skills: List[Dict[str, str]]
    skill_instructions: str
    plan_text: str
    atomic_checks: List[str]
    image_path: str
    best_image_path: str
    best_passed_count: int
    verify_result: Dict[str, Any]
    passed_checks: List[str]
    failed_checks: List[str]
    failure_tags: List[str]
    suggested_fix: str
    confidence: float
    ocr_result: Dict[str, Any]
    ocr_score: Dict[str, Any]
    ocr_constraints: List[Dict[str, Any]]
    ocr_instances: List[Dict[str, Any]]
    ocr_diagnosis: Dict[str, Any]
    recommended_action: str
    actual_action: str
    edit_attempts: int
    editor_latency_ms: int
    pending_edit: Dict[str, Any]
    attempt_history: List[Dict[str, Any]]
    memory_summary: str
    logs: List[str]
    errors: List[str]
    generation_latency_ms: int
    final_status: NotRequired[str]
    final_image_path: NotRequired[str]
    final_report_path: NotRequired[str]
