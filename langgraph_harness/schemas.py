from typing import List, Literal

from pydantic import BaseModel, Field


FailureTag = Literal[
    "hand_error",
    "text_render_error",
    "spatial_error",
    "attribute_binding_error",
    "style_error",
    "counting_error",
    "object_missing",
    "object_extra",
    "background_error",
    "low_visual_quality",
    "unknown_error",
]


class SkillInfo(BaseModel):
    id: str
    description: str = ""
    instructions: str = ""


class VisualCheck(BaseModel):
    question: str
    passed: bool
    evidence: str = ""
    failure_tags: List[FailureTag] = Field(default_factory=list)
    suggested_fix: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class VerifyResult(BaseModel):
    passed: bool
    checks: List[VisualCheck] = Field(default_factory=list)
    failed_checks: List[str] = Field(default_factory=list)
    failure_tags: List[FailureTag] = Field(default_factory=list)
    suggested_fix: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class AttemptRecord(BaseModel):
    run_id: str
    iteration: int
    original_prompt: str
    current_prompt: str
    image_path: str = ""
    passed_checks: List[str] = Field(default_factory=list)
    failed_checks: List[str] = Field(default_factory=list)
    failure_tags: List[FailureTag] = Field(default_factory=list)
    suggested_fix: str = ""
    confidence: float = 0.0
    ocr_score: dict = Field(default_factory=dict)
    ocr_diagnosis: dict = Field(default_factory=dict)
    recommended_action: str = ""
    actual_action: str = ""
    edit_attempts: int = 0
    editor_latency_ms: int = 0
    memory_summary: str = ""
    latency_ms: int = 0
    errors: List[str] = Field(default_factory=list)


class FinalReport(BaseModel):
    run_id: str
    final_status: Literal["success", "max_iter_reached", "error"]
    final_image_path: str = ""
    best_image_path: str = ""
    best_passed_count: int = 0
    total_checks: int = 0
    iterations: int = 0
    common_failure_tags: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
