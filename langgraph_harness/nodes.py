import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from agent.skill_manager import SkillManager
from langgraph_harness.config import AppConfig
from langgraph_harness.parsing import (
    parse_string_list,
    parse_verify_result,
)
from langgraph_harness.prompts import (
    DECOMPOSER_PROMPT,
    JSON_REPAIR_PROMPT,
    MEMORY_PROMPT,
    PLANNER_PROMPT,
    REFINER_PROMPT,
    SKILL_ROUTER_PROMPT,
    VERIFIER_PROMPT,
)
from langgraph_harness.run_logger import RunLogger
from langgraph_harness.schemas import AttemptRecord
from langgraph_harness.states import AgentState


def _dedupe(values: List[str]) -> List[str]:
    result = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


@dataclass
class NodeDependencies:
    config: AppConfig
    mllm: Any
    generator: Any
    logger: RunLogger
    skill_manager: SkillManager


class GEMSNodes:
    def __init__(self, dependencies: NodeDependencies):
        self.deps = dependencies

    def _log(self, message: str) -> None:
        if self.deps.config.agent.verbose:
            print(message)

    def _heuristic_skills(self, prompt: str) -> List[str]:
        lowered = prompt.lower()
        selected = []
        quoted_text = bool(re.search(r"[\"“”'][^\"“”']+[\"“”']", prompt))
        rules = {
            "hand_quality": (
                "hand",
                "hands",
                "finger",
                "fingers",
                "holding",
                "grasping",
                "手",
                "手指",
                "握",
            ),
            "text_rendering": (
                "text",
                "word",
                "letters",
                "poster",
                "logo",
                "sign",
                "文字",
                "海报",
                "标志",
                "写着",
            ),
            "spatial": (
                "left",
                "right",
                "above",
                "below",
                "behind",
                "front of",
                "左",
                "右",
                "上方",
                "下方",
                "后面",
                "前面",
            ),
            "aesthetic_drawing": (
                "masterpiece",
                "professional art",
                "award-winning",
                "高质量",
                "杰作",
                "专业艺术",
            ),
            "creative_drawing": (
                "creative",
                "dreamy",
                "surreal",
                "futuristic",
                "artistic",
                "创意",
                "梦幻",
                "超现实",
                "未来主义",
            ),
        }
        for skill_id, keywords in rules.items():
            if skill_id in self.deps.skill_manager.skills and any(
                keyword in lowered for keyword in keywords
            ):
                selected.append(skill_id)
        if quoted_text and "text_rendering" in self.deps.skill_manager.skills:
            selected.append("text_rendering")
        return _dedupe(selected)

    def skill_router(self, state: AgentState) -> Dict[str, Any]:
        fallback = self._heuristic_skills(state["original_prompt"])
        manifest = self.deps.skill_manager.get_skill_manifest()
        task = SKILL_ROUTER_PROMPT.format(
            manifest=manifest, user_prompt=state["original_prompt"]
        )
        warnings = list(state["logs"])
        try:
            selected = parse_string_list(self.deps.mllm.think(task))
            selected = [
                item
                for item in selected
                if item in self.deps.skill_manager.skills
            ]
            selected = _dedupe(fallback + selected)
        except Exception as exc:
            selected = fallback
            warnings.append("Skill router fallback: {}".format(exc))

        skills = [
            self.deps.skill_manager.skills[skill_id]
            for skill_id in selected
        ]
        instructions = "\n\n".join(
            "### {}\n{}".format(skill["id"], skill["instructions"])
            for skill in skills
        )
        self._log("[skill_router] {}".format(selected or ["NONE"]))
        return {
            "triggered_skills": skills,
            "skill_instructions": instructions,
            "logs": warnings,
        }

    def planner(self, state: AgentState) -> Dict[str, Any]:
        if not state["skill_instructions"]:
            return {
                "current_prompt": state["original_prompt"],
                "plan_text": state["original_prompt"],
            }
        task = PLANNER_PROMPT.format(
            original_prompt=state["original_prompt"],
            skill_instructions=state["skill_instructions"],
        )
        try:
            planned = self.deps.mllm.think(task).strip()
            if not planned:
                raise ValueError("Planner returned an empty prompt.")
            self._log("[planner] prompt enhanced")
            return {"current_prompt": planned, "plan_text": planned}
        except Exception as exc:
            return {
                "errors": state["errors"] + ["Planner error: {}".format(exc)]
            }

    def decomposer(self, state: AgentState) -> Dict[str, Any]:
        if state["errors"]:
            return {}
        task = DECOMPOSER_PROMPT.format(
            original_prompt=state["original_prompt"]
        )
        try:
            checks = parse_string_list(self.deps.mllm.think(task))
            selected = {item["id"] for item in state["triggered_skills"]}
            if "hand_quality" in selected:
                checks.extend(
                    [
                        "Are all visible hands anatomically plausible?",
                        "Are visible fingers separated without duplication or fusion?",
                    ]
                )
            if "text_rendering" in selected:
                checks.extend(
                    [
                        "Is the exact requested text present and correctly spelled?",
                        "Is the requested text readable and properly placed?",
                    ]
                )
            checks = _dedupe(checks)
            if not checks:
                checks = ["Does the image satisfy the user's complete request?"]
            self._log("[decomposer] {} checks".format(len(checks)))
            return {"atomic_checks": checks}
        except Exception as exc:
            return {
                "errors": state["errors"] + ["Decomposer error: {}".format(exc)]
            }

    def generator(self, state: AgentState) -> Dict[str, Any]:
        if state["errors"]:
            return {}
        iteration = state["iteration"] + 1
        cfg = self.deps.config.generator
        started = time.monotonic()
        try:
            content = self.deps.generator.generate(
                state["current_prompt"],
                seed=cfg.seed,
                width=cfg.width,
                height=cfg.height,
                num_inference_steps=cfg.num_inference_steps,
                guidance_scale=cfg.guidance_scale,
            )
            image_path = self.deps.logger.write_image(iteration, content)
            latency_ms = int((time.monotonic() - started) * 1000)
            self._log(
                "[generator] round {} saved {}".format(iteration, image_path)
            )
            return {
                "iteration": iteration,
                "image_path": image_path,
                "generation_latency_ms": latency_ms,
            }
        except Exception as exc:
            return {
                "iteration": iteration,
                "errors": state["errors"] + ["Generator error: {}".format(exc)],
            }

    def verifier(self, state: AgentState) -> Dict[str, Any]:
        if state["errors"]:
            return {}
        image_bytes = Path(state["image_path"]).read_bytes()
        checklist = "\n".join(
            "{}. {}".format(index, item)
            for index, item in enumerate(state["atomic_checks"], start=1)
        )
        task = VERIFIER_PROMPT.format(checklist=checklist)
        raw = ""
        last_error = None

        for repair_index in range(
            self.deps.config.agent.verifier_repair_attempts + 1
        ):
            try:
                if repair_index == 0:
                    raw = self.deps.mllm.think(task, images=[image_bytes])
                else:
                    repair_task = JSON_REPAIR_PROMPT.format(
                        raw_response=raw
                    )
                    raw = self.deps.mllm.think(repair_task)
                result = parse_verify_result(raw)
                break
            except Exception as exc:
                last_error = exc
        else:
            return {
                "errors": state["errors"]
                + ["Verifier parse error: {}".format(last_error)]
            }

        passed_checks = [
            check.question for check in result.checks if check.passed
        ]
        best_count = state["best_passed_count"]
        best_path = state["best_image_path"]
        if len(passed_checks) > best_count:
            best_count = len(passed_checks)
            best_path = state["image_path"]

        self._log(
            "[verifier] {}/{} passed".format(
                len(passed_checks), len(result.checks)
            )
        )
        return {
            "verify_result": result.model_dump(),
            "passed_checks": passed_checks,
            "failed_checks": result.failed_checks,
            "failure_tags": list(result.failure_tags),
            "suggested_fix": result.suggested_fix,
            "confidence": result.confidence,
            "best_passed_count": best_count,
            "best_image_path": best_path,
        }

    def memory_writer(self, state: AgentState) -> Dict[str, Any]:
        image_bytes = (
            Path(state["image_path"]).read_bytes()
            if state["image_path"] and Path(state["image_path"]).is_file()
            else None
        )
        summary = state["memory_summary"]
        if not state["errors"] and image_bytes is not None:
            task = MEMORY_PROMPT.format(
                current_prompt=state["current_prompt"],
                passed_checks=", ".join(state["passed_checks"]) or "None",
                failed_checks=", ".join(state["failed_checks"]) or "None",
                suggested_fix=state["suggested_fix"] or "None",
                previous_memory=state["memory_summary"] or "None",
            )
            try:
                summary = self.deps.mllm.think(
                    task, images=[image_bytes]
                ).strip()
            except Exception as exc:
                summary = (
                    state["memory_summary"]
                    or "Memory summary unavailable: {}".format(exc)
                )

        record = AttemptRecord(
            run_id=state["run_id"],
            iteration=state["iteration"],
            original_prompt=state["original_prompt"],
            current_prompt=state["current_prompt"],
            image_path=state["image_path"],
            passed_checks=state["passed_checks"],
            failed_checks=state["failed_checks"],
            failure_tags=state["failure_tags"],
            suggested_fix=state["suggested_fix"],
            confidence=state["confidence"],
            memory_summary=summary,
            latency_ms=state["generation_latency_ms"],
            errors=state["errors"],
        )
        self.deps.logger.write_attempt(record)
        history = state["attempt_history"] + [record.model_dump()]
        return {"attempt_history": history, "memory_summary": summary}

    def refiner(self, state: AgentState) -> Dict[str, Any]:
        if state["errors"]:
            return {}
        image_bytes = Path(state["image_path"]).read_bytes()
        task = REFINER_PROMPT.format(
            original_prompt=state["original_prompt"],
            current_prompt=state["current_prompt"],
            passed_checks=", ".join(state["passed_checks"]) or "None",
            failed_checks=", ".join(state["failed_checks"]) or "None",
            failure_tags=", ".join(state["failure_tags"]) or "None",
            suggested_fix=state["suggested_fix"] or "None",
            memory_summary=state["memory_summary"] or "None",
        )
        try:
            refined = self.deps.mllm.think(
                task, images=[image_bytes]
            ).strip()
            if not refined:
                raise ValueError("Refiner returned an empty prompt.")
            self._log("[refiner] next-round prompt ready")
            return {"current_prompt": refined}
        except Exception as exc:
            return {
                "errors": state["errors"] + ["Refiner error: {}".format(exc)]
            }

    def finalizer(self, state: AgentState) -> Dict[str, Any]:
        verify_passed = bool(state["verify_result"].get("passed"))
        if state["errors"]:
            status = "error"
        elif verify_passed:
            status = "success"
        else:
            status = "max_iter_reached"

        best_image = state["best_image_path"] or state["image_path"]
        final_image = state["image_path"] if verify_passed else best_image
        report = {
            "run_id": state["run_id"],
            "final_status": status,
            "final_image_path": final_image,
            "best_image_path": best_image,
            "best_passed_count": max(state["best_passed_count"], 0),
            "total_checks": len(state["atomic_checks"]),
            "iterations": state["iteration"],
            "common_failure_tags": RunLogger.common_tags(
                state["attempt_history"]
            ),
            "errors": state["errors"],
        }
        report_path = self.deps.logger.write_json(
            "final_report.json", report
        )
        self._log("[finalizer] {}".format(status))
        return {
            "final_status": status,
            "final_image_path": final_image,
            "final_report_path": report_path,
        }
