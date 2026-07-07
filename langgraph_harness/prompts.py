SKILL_ROUTER_PROMPT = """You are a skill router for an image-generation agent.
Choose every skill that materially improves the request. Return ONLY a JSON array
of skill IDs. Return [] when no skill is useful.

Available skills:
{manifest}

User request:
{user_prompt}
"""

PLANNER_PROMPT = """Rewrite the image-generation request into one clear, complete
prompt for Z-Image-Turbo. Preserve the user's intent and exact requested text.
Apply the selected skill instructions without introducing conflicting details.
Never translate, paraphrase, autocorrect, abbreviate, or add extra words to quoted
target text requested by the user.
Return only the rewritten prompt.

Original request:
{original_prompt}

Skill instructions:
{skill_instructions}
"""

DECOMPOSER_PROMPT = """Break the image-generation request into independent visual
requirements. Express every requirement as a yes/no question that can be checked
from the final image. Return ONLY a JSON array of strings.

User request:
{original_prompt}
"""

VERIFIER_PROMPT = """You are a strict image-generation verifier.
Evaluate the supplied image against every checklist item. Return ONLY valid JSON.

Allowed failure_tags:
hand_error, text_render_error, spatial_error, attribute_binding_error,
style_error, counting_error, object_missing, object_extra, background_error,
low_visual_quality, unknown_error.

Required schema:
{{
  "passed": boolean,
  "checks": [
    {{
      "question": string,
      "passed": boolean,
      "evidence": string,
      "failure_tags": [string],
      "suggested_fix": string,
      "confidence": number between 0 and 1
    }}
  ],
  "failed_checks": [string],
  "failure_tags": [string],
  "suggested_fix": string,
  "confidence": number between 0 and 1
}}

Checklist:
{checklist}

Image: <image>
"""

JSON_REPAIR_PROMPT = """Repair the following verifier response so that it is valid
JSON matching the requested schema. Do not add commentary or Markdown fences.

Invalid response:
{raw_response}
"""

MEMORY_PROMPT = """Summarize this image-generation attempt in at most 100 words.
State what worked, what failed, and the most useful strategy for the next attempt.
Do not use an introduction.

Prompt: {current_prompt}
Passed checks: {passed_checks}
Failed checks: {failed_checks}
Suggested fix: {suggested_fix}
Previous memory: {previous_memory}
Image: <image>
"""

REFINER_PROMPT = """Create the next Z-Image-Turbo prompt from the failed attempt.
Return only the new prompt.

Original intent: {original_prompt}
Current prompt: {current_prompt}
Passed checks to preserve: {passed_checks}
Failed checks to fix: {failed_checks}
Failure tags: {failure_tags}
Suggested fix: {suggested_fix}
Experience memory: {memory_summary}

Requirements:
1. Explicitly address failed checks.
2. Preserve passed checks and the original intent.
3. Quote exact requested text.
4. When OCR or text rendering failed, make the text larger, sharper, higher
contrast, less distorted, and placed on a clean flat surface.
5. Never translate, paraphrase, autocorrect, abbreviate, or add extra words to
the requested text.
6. Avoid conflicting or conversational language.
Image from the failed attempt: <image>
"""
