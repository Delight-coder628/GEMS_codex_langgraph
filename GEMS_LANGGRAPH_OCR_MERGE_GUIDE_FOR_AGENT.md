# GEMS + LangGraph OCR Merge Guide for Internal Agent

This document explains the OCR/text-rendering changes added on top of the
existing GEMS + LangGraph harness. It is intended for an internal coding agent
that may use Qwen3.6, GLM4.7, or a similar model, so the key integration points
are spelled out explicitly.

## 1. Goal

Add an objective text-rendering verification loop:

```text
generator -> MLLM verifier -> OCR verifier -> memory_writer -> retry/finalize
```

The original GEMS baseline remains untouched. The OCR path is only wired into
`langgraph_harness`.

The first target is text rendering, not hand quality. OCR gives measurable
signals: target text, recognized text, exact match, normalized match, edit
distance, confidence, failure reason, and retry behavior.

## 2. Files Changed or Added

- `langgraph_harness/ocr.py`
  - New module for OCR clients, target-text extraction, normalization, edit
    distance, and OCR scoring.
  - Contains `PaddleOCRClient`, `HttpOCRClient`, and `MockOCRClient`.
  - `HttpOCRClient` is only for an internal OCR sidecar, not an external API.

- `langgraph_harness/config.py`
  - Adds `OCRConfig` under `AppConfig.ocr`.
  - Supported backends: `paddle_local` and `http`.
  - Validates that HTTP OCR has an internal URL when enabled.

- `configs/langgraph_gems.yaml`
  - Adds the `ocr` section.
  - OCR is disabled by default to preserve existing behavior.

- `langgraph_harness/nodes.py`
  - Adds `ocr_verifier`.
  - OCR runs only when all are true:
    - `ocr.enabled` is true.
    - `text_rendering` skill was triggered.
    - a target text can be extracted from the original or current prompt.
  - Failed OCR appends `text_render_error` and a concrete prompt fix.

- `langgraph_harness/graph.py`
  - Changes the flow from `verifier -> memory_writer` to
    `verifier -> ocr_verifier -> memory_writer`.

- `run_langgraph_gems.py`
  - Builds the OCR client from config.
  - Uses `MockOCRClient` in mock mode.

- `agent/skills/text_rendering/SKILL.md`
  - Adds exact text preservation rules.

- `requirements-ocr.txt`
  - Optional OCR dependencies.
  - Do not merge these into the main requirements unless OCR is mandatory.

- `tests/test_ocr.py`, `tests/test_mock_e2e.py`
  - Cover extraction, normalization, scoring, and OCR pass/fail graph behavior.

## 3. Configuration

Default config:

```yaml
ocr:
  enabled: false
  backend: "paddle_local"
  url: "${OCR_URL}"
  timeout_seconds: 60
  min_confidence: 0.5
  normalized_match_threshold: 0.8
```

For local PaddleOCR:

```yaml
ocr:
  enabled: true
  backend: "paddle_local"
  url: ""
  timeout_seconds: 60
  min_confidence: 0.5
  normalized_match_threshold: 0.8
```

For an internal OCR sidecar:

```yaml
ocr:
  enabled: true
  backend: "http"
  url: "http://127.0.0.1:8002/ocr"
```

Important: `http` means intranet or localhost only. The company environment is
assumed to have no external API access.

## 4. OCR Backend Contract

### Local PaddleOCR

`PaddleOCRClient` lazy-imports `paddleocr`. If OCR is disabled, PaddleOCR is not
required. If OCR is enabled and PaddleOCR is missing, the run fails with a clear
message.

The current constructor uses:

```python
PaddleOCRClient(use_angle_cls=True, lang="ch")
```

This is a reasonable default for mixed Chinese/English prompts. If the server
uses a newer PaddleOCR version with changed arguments, update only
`make_ocr_client` in `run_langgraph_gems.py`.

### Internal HTTP OCR

`HttpOCRClient` sends multipart form data:

```text
POST /ocr
file field: image
```

Expected response can be either:

```json
{
  "lines": [
    {"text": "AI AGENT WEEK 3", "confidence": 0.98, "bbox": [[0,0],[1,0],[1,1],[0,1]]}
  ]
}
```

or:

```json
{"text": "AI AGENT WEEK 3", "confidence": 0.98}
```

## 5. How OCR Affects Retry

The MLLM verifier still judges the whole image. OCR only adds an extra check for
text tasks.

If OCR passes:

- The OCR check is appended to `passed_checks`.
- `verify_result.passed` remains whatever the MLLM verifier produced.

If OCR fails:

- The OCR check is appended to `failed_checks`.
- `text_render_error` is appended to `failure_tags`.
- `verify_result.passed` becomes false.
- `suggested_fix` is extended with OCR-specific guidance.
- The graph retries through the existing `refiner` path unless max iterations
  are reached.

Each OCR round writes:

```text
outputs/langgraph_runs/<run_id>/ocr_round_01.json
```

The final report also includes the last `ocr_score`.

## 6. Internal Deployment Notes

The company intranet should not call external OCR APIs.

Recommended first deployment:

1. Keep Z-Image-Turbo on Ascend 910B as currently configured.
2. Run OCR locally on CPU with PaddleOCR, or start an internal OCR sidecar.
3. Download PaddleOCR wheels and model files in an internet-enabled environment.
4. Upload the wheels/model cache to the server.
5. Install OCR dependencies from local wheels if the server cannot reach PyPI.

Do not spend the first week moving OCR to NPU unless OCR latency is proven to be
the bottleneck. The OCR model is much lighter than Z-Image-Turbo, and preserving
NPU memory for generation is usually more valuable.

## 7. Tests

Run the normal suite:

```bash
pytest -q
```

Useful targeted tests:

```bash
pytest -q tests/test_ocr.py
pytest -q tests/test_mock_e2e.py
```

Manual smoke test after OCR dependencies are available:

```bash
python run_langgraph_gems.py \
  --config configs/langgraph_gems.yaml \
  --prompt "一张写有 \"AI AGENT WEEK 3\" 的未来主义海报"
```

Check:

- `round_01.png`
- `ocr_round_01.json`
- `attempts.jsonl`
- `final_report.json`

## 8. Notes for Weaker Internal Agent Models

Qwen3.6 or GLM4.7 may be less stable at JSON formatting and exact instruction
following. Keep these rules:

- Do not remove `JSON_REPAIR_PROMPT`; it protects verifier parsing.
- Do not let planner/refiner translate or paraphrase quoted text.
- Prefer deterministic OCR scoring over asking the MLLM whether text is correct.
- Keep OCR disabled by default so unrelated image tasks stay stable.
- Preserve the original GEMS baseline entrypoints.
- If integration conflicts appear, keep `langgraph_harness/ocr.py` independent
  and only adjust the graph wiring.

## 9. Common Failure Modes

- PaddleOCR import error:
  - Install `requirements-ocr.txt` or switch to internal HTTP OCR.

- OCR finds no text:
  - Refiner should request larger, sharper, high-contrast text on a clean flat
    surface.

- OCR text mismatch:
  - Refiner should repeat the exact quoted text and avoid extra words.

- HTTP OCR times out:
  - Confirm the URL is intranet/local, not external.
  - Increase `ocr.timeout_seconds`.

- MLLM says the image passes but OCR fails:
  - OCR should win for exact text rendering because it is the objective signal.
