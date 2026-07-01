"""FastAPI server for local Z-Image-Turbo weights on one or two Ascend NPUs.

Torch, torch_npu and Diffusers are imported inside spawned workers so that
ASCEND_RT_VISIBLE_DEVICES can be set before the NPU runtime is initialized.
"""

import argparse
import asyncio
import io
import os
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    seed: Optional[int] = Field(default=None, ge=0)
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)
    num_inference_steps: int = Field(default=9, ge=1, le=100)
    guidance_scale: float = Field(default=0.0, ge=0.0, le=20.0)


class ServerState:
    def __init__(self) -> None:
        self.context = get_context("spawn")
        self.task_queue = None
        self.result_queue = None
        self.workers = []
        self.worker_status: Dict[int, Dict[str, Any]] = {}
        self.pending: Dict[str, threading.Event] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.collector = None
        self.stop_event = threading.Event()

    @property
    def ready(self) -> bool:
        return bool(self.worker_status) and all(
            item.get("status") == "ready"
            for item in self.worker_status.values()
        )


STATE = ServerState()


def parse_visible_devices(value: str) -> List[int]:
    if not value or not value.strip():
        raise ValueError(
            "ASCEND_RT_VISIBLE_DEVICES is required; provide one or two IDs from 0 to 7."
        )
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if len(parts) not in (1, 2):
        raise ValueError("Exactly one or two NPU IDs must be configured.")
    if any(not item.isdigit() for item in parts):
        raise ValueError("NPU IDs must be integers from 0 to 7.")
    devices = [int(item) for item in parts]
    if any(item < 0 or item > 7 for item in devices):
        raise ValueError("NPU IDs must be in the range 0-7.")
    if len(set(devices)) != len(devices):
        raise ValueError("NPU IDs must be unique.")
    return devices


def validate_model_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise ValueError("ZIMAGE_MODEL_PATH is not a directory: {}".format(path))
    required = ["model_index.json", "transformer", "text_encoder", "vae"]
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise ValueError(
            "Z-Image-Turbo snapshot is missing: {}".format(", ".join(missing))
        )
    return path


def _worker(
    logical_device: int,
    physical_device: int,
    model_path: str,
    task_queue: Any,
    result_queue: Any,
) -> None:
    try:
        import torch
        import torch_npu  # noqa: F401
        from diffusers import ZImagePipeline

        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise RuntimeError("torch_npu is installed but torch.npu is unavailable.")

        device = "npu:{}".format(logical_device)
        torch.npu.set_device(device)
        pipe = ZImagePipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            local_files_only=True,
        )
        pipe.to(device)
        result_queue.put(
            {
                "type": "worker_status",
                "logical_device": logical_device,
                "physical_device": physical_device,
                "status": "ready",
            }
        )
    except Exception as exc:
        result_queue.put(
            {
                "type": "worker_status",
                "logical_device": logical_device,
                "physical_device": physical_device,
                "status": "error",
                "error": repr(exc),
            }
        )
        return

    while True:
        task = task_queue.get()
        if task is None:
            return
        task_id = task["task_id"]
        try:
            seed = task.get("seed")
            if seed is None:
                seed = int.from_bytes(os.urandom(4), "big")
            # A CPU generator is accepted by Diffusers' randn_tensor helper and
            # avoids relying on backend-specific torch.Generator("npu") support.
            generator = torch.Generator(device="cpu").manual_seed(seed)
            with torch.inference_mode():
                image = pipe(
                    prompt=task["prompt"],
                    width=task["width"],
                    height=task["height"],
                    num_inference_steps=task["num_inference_steps"],
                    guidance_scale=task["guidance_scale"],
                    generator=generator,
                ).images[0]
            output = io.BytesIO()
            image.save(output, format="PNG")
            result_queue.put(
                {
                    "type": "result",
                    "task_id": task_id,
                    "ok": True,
                    "content": output.getvalue(),
                    "logical_device": logical_device,
                    "physical_device": physical_device,
                    "seed": seed,
                }
            )
        except Exception as exc:
            result_queue.put(
                {
                    "type": "result",
                    "task_id": task_id,
                    "ok": False,
                    "error": repr(exc),
                    "logical_device": logical_device,
                    "physical_device": physical_device,
                }
            )


def _collect_results() -> None:
    while not STATE.stop_event.is_set():
        try:
            message = STATE.result_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if message.get("type") == "worker_status":
            logical_device = message["logical_device"]
            with STATE.lock:
                STATE.worker_status[logical_device] = message
            continue
        task_id = message.get("task_id")
        with STATE.lock:
            event = STATE.pending.get(task_id)
            if event is not None:
                STATE.results[task_id] = message
                event.set()


def _start_workers() -> None:
    physical_devices = parse_visible_devices(
        os.getenv("ASCEND_RT_VISIBLE_DEVICES", "")
    )
    model_path = validate_model_path(os.getenv("ZIMAGE_MODEL_PATH", ""))
    queue_size = int(os.getenv("ZIMAGE_QUEUE_SIZE", "16"))
    STATE.task_queue = STATE.context.Queue(maxsize=max(queue_size, 1))
    STATE.result_queue = STATE.context.Queue()
    STATE.stop_event.clear()
    STATE.collector = threading.Thread(target=_collect_results, daemon=True)
    STATE.collector.start()

    for logical_device, physical_device in enumerate(physical_devices):
        process = STATE.context.Process(
            target=_worker,
            args=(
                logical_device,
                physical_device,
                str(model_path),
                STATE.task_queue,
                STATE.result_queue,
            ),
            daemon=True,
        )
        process.start()
        STATE.workers.append(process)

    deadline = time.monotonic() + float(
        os.getenv("ZIMAGE_MODEL_LOAD_TIMEOUT", "900")
    )
    while time.monotonic() < deadline:
        with STATE.lock:
            statuses = list(STATE.worker_status.values())
        if len(statuses) == len(physical_devices):
            break
        if any(not process.is_alive() for process in STATE.workers):
            break
        time.sleep(0.2)

    if not STATE.ready:
        details = {
            key: value for key, value in sorted(STATE.worker_status.items())
        }
        _stop_workers()
        raise RuntimeError(
            "Z-Image-Turbo workers failed to become ready: {}".format(details)
        )


def _stop_workers() -> None:
    STATE.stop_event.set()
    if STATE.task_queue is not None:
        for _ in STATE.workers:
            try:
                STATE.task_queue.put_nowait(None)
            except queue.Full:
                break
    for process in STATE.workers:
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
    STATE.workers.clear()


@asynccontextmanager
async def lifespan(_: FastAPI):
    _start_workers()
    try:
        yield
    finally:
        _stop_workers()


app = FastAPI(
    title="Z-Image-Turbo Ascend NPU API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> Dict[str, Any]:
    with STATE.lock:
        statuses = [
            STATE.worker_status[key]
            for key in sorted(STATE.worker_status)
        ]
    process_alive = [process.is_alive() for process in STATE.workers]
    healthy = STATE.ready and all(process_alive)
    return {
        "status": "ready" if healthy else "unhealthy",
        "workers": statuses,
        "process_alive": process_alive,
        "queue_capacity": os.getenv("ZIMAGE_QUEUE_SIZE", "16"),
    }


@app.post("/generate")
async def generate_image(
    payload: Optional[GenerateRequest] = Body(default=None),
    prompt: Optional[str] = Query(default=None),
) -> Response:
    if not STATE.ready or not all(process.is_alive() for process in STATE.workers):
        raise HTTPException(status_code=503, detail="Generator workers are not ready.")
    if payload is None:
        if not prompt or not prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt is empty.")
        payload = GenerateRequest(prompt=prompt)

    task_id = str(uuid.uuid4())
    event = threading.Event()
    task = payload.model_dump()
    task["task_id"] = task_id
    with STATE.lock:
        STATE.pending[task_id] = event
    try:
        STATE.task_queue.put_nowait(task)
    except queue.Full:
        with STATE.lock:
            STATE.pending.pop(task_id, None)
        raise HTTPException(status_code=429, detail="Generation queue is full.")

    timeout = float(os.getenv("ZIMAGE_REQUEST_TIMEOUT", "600"))
    completed = await asyncio.to_thread(event.wait, timeout)
    with STATE.lock:
        result = STATE.results.pop(task_id, None)
        STATE.pending.pop(task_id, None)
    if not completed or result is None:
        raise HTTPException(status_code=504, detail="Generation timed out.")
    if not result.get("ok"):
        raise HTTPException(
            status_code=500,
            detail="Generation failed on worker {}: {}".format(
                result.get("logical_device"), result.get("error")
            ),
        )
    return Response(
        content=result["content"],
        media_type="image/png",
        headers={
            "X-ZImage-Logical-Device": str(result["logical_device"]),
            "X-ZImage-Physical-Device": str(result["physical_device"]),
            "X-ZImage-Seed": str(result["seed"]),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve Z-Image-Turbo on one or two Ascend NPUs."
    )
    parser.add_argument(
        "--devices",
        default=os.getenv("ASCEND_RT_VISIBLE_DEVICES", ""),
        help="One or two physical IDs, for example 0 or 0,1.",
    )
    parser.add_argument(
        "--model-path",
        default=os.getenv("ZIMAGE_MODEL_PATH", ""),
        help="Local ModelScope Z-Image-Turbo snapshot path.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parse_visible_devices(args.devices)
    validate_model_path(args.model_path)
    # These are set before any spawned worker imports torch or torch_npu.
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = args.devices
    os.environ["ZIMAGE_MODEL_PATH"] = str(Path(args.model_path).resolve())
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
