from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, cast
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
import httpx

from .comfy_client import fetch_outputs, get_history, queue_prompt, upload_image, wait_for_completion
from .config import load_settings
from .worker_pool import WorkerPool


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


JSONObject = dict[str, object]


@dataclass
class OutputFileMeta:
    filename: str
    saved_path: Path
    size_bytes: int
    source_type: str
    subfolder: str


@dataclass
class JobRecord:
    job_id: str
    worker: str
    status: str
    created_at: str
    updated_at: str
    prompt_id: str | None = None
    error: str | None = None
    outputs: list[OutputFileMeta] = field(default_factory=list)


def apply_workflow_patch(
    workflow: JSONObject,
    uploaded_filename: str | None,
    image_node_id: str | None,
    text_prompt: str | None,
    text_node_id: str | None,
) -> JSONObject:
    if uploaded_filename and image_node_id:
        node = workflow.get(image_node_id)
        if not isinstance(node, dict):
            raise HTTPException(status_code=400, detail=f"image_node_id '{image_node_id}' not found in workflow")
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            raise HTTPException(status_code=400, detail=f"workflow node '{image_node_id}' has no 'inputs' object")
        inputs["image"] = uploaded_filename

    if text_prompt is not None and text_node_id:
        node = workflow.get(text_node_id)
        if not isinstance(node, dict):
            raise HTTPException(status_code=400, detail=f"text_node_id '{text_node_id}' not found in workflow")
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            raise HTTPException(status_code=400, detail=f"workflow node '{text_node_id}' has no 'inputs' object")
        inputs["text"] = text_prompt

    return workflow


async def wait_until_history_present(
    client: httpx.AsyncClient,
    worker_http_base_url: str,
    prompt_id: str,
    timeout_seconds: float,
) -> JSONObject:
    async with asyncio.timeout(timeout_seconds):
        while True:
            history_payload = await get_history(
                client=client,
                http_base_url=worker_http_base_url,
                prompt_id=prompt_id,
            )
            history_item = history_payload.get(prompt_id)
            if isinstance(history_item, dict):
                return cast(JSONObject, history_item)
            await asyncio.sleep(1.0)


settings = load_settings()
pool = WorkerPool(settings.comfyui_workers)
jobs: dict[str, JobRecord] = {}


@asynccontextmanager
async def app_lifespan(_app: FastAPI):
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="ComfyUI GPU Router", version="1.0.0", lifespan=app_lifespan)


@app.get("/healthz")
async def healthz() -> JSONObject:
    return {
        "ok": True,
        "capacity": pool.capacity,
        "available": pool.available,
        "workers": [worker.host_port for worker in pool.workers],
    }


@app.get("/v1/jobs/{job_id}")
async def get_job(job_id: str) -> JSONObject:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    return {
        "job_id": job.job_id,
        "worker": job.worker,
        "status": job.status,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "prompt_id": job.prompt_id,
        "error": job.error,
        "outputs": [
            {
                "index": idx,
                "filename": output.filename,
                "size_bytes": output.size_bytes,
                "source_type": output.source_type,
                "subfolder": output.subfolder,
                "download_url": f"/v1/jobs/{job_id}/outputs/{idx}",
            }
            for idx, output in enumerate(job.outputs)
        ],
    }


@app.get("/v1/jobs/{job_id}/outputs/{index}")
async def get_job_output(job_id: str, index: int) -> Response:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    if index < 0 or index >= len(job.outputs):
        raise HTTPException(status_code=404, detail="output not found")

    output = job.outputs[index]
    if not output.saved_path.exists():
        raise HTTPException(status_code=404, detail="saved output file missing")

    return FileResponse(path=output.saved_path, filename=output.filename)


@app.post("/v1/infer")
async def infer(
    workflow_file: Annotated[UploadFile, File(...)],
    image_file: Annotated[UploadFile | None, File()] = None,
    text_prompt: Annotated[str | None, Form()] = None,
    image_node_id: Annotated[str | None, Form()] = None,
    text_node_id: Annotated[str | None, Form()] = None,
) -> JSONObject:
    try:
        workflow_raw = await workflow_file.read()
        workflow_obj_raw = json.loads(workflow_raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="workflow_file must be valid JSON") from exc

    if not isinstance(workflow_obj_raw, dict):
        raise HTTPException(status_code=400, detail="workflow JSON root must be an object")
    workflow_obj = cast(JSONObject, workflow_obj_raw)

    async with pool.lease() as worker:
        job_id = str(uuid4())
        created_at = now_iso()
        job = JobRecord(
            job_id=job_id,
            worker=worker.host_port,
            status="running",
            created_at=created_at,
            updated_at=created_at,
        )
        jobs[job_id] = job

        timeout = httpx.Timeout(settings.request_timeout_seconds)
        limits = httpx.Limits(max_keepalive_connections=10, max_connections=30)
        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
            try:
                uploaded_filename: str | None = None
                if image_file is not None:
                    image_bytes = await image_file.read()
                    if not image_bytes:
                        raise HTTPException(status_code=400, detail="image_file is empty")

                    uploaded = await upload_image(
                        client=client,
                        http_base_url=worker.http_base_url,
                        image_name=image_file.filename or "input.png",
                        image_bytes=image_bytes,
                    )
                    uploaded_filename = uploaded.name

                workflow_obj = apply_workflow_patch(
                    workflow=workflow_obj,
                    uploaded_filename=uploaded_filename,
                    image_node_id=image_node_id,
                    text_prompt=text_prompt,
                    text_node_id=text_node_id,
                )

                client_id = str(uuid4())
                prompt_id = await queue_prompt(
                    client=client,
                    http_base_url=worker.http_base_url,
                    workflow=workflow_obj,
                    client_id=client_id,
                )
                job.prompt_id = prompt_id
                job.updated_at = now_iso()

                try:
                    await wait_for_completion(
                        ws_base_url=worker.ws_base_url,
                        client_id=client_id,
                        prompt_id=prompt_id,
                        timeout_seconds=settings.request_timeout_seconds,
                    )
                    history_obj = await wait_until_history_present(
                        client=client,
                        worker_http_base_url=worker.http_base_url,
                        prompt_id=prompt_id,
                        timeout_seconds=settings.request_timeout_seconds,
                    )
                except Exception:
                    history_obj = await wait_until_history_present(
                        client=client,
                        worker_http_base_url=worker.http_base_url,
                        prompt_id=prompt_id,
                        timeout_seconds=settings.request_timeout_seconds,
                    )

                output_artifacts = await fetch_outputs(
                    client=client,
                    http_base_url=worker.http_base_url,
                    history_item=history_obj,
                )

                job_output_dir = settings.output_dir / job_id
                job_output_dir.mkdir(parents=True, exist_ok=True)
                output_metas: list[OutputFileMeta] = []
                for idx, artifact in enumerate(output_artifacts):
                    target_name = f"{idx:03d}_{artifact.filename}"
                    target_path = job_output_dir / target_name
                    _ = target_path.write_bytes(artifact.data)
                    output_metas.append(
                        OutputFileMeta(
                            filename=artifact.filename,
                            saved_path=target_path,
                            size_bytes=len(artifact.data),
                            source_type=artifact.folder_type,
                            subfolder=artifact.subfolder,
                        )
                    )

                job.outputs = output_metas
                job.status = "completed"
                job.updated_at = now_iso()

            except HTTPException:
                job.status = "failed"
                job.error = "invalid request payload"
                job.updated_at = now_iso()
                raise
            except Exception as exc:
                job.status = "failed"
                job.error = str(exc)
                job.updated_at = now_iso()
                raise HTTPException(status_code=500, detail=f"inference failed on worker {worker.host_port}: {exc}") from exc

    return {
        "job_id": job.job_id,
        "status": job.status,
        "worker": job.worker,
        "prompt_id": job.prompt_id,
        "outputs": [
            {
                "index": idx,
                "filename": output.filename,
                "size_bytes": output.size_bytes,
                "source_type": output.source_type,
                "subfolder": output.subfolder,
                "download_url": f"/v1/jobs/{job.job_id}/outputs/{idx}",
            }
            for idx, output in enumerate(job.outputs)
        ],
    }
