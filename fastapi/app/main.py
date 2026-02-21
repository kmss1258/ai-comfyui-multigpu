from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, cast
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
import httpx
import websockets

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
worker_by_host_port = {worker.host_port: worker for worker in pool.workers}
client_worker_map: dict[str, str] = {}
prompt_worker_map: dict[str, str] = {}
artifact_worker_map: dict[tuple[str, str, str], str] = {}
prompt_worker_lease_map: dict[str, tuple[str, float]] = {}
busy_workers: set[str] = set()
router_state_lock = asyncio.Lock()
round_robin_index = 0


@asynccontextmanager
async def app_lifespan(_app: FastAPI):
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="ComfyUI GPU Router", version="1.0.0", lifespan=app_lifespan)


def _request_headers_for_upstream(request: Request) -> dict[str, str]:
    blocked = {"host", "content-length", "connection"}
    return {k: v for k, v in request.headers.items() if k.lower() not in blocked}


def _response_headers_for_client(headers: httpx.Headers) -> dict[str, str]:
    blocked = {"content-length", "transfer-encoding", "connection"}
    return {k: v for k, v in headers.items() if k.lower() not in blocked}


async def _passthrough_http(request: Request, worker_host_port: str) -> Response:
    worker = _get_worker_or_500(worker_host_port)
    timeout = httpx.Timeout(settings.request_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        upstream_response = await client.request(
            method=request.method,
            url=f"{worker.http_base_url}{request.url.path}",
            params=request.query_params,
            content=await request.body(),
            headers=_request_headers_for_upstream(request),
        )

    return Response(
        content=upstream_response.content,
        status_code=upstream_response.status_code,
        headers=_response_headers_for_client(upstream_response.headers),
        media_type=upstream_response.headers.get("content-type"),
    )


def _get_worker_or_500(host_port: str):
    worker = worker_by_host_port.get(host_port)
    if worker is None:
        raise HTTPException(status_code=500, detail=f"unknown worker mapping: {host_port}")
    return worker


async def _pick_worker(client_id: str | None) -> str:
    global round_robin_index
    async with router_state_lock:
        if client_id and client_id in client_worker_map:
            return client_worker_map[client_id]

        selected = pool.workers[round_robin_index % len(pool.workers)].host_port
        round_robin_index += 1
        if client_id:
            client_worker_map[client_id] = selected
        return selected


def _cleanup_expired_prompt_leases_locked(now_monotonic: float) -> None:
    expired_prompt_ids: list[str] = []
    for prompt_id, lease in prompt_worker_lease_map.items():
        _, started_at = lease
        if now_monotonic - started_at >= settings.prompt_lease_ttl_seconds:
            expired_prompt_ids.append(prompt_id)

    for prompt_id in expired_prompt_ids:
        host_port, _started_at = prompt_worker_lease_map.pop(prompt_id)
        busy_workers.discard(host_port)
        prompt_worker_map.pop(prompt_id, None)


async def _acquire_prompt_worker(client_id: str | None) -> str:
    global round_robin_index

    while True:
        async with router_state_lock:
            _cleanup_expired_prompt_leases_locked(time.monotonic())

            mapped_worker = client_worker_map.get(client_id) if client_id else None
            if mapped_worker:
                if mapped_worker not in busy_workers:
                    busy_workers.add(mapped_worker)
                    return mapped_worker
            else:
                worker_count = len(pool.workers)
                for offset in range(worker_count):
                    idx = (round_robin_index + offset) % worker_count
                    candidate = pool.workers[idx].host_port
                    if candidate in busy_workers:
                        continue
                    round_robin_index = (idx + 1) % worker_count
                    busy_workers.add(candidate)
                    if client_id:
                        client_worker_map[client_id] = candidate
                    return candidate

        await asyncio.sleep(0.05)


async def _release_prompt_worker_by_prompt_id(prompt_id: str) -> None:
    async with router_state_lock:
        lease = prompt_worker_lease_map.pop(prompt_id, None)
        if lease is None:
            return
        host_port, _started_at = lease
        busy_workers.discard(host_port)


async def _index_history_artifacts(prompt_id: str, history_payload: JSONObject) -> None:
    history_item_obj = history_payload.get(prompt_id)
    if not isinstance(history_item_obj, dict):
        return
    history_item = cast(JSONObject, history_item_obj)
    outputs_obj = history_item.get("outputs")
    if not isinstance(outputs_obj, dict):
        return

    async with router_state_lock:
        worker = prompt_worker_map.get(prompt_id)
        if worker is None:
            return
        for node_output_obj in outputs_obj.values():
            if not isinstance(node_output_obj, dict):
                continue
            node_output = cast(JSONObject, node_output_obj)
            for file_key in ("images", "videos", "gifs", "files"):
                files_obj = node_output.get(file_key)
                if not isinstance(files_obj, list):
                    continue
                for file_obj in files_obj:
                    if not isinstance(file_obj, dict):
                        continue
                    file_info = cast(JSONObject, file_obj)
                    filename = str(file_info.get("filename", ""))
                    subfolder = str(file_info.get("subfolder", ""))
                    folder_type = str(file_info.get("type", "output"))
                    if filename:
                        artifact_worker_map[(filename, subfolder, folder_type)] = worker
        _cleanup_expired_prompt_leases_locked(time.monotonic())


@app.get("/healthz")
async def healthz() -> JSONObject:
    return {
        "ok": True,
        "capacity": pool.capacity,
        "available": pool.available,
        "workers": [worker.host_port for worker in pool.workers],
    }


@app.get("/router/state")
async def router_state() -> JSONObject:
    now_mono = time.monotonic()
    async with router_state_lock:
        _cleanup_expired_prompt_leases_locked(now_mono)
        lease_items = [
            {
                "prompt_id": prompt_id,
                "worker": host_port,
                "age_seconds": round(max(0.0, now_mono - started_at), 3),
            }
            for prompt_id, (host_port, started_at) in prompt_worker_lease_map.items()
        ]

        return {
            "workers": [worker.host_port for worker in pool.workers],
            "busy_workers": sorted(busy_workers),
            "busy_count": len(busy_workers),
            "prompt_worker_map_size": len(prompt_worker_map),
            "client_worker_map_size": len(client_worker_map),
            "artifact_worker_map_size": len(artifact_worker_map),
            "prompt_worker_lease_count": len(prompt_worker_lease_map),
            "prompt_leases": lease_items,
            "lease_ttl_seconds": settings.prompt_lease_ttl_seconds,
        }


@app.post("/upload/image")
async def comfy_upload_image_proxy(
    image: Annotated[UploadFile, File(...)],
    overwrite: Annotated[str | None, Form()] = None,
    subfolder: Annotated[str | None, Form()] = None,
    folder_type: Annotated[str | None, Form(alias="type")] = None,
    client_id: Annotated[str | None, Form()] = None,
) -> JSONObject:
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="image file is empty")

    form_data: dict[str, str] = {}
    if overwrite is not None:
        form_data["overwrite"] = overwrite
    if subfolder is not None:
        form_data["subfolder"] = subfolder
    if folder_type is not None:
        form_data["type"] = folder_type

    selected_host = await _pick_worker(client_id)
    worker = _get_worker_or_500(selected_host)

    timeout = httpx.Timeout(settings.request_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{worker.http_base_url}/upload/image",
            files={"image": (image.filename or "input.bin", image_bytes, "application/octet-stream")},
            data=form_data,
        )
        _ = response.raise_for_status()
        payload_obj = response.json()
    if not isinstance(payload_obj, dict):
        raise HTTPException(status_code=502, detail="invalid upload response from worker")
    return cast(JSONObject, payload_obj)


@app.post("/prompt")
async def comfy_prompt_proxy(request: Request) -> Response:
    try:
        body_obj = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="request body must be JSON") from exc

    if not isinstance(body_obj, dict):
        raise HTTPException(status_code=400, detail="request body must be a JSON object")

    body = cast(JSONObject, body_obj)
    client_id_obj = body.get("client_id")
    client_id = str(client_id_obj) if client_id_obj is not None else None

    target_host = await _acquire_prompt_worker(client_id)
    worker = _get_worker_or_500(target_host)

    keep_slot_reserved = False
    timeout = httpx.Timeout(settings.request_timeout_seconds)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{worker.http_base_url}/prompt", json=body)
            payload_bytes = response.content
            status_code = response.status_code

        if status_code < 400:
            try:
                payload_obj = json.loads(payload_bytes)
            except json.JSONDecodeError:
                payload_obj = None
            if isinstance(payload_obj, dict):
                prompt_id_obj = payload_obj.get("prompt_id")
                if isinstance(prompt_id_obj, str) and prompt_id_obj:
                    async with router_state_lock:
                        prompt_worker_map[prompt_id_obj] = target_host
                        prompt_worker_lease_map[prompt_id_obj] = (
                            target_host,
                            time.monotonic(),
                        )
                        if client_id:
                            client_worker_map[client_id] = target_host
                        _cleanup_expired_prompt_leases_locked(time.monotonic())
                    keep_slot_reserved = True

        return Response(content=payload_bytes, status_code=status_code, media_type="application/json")
    finally:
        if not keep_slot_reserved:
            async with router_state_lock:
                busy_workers.discard(target_host)
                if client_id and client_worker_map.get(client_id) == target_host:
                    client_worker_map.pop(client_id, None)
                _cleanup_expired_prompt_leases_locked(time.monotonic())


@app.get("/history/{prompt_id}")
async def comfy_history_proxy(prompt_id: str) -> JSONObject:
    timeout = httpx.Timeout(settings.request_timeout_seconds)
    async with router_state_lock:
        mapped_worker = prompt_worker_map.get(prompt_id)

    workers_to_try = [mapped_worker] if mapped_worker else []
    workers_to_try.extend(worker.host_port for worker in pool.workers if worker.host_port != mapped_worker)

    async with httpx.AsyncClient(timeout=timeout) as client:
        for host_port in workers_to_try:
            worker = _get_worker_or_500(host_port)
            response = await client.get(f"{worker.http_base_url}/history/{prompt_id}")
            _ = response.raise_for_status()
            payload_obj = response.json()
            if not isinstance(payload_obj, dict):
                continue
            payload = cast(JSONObject, payload_obj)
            if prompt_id in payload:
                async with router_state_lock:
                    prompt_worker_map[prompt_id] = host_port
                    _cleanup_expired_prompt_leases_locked(time.monotonic())
                await _index_history_artifacts(prompt_id, payload)
                await _release_prompt_worker_by_prompt_id(prompt_id)
                return payload

    return {}


@app.get("/view")
async def comfy_view_proxy(
    filename: str,
    subfolder: str = "",
    folder_type: Annotated[str, Query(alias="type")] = "output",
) -> Response:
    key = (filename, subfolder, folder_type)
    async with router_state_lock:
        mapped_worker = artifact_worker_map.get(key)

    workers_to_try = [mapped_worker] if mapped_worker else []
    workers_to_try.extend(worker.host_port for worker in pool.workers if worker.host_port != mapped_worker)

    timeout = httpx.Timeout(settings.request_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for host_port in workers_to_try:
            worker = _get_worker_or_500(host_port)
            response = await client.get(
                f"{worker.http_base_url}/view",
                params={"filename": filename, "subfolder": subfolder, "type": folder_type},
            )
            if response.status_code == 200:
                async with router_state_lock:
                    artifact_worker_map[key] = host_port
                content_type = response.headers.get("content-type", "application/octet-stream")
                return Response(content=response.content, media_type=content_type)

    raise HTTPException(status_code=404, detail="file not found on any worker")


@app.websocket("/ws")
async def comfy_ws_proxy(websocket: WebSocket) -> None:
    client_id = websocket.query_params.get("clientId")
    target_host = await _pick_worker(client_id)
    worker = _get_worker_or_500(target_host)

    upstream_url = f"{worker.ws_base_url}/ws"
    if client_id:
        upstream_url = f"{upstream_url}?clientId={client_id}"

    await websocket.accept()

    try:
        async with websockets.connect(upstream_url, max_size=None) as upstream:
            async def client_to_upstream() -> None:
                while True:
                    message = await websocket.receive()
                    message_type = message.get("type")
                    if message_type == "websocket.disconnect":
                        return

                    text_data = message.get("text")
                    bytes_data = message.get("bytes")
                    if isinstance(text_data, str):
                        await upstream.send(text_data)
                    elif isinstance(bytes_data, bytes):
                        await upstream.send(bytes_data)

            async def upstream_to_client() -> None:
                while True:
                    incoming = await upstream.recv()
                    if isinstance(incoming, bytes):
                        await websocket.send_bytes(incoming)
                    else:
                        try:
                            event_obj = json.loads(incoming)
                        except json.JSONDecodeError:
                            event_obj = None
                        if isinstance(event_obj, dict):
                            event_type = event_obj.get("type")
                            event_data_obj = event_obj.get("data")
                            event_data = event_data_obj if isinstance(event_data_obj, dict) else None
                            if event_data is not None:
                                event_prompt_id = event_data.get("prompt_id")
                                if isinstance(event_prompt_id, str) and event_prompt_id:
                                    if event_type == "execution_success":
                                        await _release_prompt_worker_by_prompt_id(event_prompt_id)
                                    elif event_type == "executing" and event_data.get("node") is None:
                                        await _release_prompt_worker_by_prompt_id(event_prompt_id)
                                    elif event_type in ("execution_error", "execution_interrupted"):
                                        await _release_prompt_worker_by_prompt_id(event_prompt_id)
                        await websocket.send_text(incoming)

            tasks = [
                asyncio.create_task(client_to_upstream()),
                asyncio.create_task(upstream_to_client()),
            ]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                _ = task.cancel()
            for task in done:
                _ = task.result()
    except WebSocketDisconnect:
        return
    except Exception:
        await websocket.close(code=1011)


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


@app.api_route(
    "/{full_path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def catch_all_passthrough(full_path: str, request: Request) -> Response:
    if full_path.startswith("v1/") or full_path == "healthz":
        raise HTTPException(status_code=404, detail="route not found")
    async with pool.lease() as worker:
        return await _passthrough_http(request, worker.host_port)
