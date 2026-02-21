from __future__ import annotations

from dataclasses import dataclass
import asyncio
import json
from typing import cast

import httpx
import websockets


JSONObject = dict[str, object]


@dataclass(frozen=True)
class UploadedImage:
    name: str
    subfolder: str
    folder_type: str


@dataclass(frozen=True)
class OutputArtifact:
    filename: str
    subfolder: str
    folder_type: str
    data: bytes


async def upload_image(
    client: httpx.AsyncClient,
    http_base_url: str,
    image_name: str,
    image_bytes: bytes,
) -> UploadedImage:
    files = {"image": (image_name, image_bytes, "application/octet-stream")}
    data = {"overwrite": "true"}
    response = await client.post(f"{http_base_url}/upload/image", files=files, data=data)
    _ = response.raise_for_status()
    payload_obj = response.json()
    if not isinstance(payload_obj, dict):
        raise RuntimeError("ComfyUI /upload/image response must be an object")
    payload = cast(JSONObject, payload_obj)
    if "name" not in payload:
        raise RuntimeError("ComfyUI /upload/image did not return uploaded file name")
    return UploadedImage(
        name=str(payload["name"]),
        subfolder=str(payload.get("subfolder", "")),
        folder_type=str(payload.get("type", "input")),
    )


async def queue_prompt(
    client: httpx.AsyncClient,
    http_base_url: str,
    workflow: JSONObject,
    client_id: str,
) -> str:
    response = await client.post(
        f"{http_base_url}/prompt",
        json={"prompt": workflow, "client_id": client_id},
    )
    _ = response.raise_for_status()
    payload_obj = response.json()
    if not isinstance(payload_obj, dict):
        raise RuntimeError("ComfyUI /prompt response must be an object")
    payload = cast(JSONObject, payload_obj)
    prompt_id = payload.get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id:
        raise RuntimeError("ComfyUI /prompt did not return prompt_id")
    return prompt_id


async def wait_for_completion(
    ws_base_url: str,
    client_id: str,
    prompt_id: str,
    timeout_seconds: float,
) -> None:
    ws_url = f"{ws_base_url}/ws?clientId={client_id}"
    async with asyncio.timeout(timeout_seconds):
        async with websockets.connect(ws_url, max_size=None) as websocket:
            while True:
                raw_message = await websocket.recv()
                if isinstance(raw_message, bytes):
                    continue

                payload_obj = json.loads(raw_message)
                if not isinstance(payload_obj, dict):
                    continue
                payload = cast(JSONObject, payload_obj)
                message_type = payload.get("type")
                data_obj = payload.get("data", {})
                data = data_obj if isinstance(data_obj, dict) else {}

                if message_type == "execution_error" and data.get("prompt_id") == prompt_id:
                    raise RuntimeError(f"ComfyUI execution_error: {data}")

                if message_type == "execution_interrupted" and data.get("prompt_id") == prompt_id:
                    raise RuntimeError(f"ComfyUI execution_interrupted: {data}")

                if (
                    message_type == "executing"
                    and data.get("prompt_id") == prompt_id
                    and data.get("node") is None
                ):
                    return

                if message_type == "execution_success" and data.get("prompt_id") == prompt_id:
                    return


async def get_history(
    client: httpx.AsyncClient,
    http_base_url: str,
    prompt_id: str,
) -> JSONObject:
    response = await client.get(f"{http_base_url}/history/{prompt_id}")
    _ = response.raise_for_status()
    payload_obj = response.json()
    if not isinstance(payload_obj, dict):
        raise RuntimeError("ComfyUI /history response must be an object")
    return cast(dict[str, object], payload_obj)


async def fetch_outputs(
    client: httpx.AsyncClient,
    http_base_url: str,
    history_item: JSONObject,
) -> list[OutputArtifact]:
    artifacts: list[OutputArtifact] = []
    outputs = history_item.get("outputs", {})
    if not isinstance(outputs, dict):
        return artifacts

    for node_output in outputs.values():
        if not isinstance(node_output, dict):
            continue
        for file_group in ("images", "videos", "gifs", "files"):
            file_items = node_output.get(file_group, [])
            if not isinstance(file_items, list):
                continue
            for file_info in file_items:
                if not isinstance(file_info, dict):
                    continue
                filename = str(file_info.get("filename", ""))
                subfolder = str(file_info.get("subfolder", ""))
                folder_type = str(file_info.get("type", "output"))
                if not filename:
                    continue
                response = await client.get(
                    f"{http_base_url}/view",
                    params={
                        "filename": filename,
                        "subfolder": subfolder,
                        "type": folder_type,
                    },
                )
                _ = response.raise_for_status()
                artifacts.append(
                    OutputArtifact(
                        filename=filename,
                        subfolder=subfolder,
                        folder_type=folder_type,
                        data=response.content,
                    )
                )

    return artifacts
