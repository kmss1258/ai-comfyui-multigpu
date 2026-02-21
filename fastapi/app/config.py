from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    comfyui_workers: tuple[str, ...]
    request_timeout_seconds: float
    output_dir: Path


def load_settings() -> Settings:
    raw_workers = os.getenv("COMFYUI_WORKERS", "comfyui-gpu0:8188,comfyui-gpu2:8188,comfyui-gpu3:8188")
    workers = tuple(part.strip() for part in raw_workers.split(",") if part.strip())
    if not workers:
        raise ValueError("COMFYUI_WORKERS must include at least one worker")

    timeout_raw = os.getenv("COMFY_REQUEST_TIMEOUT_SECONDS", "3600")
    try:
        timeout = float(timeout_raw)
    except ValueError as exc:
        raise ValueError("COMFY_REQUEST_TIMEOUT_SECONDS must be numeric") from exc
    if timeout <= 0:
        raise ValueError("COMFY_REQUEST_TIMEOUT_SECONDS must be greater than zero")

    output_dir = Path(os.getenv("OUTPUT_DIR", "/data/outputs")).resolve()

    return Settings(
        comfyui_workers=workers,
        request_timeout_seconds=timeout,
        output_dir=output_dir,
    )
