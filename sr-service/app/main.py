from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from threading import Lock

import torch
import torchaudio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_device(device_name: str) -> str:
    device_name = device_name.strip().lower()
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        return "cpu"
    if device_name not in {"cpu", "cuda"}:
        return "cpu"
    return device_name


def _infer_suffix(filename: str | None, content_type: str | None) -> str:
    if filename:
        ext = Path(filename).suffix.lower()
        if ext in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
            return ext
    if content_type in {"audio/wav", "audio/x-wav"}:
        return ".wav"
    if content_type == "audio/mpeg":
        return ".mp3"
    if content_type == "audio/flac":
        return ".flac"
    if content_type == "audio/ogg":
        return ".ogg"
    if content_type in {"audio/mp4", "audio/x-m4a"}:
        return ".m4a"
    return ".bin"


class NovaSRService:
    def __init__(self) -> None:
        self.ffmpeg_path = os.getenv("FFMPEG_PATH", "ffmpeg")
        self.device = _resolve_device(os.getenv("NOVASR_DEVICE", "auto"))
        self.half = _env_bool("NOVASR_HALF", True) and self.device == "cuda"
        self.model_lock = Lock()

        if self.device == "cpu":
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

        from NovaSR import FastSR

        self.model = FastSR(half=self.half)
        target = torch.device(self.device)
        self.model.device = target
        self.model.model = self.model.model.to(target)
        if self.half:
            self.model.model = self.model.model.half()
        else:
            self.model.model = self.model.model.float()

    def _run_ffmpeg(self, args: list[str], error_prefix: str) -> None:
        output = subprocess.run(
            [self.ffmpeg_path, *args],
            capture_output=True,
            text=True,
            check=False,
        )
        if output.returncode != 0:
            raise RuntimeError(f"{error_prefix}: {output.stderr.strip()}")

    def super_resolve_bytes(self, audio_bytes: bytes, suffix: str, output_format: str) -> tuple[bytes, str]:
        with tempfile.TemporaryDirectory(prefix="novasr-") as tmp_dir:
            tmp = Path(tmp_dir)
            input_path = tmp / f"input{suffix}"
            input_path.write_bytes(audio_bytes)

            input_wav = tmp / "input_24k.wav"
            self._run_ffmpeg(
                [
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(input_path),
                    "-ac",
                    "1",
                    "-ar",
                    "24000",
                    "-c:a",
                    "pcm_s16le",
                    str(input_wav),
                ],
                "ffmpeg decode/transcode failed",
            )

            with self.model_lock:
                lowres = self.model.load_audio(str(input_wav)).to(self.model.device)
                with torch.no_grad():
                    highres = self.model.infer(lowres).detach().cpu().float()

            if highres.dim() == 1:
                highres = highres.unsqueeze(0)
            elif highres.dim() > 2:
                highres = highres.reshape(1, -1)

            out_wav = tmp / "output_48k.wav"
            torchaudio.save(str(out_wav), highres, 48000)

            if output_format == "mp3":
                out_mp3 = tmp / "output_48k.mp3"
                self._run_ffmpeg(
                    [
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-i",
                        str(out_wav),
                        "-ac",
                        "1",
                        "-ar",
                        "48000",
                        "-codec:a",
                        "libmp3lame",
                        str(out_mp3),
                    ],
                    "ffmpeg mp3 encode failed",
                )
                return out_mp3.read_bytes(), "audio/mpeg"

            return out_wav.read_bytes(), "audio/wav"


app = FastAPI(title="NovaSR API", version="1.0.0")
service: NovaSRService | None = None
startup_error: str | None = None


@app.on_event("startup")
def _startup() -> None:
    global service
    global startup_error
    try:
        service = NovaSRService()
        startup_error = None
    except Exception as exc:
        service = None
        startup_error = str(exc)


@app.get("/healthz")
def healthz() -> Response:
    if service is None:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "error": startup_error or "service unavailable"},
        )
    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "device": service.device,
            "half": service.half,
        },
    )


@app.post("/v1/audio/super-resolve")
async def super_resolve(
    audio: UploadFile = File(...),
    output_format: str = Form("wav"),
) -> Response:
    if service is None:
        raise HTTPException(status_code=503, detail=startup_error or "NovaSR service not ready")

    output_format = output_format.strip().lower()
    if output_format not in {"wav", "mp3"}:
        raise HTTPException(status_code=400, detail="output_format must be one of: wav, mp3")

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="audio is empty")

    suffix = _infer_suffix(audio.filename, audio.content_type)

    try:
        result_bytes, content_type = service.super_resolve_bytes(audio_bytes, suffix, output_format)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"NovaSR inference failed: {exc}") from exc

    return Response(content=result_bytes, media_type=content_type)
