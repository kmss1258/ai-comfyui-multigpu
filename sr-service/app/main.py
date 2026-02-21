from __future__ import annotations

from io import BytesIO

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydub import AudioSegment, effects


MEDIA_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
}


app = FastAPI(title="Audio Super Resolve", version="1.0.0")


@app.get("/healthz")
async def healthz() -> dict[str, bool]:
    return {"ok": True}


@app.post("/v1/audio/super-resolve")
async def super_resolve(
    audio: UploadFile = File(...),
    output_format: str = Form("wav"),
) -> Response:
    fmt = output_format.lower().strip()
    if fmt not in MEDIA_TYPES:
        raise HTTPException(status_code=400, detail=f"unsupported output_format: {output_format}")

    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="audio file is empty")

    try:
        source = AudioSegment.from_file(BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"failed to decode audio: {exc}") from exc

    enhanced = effects.normalize(source.set_frame_rate(48000).set_channels(1))

    output_bytes = BytesIO()
    export_kwargs: dict[str, str] = {}
    if fmt == "mp3":
        export_kwargs["bitrate"] = "192k"

    try:
        enhanced.export(output_bytes, format=fmt, **export_kwargs)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"failed to encode output audio: {exc}") from exc

    return Response(content=output_bytes.getvalue(), media_type=MEDIA_TYPES[fmt])
