# ComfyUI Multi-GPU Router

FastAPI 1개에서 ComfyUI 여러 GPU 워커로 요청을 자동 분배하는 구성입니다.

## Topology

- FastAPI Router: `http://localhost:19164`
- ComfyUI GPU0: `http://localhost:19165` (외부 공개)
- ComfyUI GPU2: 내부 네트워크 전용
- ComfyUI GPU3: 내부 네트워크 전용
- ComfyUI GPU1: `docker-compose.yml`에서 주석 처리됨

## Run

```bash
docker compose up --build
```

## API

### 1) Health

```bash
curl -s http://localhost:19164/healthz
```

### 2) Inference (upload image + patch workflow + websocket wait)

`workflow_file`는 ComfyUI의 API JSON 형식 파일입니다.

```bash
curl -X POST "http://localhost:19164/v1/infer" \
  -F "workflow_file=@[API]wan2.2_i2v.json" \
  -F "image_file=@046_썸네일_박가린_janjju.png" \
  -F "text_prompt=a woman smiling at camera" \
  -F "image_node_id=67" \
  -F "text_node_id=98"
```

응답 예시:

```json
{
  "job_id": "...",
  "status": "completed",
  "worker": "comfyui-gpu2:8188",
  "prompt_id": "...",
  "outputs": [
    {
      "index": 0,
      "filename": "ComfyUI_00001_.png",
      "size_bytes": 123456,
      "source_type": "output",
      "subfolder": "",
      "download_url": "/v1/jobs/<job_id>/outputs/0"
    }
  ]
}
```

### 3) Job 조회

```bash
curl -s "http://localhost:19164/v1/jobs/<job_id>"
```

### 4) 결과 파일 다운로드

```bash
curl -L "http://localhost:19164/v1/jobs/<job_id>/outputs/0" -o result.bin
```

## Data Locations

- Comfy GPU0 run/basedir: `./comfy/gpu0/run`, `./comfy/gpu0/basedir`
- Comfy GPU2 run/basedir: `./comfy/gpu2/run`, `./comfy/gpu2/basedir`
- Comfy GPU3 run/basedir: `./comfy/gpu3/run`, `./comfy/gpu3/basedir`
- FastAPI saved outputs: `./router_data/outputs`

## Notes

- 컨테이너 첫 기동 시 `comfy/user_script.bash`가 아래 패키지를 설치합니다.
  - `transformers==4.57.3`
  - `aiofiles`
  - `python-socketio>=5`
- 호스트에 NVIDIA Container Toolkit이 정상 설치되어 있어야 합니다.
