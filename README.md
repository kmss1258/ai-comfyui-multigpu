# ai-comfyui-multigpu

ComfyUI를 Docker로 띄우고, FastAPI 라우터(`19164`)로 요청을 받아 멀티 GPU 워커로 라운드로빈 라우팅하는 레포입니다.

- `docker-compose.gpu0.yml`: GPU0 단일 점검/세팅용
- `docker-compose.yml`: 멀티 GPU(0, 2, 3) 라우팅용

## 실행 순서 (명령어 생략 없이)

### 1) 레포 클론

```bash
git clone https://github.com/kmss1258/ai-comfyui-multigpu.git
cd ai-comfyui-multigpu
```

### 2) GPU0 단일 구성 기동 (`docker-compose.gpu0.yml`)

```bash
docker compose -f docker-compose.gpu0.yml down --remove-orphans
docker compose -f docker-compose.gpu0.yml up -d --build --force-recreate
docker compose -f docker-compose.gpu0.yml ps
docker compose -f docker-compose.gpu0.yml logs -f comfyui-gpu0 comfy-fastapi-router
```

### 3) ComfyUI 확인 및 노드 설치

1. 브라우저에서 `http://localhost:19163` 접속
2. 필요한 ComfyUI 노드 설치
3. 워크플로우 실행 후 정상 동작 확인

실행이 안 되면:

1. 컨테이너 로그에서 에러 확인
2. 레포 루트 `commands.txt`에 적힌 명령으로 추가 설치 진행

### 4) 멀티 GPU(0,2,3) 라우팅 기동 후 19164로 동시 요청

```bash
docker compose down --remove-orphans
docker compose up -d --build --force-recreate
docker compose ps
docker compose logs -f comfy-fastapi-router comfyui-gpu0 comfyui-gpu2 comfyui-gpu3
```

라우터 상태 확인:

```bash
curl -s http://localhost:19164/healthz
curl -s http://localhost:19164/router/state
```

동시 요청 예시(3개 병렬):

```bash
curl -s -X POST "http://localhost:19164/prompt" -H "Content-Type: application/json" -d '{"prompt": {}}' &
curl -s -X POST "http://localhost:19164/prompt" -H "Content-Type: application/json" -d '{"prompt": {}}' &
curl -s -X POST "http://localhost:19164/prompt" -H "Content-Type: application/json" -d '{"prompt": {}}' &
wait
```

실제 사용 시에는 빈 `{}` 대신 Comfy API 포맷의 실제 workflow JSON을 `prompt`에 넣어야 합니다.

### Super Resolution API (19164)

`/v1/audio/super-resolve`는 첫 번째 인식 워커(현재 설정상 `comfyui-gpu0`)로 프록시됩니다.

```bash
curl -X POST "http://localhost:19164/v1/audio/super-resolve" \
  -F "audio=@./input.mp3" \
  -F "output_format=wav" \
  --output sr_output.wav
```

## 포트 요약

- ComfyUI(GPU0): `19163`
- FastAPI Router: `19164`
- ComfyUI(GPU2, GPU3): 내부 네트워크 전용
