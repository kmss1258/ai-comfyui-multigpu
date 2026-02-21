ARG BASE=nvcr.io/nvidia/pytorch:23.10-py3

FROM ${BASE}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/sr

COPY requirements.txt /tmp/sr-requirements.txt
RUN pip install --no-cache-dir -r /tmp/sr-requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torchaudio==2.1.0
RUN pip install --no-cache-dir --no-deps git+https://github.com/ysharma3501/NovaSR.git

COPY app /srv/sr/app

ENV NOVASR_DEVICE=auto
ENV NOVASR_HALF=1
ENV FFMPEG_PATH=ffmpeg

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
