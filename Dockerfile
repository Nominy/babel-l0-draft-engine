FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/cache \
    HF_HOME=/cache/huggingface \
    TMPDIR=/var/tmp/l0-draft

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates ffmpeg libsndfile1 python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python3 -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

COPY l0_draft_engine /app/l0_draft_engine
RUN useradd --create-home --uid 10001 engine \
    && mkdir -p /cache/huggingface /var/tmp/l0-draft \
    && chown -R engine:engine /cache /var/tmp/l0-draft

USER engine
EXPOSE 8767
CMD ["python3", "-m", "uvicorn", "l0_draft_engine.app:app", "--host", "0.0.0.0", "--port", "8767"]
