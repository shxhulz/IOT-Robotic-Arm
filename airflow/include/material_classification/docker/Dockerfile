FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_DIR=/models \
    TORCH_HOME=/models/torch \
    HF_HOME=/models/hf \
    TRANSFORMERS_CACHE=/models/hf \
    OPENCLIP_CACHE_DIR=/models/clip

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-venv \
    ca-certificates \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY include/material_classification/docker/requirements-gpu.txt /tmp/requirements-gpu.txt

RUN pip install --upgrade pip setuptools wheel && \
    pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1 torchvision==0.19.1 && \
    pip install --no-cache-dir -r /tmp/requirements-gpu.txt

COPY include/material_classification/app /opt/material/app
RUN chmod +x /opt/material/app/entrypoint.sh

RUN useradd -m -u 10001 appuser && \
    mkdir -p /input /output /models && \
    chown -R appuser:appuser /opt/material /input /output /models

ARG PRELOAD_MODELS=0
RUN if [ "$PRELOAD_MODELS" = "1" ]; then python /opt/material/app/download_models.py --best-effort; fi

USER appuser
WORKDIR /opt/material/app

ENTRYPOINT ["/opt/material/app/entrypoint.sh"]
