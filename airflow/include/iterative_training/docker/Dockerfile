FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_CACHE_DIR=/models \
    TORCH_HOME=/models/torch \
    HF_HOME=/models/hf \
    TRANSFORMERS_CACHE=/models/hf \
    OPENCLIP_CACHE_DIR=/models/clip

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    python3.10-dev \
    build-essential \
    ca-certificates \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY include/iterative_training/docker/requirements-gpu.txt /tmp/requirements-gpu.txt

RUN pip install --upgrade pip setuptools wheel && \
    pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1 torchvision==0.19.1 && \
    pip install --no-cache-dir -r /tmp/requirements-gpu.txt

COPY include/iterative_training /opt/iterative_training

ARG PRELOAD_MODELS=0
RUN if [ "$PRELOAD_MODELS" = "1" ]; then \
      python /opt/iterative_training/scripts/download_models.py --best-effort; \
    fi

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_CACHE_DIR=/models \
    TORCH_HOME=/models/torch \
    HF_HOME=/models/hf \
    TRANSFORMERS_CACHE=/models/hf \
    OPENCLIP_CACHE_DIR=/models/clip \
    PYTHONPATH=/opt/iterative_training

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/iterative_training /opt/iterative_training

ENV PATH=/opt/venv/bin:$PATH

RUN useradd -m -u 10001 appuser && \
    mkdir -p /models /opt/iterative_training/data && \
    chown -R appuser:appuser /models /opt/iterative_training

USER appuser
WORKDIR /opt/iterative_training

CMD ["python", "/opt/iterative_training/scripts/process_sam2.py"]
