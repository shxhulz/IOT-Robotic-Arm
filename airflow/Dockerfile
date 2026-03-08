FROM astrocrpublic.azurecr.io/runtime:3.1-13

USER root

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

USER astro

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install opencv-python-headless minio ultralytics nvidia-nvjitlink-cu12
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.12/site-packages/nvidia/nvjitlink/lib:/home/astro/.local/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH"