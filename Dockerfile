# Base image with CUDA 12.8 and PyTorch compatible drivers
FROM runpod/pytorch:1.0.2-cu1281-torch271-ubuntu2204 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FORCE_CUDA=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libglib2.0-0 libsm6 libxext6 libxrender1 git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (leverage layer cache)
WORKDIR /app

# Prepare model directories
ENV MODELS_DIR=/models 
ENV HF_HOME=${MODELS_DIR} \
    HF_HUB_CACHE=${MODELS_DIR}

# Build arguments for model configuration
ARG RANK=128
ARG LIGHTING=8
ARG USE_ORIGINAL_TEXT_ENCODER="true"

# Make build args available as environment variables
ENV RANK=${RANK} \
    LIGHTING=${LIGHTING} \
    USE_ORIGINAL_TEXT_ENCODER=${USE_ORIGINAL_TEXT_ENCODER}
    
# Copy application code and requirements
COPY requirements.txt ./
COPY handler.py ./
COPY download_models.py ./
COPY ./models/ ${MODELS_DIR}/

# Install Python dependencies and Nunchaku
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
 && python3 -m pip install --no-cache-dir --no-compile -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128 \
 && python3 -m pip install --no-cache-dir https://github.com/nunchaku-tech/nunchaku/releases/download/v1.0.1/nunchaku-1.0.1+torch2.7-cp310-cp310-linux_x86_64.whl \
 && python3 download_models.py ${MODELS_DIR} \
 && rm -rf /root/.cache/pip \
 && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
 && find ${MODELS_DIR} -type d -name ".git" -exec rm -rf {} + 2>/dev/null || true \
 && find ${MODELS_DIR} -type f \( -name "*.msgpack" -o -name "*.parquet" -o -name "*.h5" -o -name "*.bin" \) -delete 2>/dev/null || true \
 && (find /usr/local/lib/python3.10 -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete 2>/dev/null || true) \
 && (find /usr/local/lib/python3.10 -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true) \
 && rm -rf /root/.cache /tmp/* /var/tmp/*

# Force offline usage at runtime to avoid accidental downloads
ENV TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

# Expose server port
ENV RP_HANDLER=handler \
    RP_NUM_WORKERS=1 \
    RP_PORT=3000 \
    RP_HTTP=1

EXPOSE 3000

# Health check and start
CMD ["python3", "handler.py"]
