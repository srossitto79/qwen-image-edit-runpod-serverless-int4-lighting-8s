# Base image with CUDA 12.4 and PyTorch compatible drivers
FROM runpod/pytorch:0.7.2-dev-cu1241-torch260-ubuntu2204

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FORCE_CUDA=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget ca-certificates libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (leverage layer cache)
WORKDIR /app
COPY requirements.txt ./
RUN python3 -m pip install --upgrade pip setuptools wheel \
 && python3 -m pip install -r requirements.txt

# Install Nunchaku prebuilt wheel compatible with Torch 2.6 and Python 3.10
# Note: adjust the URL if Nunchaku updates the artifact naming.
RUN python3 -m pip install \
    https://github.com/nunchaku-tech/nunchaku/releases/download/v1.0.1/nunchaku-1.0.1+torch2.6-cp310-cp310-linux_x86_64.whl

RUN pip install --upgrade runpod

# Prepare model directories
ENV MODELS_DIR=/models
RUN mkdir -p ${MODELS_DIR}

# --- Embed model weights into the image ---
# 1) Download Qwen/Qwen-Image-Edit pipeline locally (Diffusers format)
# 2) Download Nunchaku quantized transformer safetensors file
# We select INT4 rank-32 to keep size small; adjust if needed.
ARG RANK=128
ARG NUNCHAKU_REPO="nunchaku-tech/nunchaku-qwen-image-edit"
ARG TRANSFORMER_FILE="svdq-int4_r${RANK}-qwen-image-edit-lightningv1.0-8steps.safetensors"
#ARG TRANSFORMER_FILE="svdq-int4_r${RANK}-qwen-image-edit.safetensors"

# Use huggingface_hub to download to a local dir under /models (minimize files pulled)
RUN python3 - <<'PY'
import os
from huggingface_hub import snapshot_download, hf_hub_download
models_dir = os.environ.get('MODELS_DIR','/models')
# Download diffusers pipeline snapshot
pipe_dir = os.path.join(models_dir, 'Qwen-Image-Edit')
if not os.path.exists(pipe_dir):
    allow = [
        'model_index.json',
        # schedulers/configs
        'scheduler/*', 'scheduler/**',
        # VAE
        'vae/*', 'vae/**',
    # text encoder configs ONLY (skip original huge shards and index)
    'text_encoder/config.json', 'text_encoder/generation_config.json',
        # tokenizers
        'tokenizer/*', 'tokenizer/**',
        # processors / feature extractors
        'image_processor/*', 'image_processor/**',
        'processor/*', 'processor/**',
        # misc small configs
        '*.json', '*.txt'
    ]
    # Intentionally DO NOT fetch transformer/unet weights; we use Nunchaku quantized transformer.
    snapshot_download(
        repo_id='Qwen/Qwen-Image-Edit',
        local_dir=pipe_dir,
        allow_patterns=allow,
    )
# Download nunchaku quantized transformer
trans_path = os.path.join(models_dir, 'transformer.safetensors')
repo = os.environ.get('NUNCHAKU_REPO','nunchaku-tech/nunchaku-qwen-image-edit')
filename = os.environ.get('TRANSFORMER_FILE','svdq-int4_r128-qwen-image-edit.safetensors')
fp = hf_hub_download(repo_id=repo, filename=filename)
import shutil; shutil.copy(fp, trans_path)
print('Downloaded pipeline to', pipe_dir)
print('Downloaded transformer to', trans_path)
te_dir = os.path.join(pipe_dir, 'text_encoder')
os.makedirs(te_dir, exist_ok=True)
# Choose a compact single-file text encoder from Comfy-Org (16-bit by default)
te_repo = os.environ.get('TE_REPO', 'Comfy-Org/Qwen-Image_ComfyUI')
te_file = os.environ.get('TE_FILE', 'split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors')
te_src = hf_hub_download(repo_id=te_repo, filename=te_file)
import shutil as _shutil; _shutil.copy(te_src, os.path.join(te_dir, 'model.safetensors'))
# Ensure no stale index references shards; remove if exists
idx = os.path.join(te_dir, 'model.safetensors.index.json')
try:
    if os.path.exists(idx):
        os.remove(idx)
        print('Removed stale index file:', idx)
except Exception as _e:
    print('Warn: could not remove index file:', _e)
print('Downloaded text encoder to', os.path.join(te_dir, 'model.safetensors'))
PY

# Repack large text encoders in FP16 to reduce on-disk size
# FP16 repack no longer needed; we install a compact, single-file text encoder directly.

# Copy source
COPY handler.py ./
COPY .runpod ./
COPY README.md ./
COPY .dockerignore ./

# Expose server port
ENV RP_HANDLER=handler \
    RP_NUM_WORKERS=1 \
    RP_PORT=3000 \
    RP_HTTP=1
EXPOSE 3000

# Health / start
CMD ["python3", "handler.py"]
