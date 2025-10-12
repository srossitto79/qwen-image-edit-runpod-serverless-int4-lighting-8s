# Base image with CUDA 12.4 and PyTorch compatible drivers
FROM runpod/pytorch:0.7.2-dev-cu1241-torch260-ubuntu2204

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FORCE_CUDA=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libglib2.0-0 libsm6 libxext6 libxrender1 git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (leverage layer cache)
WORKDIR /app
COPY requirements.txt ./
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
 && python3 -m pip install --no-cache-dir --no-compile -r requirements.txt

# Install Nunchaku prebuilt wheel compatible with Torch 2.6 and Python 3.10
# Note: adjust the URL if Nunchaku updates the artifact naming.
RUN python3 -m pip install --no-cache-dir --no-deps \
    https://github.com/nunchaku-tech/nunchaku/releases/download/v1.0.1/nunchaku-1.0.1+torch2.6-cp310-cp310-linux_x86_64.whl

# Prepare model directories
ENV MODELS_DIR=/models
RUN mkdir -p ${MODELS_DIR}

# --- Embed model weights into the image ---
# 1) Download Qwen/Qwen-Image-Edit pipeline locally (Diffusers format)
# 2) Download Nunchaku quantized transformer safetensors file
# We select INT4 rank-32 to keep size small; adjust if needed.
ARG RANK=128
ARG LIGHTING=8
ARG USE_ORIGINAL_TEXT_ENCODER="true"

# Make the build-args available to the Python step below
ENV RANK=${RANK}
ENV LIGHTING=${LIGHTING}
ENV USE_ORIGINAL_TEXT_ENCODER=${USE_ORIGINAL_TEXT_ENCODER}

# Use huggingface_hub to download to a local dir under /models (minimize files pulled)
RUN python3 - <<'PY'
import os
import shutil
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
# Download nunchaku quantized transformer (derived only from RANK and LIGHTING)
trans_path = os.path.join(models_dir, 'transformer.safetensors')
rank_env = os.environ.get('RANK', '128')
try:
    rank = int(rank_env)
except Exception:
    rank = 128
lighting = os.environ.get('LIGHTING', 'NONE').strip().lower()
if lighting == '8':
    filename = f"svdq-int4_r{rank}-qwen-image-edit-lightningv1.0-8steps.safetensors"
elif lighting == '4':
    filename = f"svdq-int4_r{rank}-qwen-image-edit-lightningv1.0-4steps.safetensors"
else:
    filename = f"svdq-int4_r{rank}-qwen-image-edit.safetensors"
repo = 'nunchaku-tech/nunchaku-qwen-image-edit'
fp = hf_hub_download(repo_id=repo, filename=filename)
shutil.copy(fp, trans_path)
print('Downloaded pipeline to', pipe_dir)
print('Downloaded transformer to', trans_path)
te_dir = os.path.join(pipe_dir, 'text_encoder')
os.makedirs(te_dir, exist_ok=True)
use_original = os.environ.get('USE_ORIGINAL_TEXT_ENCODER', 'false').lower() in ('1','true','yes','y')
if use_original:
    # Fetch original sharded text encoder (index + shard files) into pipe_dir
    snapshot_download(
        repo_id='Qwen/Qwen-Image-Edit',
        local_dir=pipe_dir,
        allow_patterns=[
            'text_encoder/model.safetensors.index.json',
            'text_encoder/model-*.safetensors',
        ],
    )
    # If a single-file encoder exists from a previous build layer, remove it to avoid ambiguity
    single = os.path.join(te_dir, 'model.safetensors')
    if os.path.exists(single):
        try:
            os.remove(single)
            print('Removed single-file text encoder:', single)
        except Exception as e:
            print('Warn: could not remove single-file encoder:', e)
    print('Using original sharded text encoder under', te_dir)
else:
    # Choose a compact single-file text encoder from Comfy-Org (16-bit by default)
    te_repo = 'Comfy-Org/Qwen-Image_ComfyUI'
    te_file = 'split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors'
    te_src = hf_hub_download(repo_id=te_repo, filename=te_file)
    shutil.copy(te_src, os.path.join(te_dir, 'model.safetensors'))
    # Ensure no stale index references shards; remove if exists
    idx = os.path.join(te_dir, 'model.safetensors.index.json')
    if os.path.exists(idx):
        try:
            os.remove(idx)
            print('Removed stale index file:', idx)
        except Exception as _e:
            print('Warn: could not remove index file:', _e)
    # Remove any stray shard files to avoid accidental loading of sharded weights
    for _n in os.listdir(te_dir):
        if _n.startswith('model.safetensors-') and _n.endswith('.safetensors'):
            try:
                os.remove(os.path.join(te_dir, _n))
                print('Removed stale shard:', _n)
            except Exception as _e:
                print('Warn: could not remove shard', _n, ':', _e)
    print('Downloaded compact single-file text encoder to', os.path.join(te_dir, 'model.safetensors'))
PY

RUN python3 - <<'PY'
from safetensors import safe_open, safe_save
import os, torch, tempfile

def compress(path):
    if not os.path.exists(path): return
    tensors = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    tmp = tempfile.mktemp(suffix=".safetensors")
    safe_save(tensors, tmp, metadata={"compression": "zstd"})
    os.replace(tmp, path)
    print("Compressed", path)

for root, _, files in os.walk("/models"):
    for f in files:
        if f.endswith(".safetensors"):
            compress(os.path.join(root, f))
PY

RUN rm -rf /root/.cache/pip \
 && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
 && find /models -type d -name ".git" -exec rm -rf {} + \
 && find /models -type f \( -name "*.msgpack" -o -name "*.parquet" -o -name "*.h5" -o -name "*.bin" \) -delete \
 && (find /usr/local/lib/python3.10 -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete 2>/dev/null || true) \
 && (find /usr/local/lib/python3.10 -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true) \
 && strip --strip-unneeded $(find /usr/local/lib -type f -name "*.so" 2>/dev/null || true) || true \
 && rm -rf /root/.cache /tmp/* /var/tmp/*

# Force offline usage at runtime to avoid accidental downloads
ENV TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

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
