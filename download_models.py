"""
Model download utility for Qwen Image Edit with Nunchaku quantization.
This script downloads the necessary model files and can be run during Docker build.
"""
import os
import shutil
from huggingface_hub import snapshot_download, hf_hub_download


def assure_pipeline_files(model_id: str = "Qwen/Qwen-Image-Edit", cache_dir: str = None, use_original_text_encoder: bool = True):
    """
    Download only the small config files from the HF repo.
    Avoids downloading large model weights since we use local Nunchaku files.

    Args:
        model_id: HuggingFace model ID (default: "Qwen/Qwen-Image-Edit")
        cache_dir: Optional cache directory for downloads
        use_original_text_encoder: Whether to download original text encoder or use compact FP8 version
    """
    print(f"Downloading pipeline config files from {model_id}...")

    allow = [
        "model_index.json",
        # schedulers/configs
        "scheduler/*",
        "scheduler/**",
        # VAE
        "vae/*",
        "vae/**",
        # tokenizers
        "tokenizer/*",
        "tokenizer/**",
        # processors / feature extractors
        "image_processor/*",
        "image_processor/**",
        "processor/*",
        "processor/**",
        # misc small configs
        "*.json",
        "*.txt",
    ]

    if use_original_text_encoder:
        allow.extend([
            # text encoder configs and weights
            "text_encoder/*",
            "text_encoder/**",
        ])
    else:
        allow.extend([
            # text encoder configs ONLY (skip original huge shards and index)
            "text_encoder/config.json",
            "text_encoder/generation_config.json",
        ])

    # Intentionally DO NOT fetch transformer/unet weights; we use Nunchaku quantized files
    snapshot_download(
        repo_id=model_id,
        repo_type="model",
        allow_patterns=allow,
        local_dir_use_symlinks=False,
        cache_dir=cache_dir,
    )
    print(f"Pipeline config files downloaded successfully")


def download_nunchaku_transformer(
    output_path: str,
    repo_id: str = "nunchaku-tech/nunchaku-qwen-image-edit",
    rank: int = 128,
    lighting_steps: str = "8"
):
    """
    Download Nunchaku quantized transformer model.

    Args:
        output_path: Where to save the downloaded file
        repo_id: HuggingFace repo ID
        rank: SVD quantization rank (32, 64, 128)
        lighting_steps: Lightning steps ("4", "8", or "NONE" for original)
    """
    if os.path.exists(output_path):
        print(f"✓ Nunchaku transformer already exists at {output_path}, skipping download")
        return output_path

    print(f"Downloading Nunchaku transformer from {repo_id}...")

    # Determine filename based on rank and lighting
    if lighting_steps == "8":
        filename = f"svdq-int4_r{rank}-qwen-image-edit-lightningv1.0-8steps.safetensors"
    elif lighting_steps == "4":
        filename = f"svdq-int4_r{rank}-qwen-image-edit-lightningv1.0-4steps.safetensors"
    else:
        filename = f"svdq-int4_r{rank}-qwen-image-edit.safetensors"

    print(f"Downloading file: {filename}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=os.path.dirname(output_path),
        local_dir_use_symlinks=False
    )

    # Rename to consistent name
    final_path = os.path.join(os.path.dirname(output_path), "transformer.safetensors")
    if downloaded_path != final_path:
        shutil.move(downloaded_path, final_path)

    print(f"✓ Nunchaku transformer downloaded to {final_path}")
    return final_path


def download_compact_text_encoder(
    output_path: str,
    repo_id: str = "Comfy-Org/Qwen-Image_ComfyUI",
    filename: str = "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"
):
    """
    Download compact FP8 quantized text encoder as an alternative to the original.

    Args:
        output_path: Where to save the downloaded file
        repo_id: HuggingFace repo ID
        filename: Filename in the repo
    """
    if os.path.exists(output_path):
        print(f"✓ Compact text encoder already exists at {output_path}, skipping download")
        return output_path

    print(f"Downloading compact text encoder from {repo_id}/{filename}...")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=os.path.dirname(output_path),
        local_dir_use_symlinks=False
    )

    print(f"✓ Compact text encoder downloaded to {downloaded_path}")
    return downloaded_path


def download_lora_weights(
    output_path: str,
    repo_id: str = "lightx2v/Qwen-Image-Lightning",
    filename: str = "Qwen-Image-Lightning-8steps-V2.0.safetensors"
):
    """
    Download LoRA weights for faster inference (if using LoRA-capable pipeline).

    Args:
        output_path: Where to save the downloaded file
        repo_id: HuggingFace repo ID
        filename: Filename in the repo
    """
    if os.path.exists(output_path):
        print(f"✓ LoRA weights already exist at {output_path}, skipping download")
        return output_path

    print(f"Downloading LoRA weights from {repo_id}/{filename}...")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=os.path.dirname(output_path),
        local_dir_use_symlinks=False
    )

    print(f"✓ LoRA weights downloaded to {downloaded_path}")
    return downloaded_path


def compress_safetensors(file_path: str):
    """
    Compress safetensors files using zstd compression to save space.
    """
    if not os.path.exists(file_path) or not file_path.endswith('.safetensors'):
        return

    try:
        from safetensors import safe_open, safe_save
        import tempfile

        print(f"Compressing {file_path}...")
        
        tensors = {}
        with safe_open(file_path, framework="pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        
        tmp = tempfile.mktemp(suffix=".safetensors")
        safe_save(tensors, tmp, metadata={"compression": "zstd"})
        os.replace(tmp, file_path)
        
        print(f"✓ Compressed {file_path}")
    except Exception as e:
        print(f"Warning: Failed to compress {file_path}: {e}")


def download_all_models(
    models_dir: str = "/models", 
    rank: int = 128, 
    lighting_steps: str = "8",
    use_original_text_encoder: bool = True
):
    """
    Download all required models for the Qwen Image Edit pipeline with Nunchaku quantization.
    Skips files that already exist (e.g., from local copies or symlinks).
    Always ensures pipeline config files are present.

    Args:
        models_dir: Base directory for storing models
        rank: SVD quantization rank for Nunchaku transformer
        lighting_steps: Lightning steps for faster inference
        use_original_text_encoder: Whether to use original or compact text encoder
    """
    print(f"\n{'='*60}")
    print(f"Ensuring models are present in {models_dir}")
    print(f"Rank: {rank}, Lightning: {lighting_steps}, Original TE: {use_original_text_encoder}")
    print(f"{'='*60}\n")

    # Set HuggingFace cache directories to models_dir
    os.environ["HF_HOME"] = models_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = models_dir
    os.environ["TRANSFORMERS_CACHE"] = models_dir

    # Always download pipeline configs (small files, ensures correct setup)
    print("1. Ensuring pipeline config files...")
    assure_pipeline_files(cache_dir=models_dir, use_original_text_encoder=use_original_text_encoder)

    # Download Nunchaku transformer (skips if exists)
    print("\n2. Checking Nunchaku transformer model...")
    transformer_path = os.path.join(models_dir, "diffusion_models", "transformer.safetensors")
    download_nunchaku_transformer(transformer_path, rank=rank, lighting_steps=lighting_steps)

    # Download compact text encoder if not using original
    if not use_original_text_encoder:
        print("\n3. Checking compact text encoder...")
        te_path = os.path.join(models_dir, "text_encoders", "qwen_2.5_vl_7b_fp8_scaled.safetensors")
        download_compact_text_encoder(te_path)

        # Clean up any conflicting files in pipeline text_encoder directory
        pipeline_dir = os.path.join(models_dir, "models--Qwen--Qwen-Image-Edit")
        te_dir = os.path.join(pipeline_dir, "text_encoder")
        if os.path.exists(te_dir):
            # Remove index file if exists to avoid sharded loading
            idx_file = os.path.join(te_dir, "model.safetensors.index.json")
            if os.path.exists(idx_file):
                os.remove(idx_file)
                print("Removed conflicting index file")
            
            # Remove shard files
            for f in os.listdir(te_dir):
                if f.startswith("model-") and f.endswith(".safetensors"):
                    os.remove(os.path.join(te_dir, f))
                    print(f"Removed conflicting shard: {f}")

    print(f"\n{'='*60}")
    print(f"✓ All models ready!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import sys

    models_dir = sys.argv[1] if len(sys.argv) > 1 else "./models"
    
    # Get configuration from environment variables
    rank = int(os.getenv("RANK", "128"))
    lighting_steps = os.getenv("LIGHTING", "8")
    use_original_te = os.getenv("USE_ORIGINAL_TEXT_ENCODER", "true").lower() in ("true", "1", "yes")

    download_all_models(
        models_dir=models_dir,
        rank=rank,
        lighting_steps=lighting_steps,
        use_original_text_encoder=use_original_te
    )