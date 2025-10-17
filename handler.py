import base64
import io
import os
import time
import warnings
from typing import Any, Dict, Optional

# Ensure HF cache dirs are configured BEFORE importing HF libs
_MODELS_DIR_BOOT = os.getenv("MODELS_DIR", "./models")
os.environ.setdefault("HF_HOME", _MODELS_DIR_BOOT)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", _MODELS_DIR_BOOT)
os.environ.setdefault("DIFFUSERS_CACHE", _MODELS_DIR_BOOT)

import runpod
from PIL import Image

import torch
from diffusers import QwenImageEditPlusPipeline
from diffusers.utils import load_image

from nunchaku import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_gpu_memory, get_precision
from transformers import Qwen2_5_VLForConditionalGeneration, AutoConfig

warnings.filterwarnings("ignore", message=".*Some weights of the model checkpoint.*were not used.*")


def calculate_optimal_blocks_on_gpu(available_memory_gb: float, transformer_model=None) -> int:
    """
    Calculate optimal number of transformer blocks to keep on GPU based on available memory.
    
    Args:
        available_memory_gb: Available GPU memory in GB
        transformer_model: The transformer model to analyze (optional, for getting actual block count)
    
    Returns:
        Optimal number of blocks to keep on GPU
    """
    # Try to get actual number of blocks from the model
    total_blocks = 64  # Default for Qwen Image Edit
    if transformer_model is not None:
        try:
            # Different ways the blocks might be accessible
            if hasattr(transformer_model, 'transformer_blocks'):
                total_blocks = len(transformer_model.transformer_blocks)
            elif hasattr(transformer_model, 'blocks'):
                total_blocks = len(transformer_model.blocks)
            elif hasattr(transformer_model, 'config') and hasattr(transformer_model.config, 'num_layers'):
                total_blocks = transformer_model.config.num_layers
            print(f"Detected {total_blocks} transformer blocks in model")
        except Exception as e:
            print(f"Could not detect block count, using default: {e}")
    
    # Reserve memory for other components (VAE, text encoder, intermediate tensors, etc.)
    reserved_memory_gb = 4.0  # Conservative estimate
    
    # Memory available for transformer blocks
    available_for_blocks = max(0, available_memory_gb - reserved_memory_gb)
    
    # Estimate memory per block based on model type
    # INT4 quantized models use significantly less memory than FP16/BF16
    memory_per_block_gb = 0.15  # ~150MB per block for INT4 quantized model
    
    # Calculate how many blocks can fit
    max_blocks_that_fit = int(available_for_blocks / memory_per_block_gb)
    
    # Don't exceed total blocks in model
    optimal_blocks = min(max_blocks_that_fit, total_blocks)
    
    # Ensure at least 1 block on GPU for performance
    optimal_blocks = max(1, optimal_blocks)
    
    print(f"GPU Memory: {available_memory_gb:.1f}GB, Reserved: {reserved_memory_gb:.1f}GB")
    print(f"Available for blocks: {available_for_blocks:.1f}GB")
    print(f"Estimated blocks that fit: {max_blocks_that_fit}, Total blocks: {total_blocks}, Using: {optimal_blocks}")
    
    return optimal_blocks

# Model locations baked into the container at build-time
MODELS_DIR = os.getenv("MODELS_DIR", "./models")
MODEL_ID = "Qwen/Qwen-Image-Edit"
TRANSFORMER_PATH = os.path.join(MODELS_DIR, "diffusion_models", "transformer.safetensors")  # nunchaku quantized

# Configuration options
USE_ORIGINAL_TEXT_ENCODER = os.getenv("USE_ORIGINAL_TEXT_ENCODER", "true").lower() in ("true", "1", "yes")
COMPACT_TE_PATH = os.path.join(MODELS_DIR, "text_encoders", "qwen_2.5_vl_7b_fp8_scaled.safetensors")

DEFAULT_STEPS = int(os.getenv("DEFAULT_STEPS", "8"))
DEFAULT_SCALE = float(os.getenv("DEFAULT_SCALE", "4.0"))
DEFAULT_RANK = int(os.getenv("DEFAULT_RANK", "128"))  # informational only; used at build time


if not os.path.exists(TRANSFORMER_PATH):
    raise FileNotFoundError(f"Transformer model file not found at {TRANSFORMER_PATH}")

# Lazy globals
pipe: Optional[QwenImageEditPlusPipeline] = None


def load_text_encoder(checkpoint_path: str, model_source: str, torch_dtype: torch.dtype = torch.bfloat16) -> Qwen2_5_VLForConditionalGeneration:
    """
    Load text encoder from a safetensors file or directory.
    Optimized for both original sharded weights and compact FP8 quantized safetensors files.
    """
    # If it's a directory, use standard from_pretrained
    if os.path.isdir(checkpoint_path):
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            checkpoint_path,
            torch_dtype=torch_dtype,
            local_files_only=True,
            low_cpu_mem_usage=False,
        )

    # For single safetensors file: load directly with minimal overhead
    from safetensors.torch import load_file

    print("Loading config...")
    config = AutoConfig.from_pretrained(
        model_source,
        trust_remote_code=True,
        subfolder="text_encoder",
        local_files_only=True,
        cache_dir=MODELS_DIR,
    )

    print("Loading weights from safetensors...")
    state_dict = load_file(checkpoint_path, device="cpu")
    
    print("Instantiating text encoder from config with provided state_dict...")
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            None,
            config=config,
            state_dict=state_dict,
            local_files_only=True,
            cache_dir=MODELS_DIR,
        )
    except TypeError:
        # Fallback for older transformers: manual init + load_state_dict
        model = Qwen2_5_VLForConditionalGeneration(config)
        model.load_state_dict(state_dict, strict=False)
    
    # Ensure text encoder data type
    if torch_dtype is not None:
        model.to(dtype=torch_dtype)

        # Some float8 tensors may not convert with a single .to; enforce per-param cast
        for p in model.parameters():
            if p.dtype != torch_dtype:
                p.data = p.data.to(torch_dtype)
        for b_name, b in model.named_buffers():
            try:
                if hasattr(b, 'dtype') and b.dtype != torch_dtype:
                    # Replace buffer in-place
                    setattr(model, b_name, b.to(torch_dtype))
            except Exception:
                pass

    print("Text encoder loaded successfully!")
    return model


def load_pipeline(target_dtype: torch.dtype = torch.bfloat16) -> QwenImageEditPlusPipeline:
    global pipe
    if pipe is not None:
        return pipe

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading Nunchaku transformer model...")
    transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(TRANSFORMER_PATH)

    # Load text encoder based on configuration
    if USE_ORIGINAL_TEXT_ENCODER or not os.path.exists(COMPACT_TE_PATH):
        print("Loading original text encoder model...")
        print("Finalizing pipeline load...")
        pipe_local = QwenImageEditPlusPipeline.from_pretrained(
            MODEL_ID,
            transformer=transformer,
            torch_dtype=target_dtype if device == "cuda" else torch.float32,
            local_files_only=True,
            low_cpu_mem_usage=False,
            cache_dir=MODELS_DIR
        )
    else:
        print("Loading compact FP8 text encoder model...")
        text_encoder = load_text_encoder(COMPACT_TE_PATH, MODEL_ID, torch_dtype=target_dtype)

        print("Finalizing pipeline load...")
        pipe_local = QwenImageEditPlusPipeline.from_pretrained(
            MODEL_ID,
            transformer=transformer,
            text_encoder=text_encoder,
            torch_dtype=target_dtype if device == "cuda" else torch.float32,
            local_files_only=True,
            low_cpu_mem_usage=False,
            cache_dir=MODELS_DIR
        )

    # # Configure memory management based on available GPU memory
    # if device == "cuda":
    #     if get_gpu_memory() > 18:
    #         print("Configuring standard offload for high-memory GPU...")
    #         pipe_local.enable_model_cpu_offload()
    #     else:
    #         print("Configuring Nunchaku offload for low-memory GPU...")
    #         transformer.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=1)
    #         pipe_local._exclude_from_cpu_offload.append("transformer")
    #         pipe_local.enable_sequential_cpu_offload()
    if device == "cuda":
        gpu_memory = get_gpu_memory()
        if gpu_memory > 18:
            print("Configuring standard offload for high-memory GPU...")
            pipe_local.to("cuda")
        else:
            # Calculate optimal number of blocks based on available GPU memory
            optimal_blocks = calculate_optimal_blocks_on_gpu(gpu_memory, transformer)
            print(f"Configuring Nunchaku offload for low-memory GPU (using {optimal_blocks} blocks)...")
            transformer.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=optimal_blocks)
            pipe_local._exclude_from_cpu_offload.append("transformer")
            pipe_local.enable_sequential_cpu_offload()
            #pipe_local.enable_model_cpu_offload()

    pipe_local.set_progress_bar_config(disable=None)
    pipe = pipe_local

    # Print dtype information for debugging
    print(f"Transformer dType: {pipe.transformer.dtype if hasattr(pipe.transformer, 'dtype') else 'N/A'}")
    print(f"Text encoder dType: {pipe.text_encoder.dtype}")
    print(f"VAE dType: {pipe.vae.dtype}")

    return pipe


def read_image(source: str) -> Image.Image:
    # If it's a URL, let diffusers util load it; supports http(s)
    if source.startswith("http://") or source.startswith("https://"):
        img = load_image(source)
        return img.convert("RGB")
    # If it's base64 (optionally data URL)
    if source.startswith("data:image"):
        base64_str = source.split(",", 1)[1]
    else:
        base64_str = source
    try:
        data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(data)).convert("RGB")
        max_pixels = 1024 * 1024
        if img.width * img.height > max_pixels:
            img.thumbnail((1024, 1024), Image.LANCZOS)
        return img
    except Exception as e:
        raise ValueError(f"Failed to decode image input: {e}")


def encode_image(img: Image.Image, format: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=format)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# Runpod handler

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    inp = job.get("input") or {}
    image_in = inp.get("image")
    images_in = inp.get("images")
    prompt = inp.get("prompt")
    negative_prompt = inp.get("negative_prompt", " ")
    num_inference_steps = int(inp.get("num_inference_steps", DEFAULT_STEPS))
    true_cfg_scale = float(inp.get("true_cfg_scale", DEFAULT_SCALE))
    width = inp.get("width")
    height = inp.get("height")

    missing_fields = []
    
    if not prompt:
        missing_fields.append("prompt")

    if not image_in and not images_in:
        missing_fields.append("image or images")

    if missing_fields:  
        return {"error": f"Missing required fields: {', '.join(missing_fields)}"}
    
    start_time = time.time()

    print("Loading pipeline...")
    pipe = load_pipeline(target_dtype=torch.bfloat16)

    images = []
    print("Reading input image...")
    if image_in:
        images.append(read_image(image_in))

    if images_in and isinstance(images_in, list) and len(images_in) > 0:
        for img_src in images_in:
            images.append(read_image(img_src))            

    kwargs = {
        "image": images,
        "prompt": prompt,
        "true_cfg_scale": true_cfg_scale,
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_inference_steps,
    }
    
    if width:
        kwargs["width"] = int(width)
    if height:
        kwargs["height"] = int(height)

    print(f"Pipeline loading time: {time.time() - start_time:.2f} seconds")
    print("Generating image...")
    
    torch.cuda.empty_cache()

    out = pipe(**kwargs)
    out_img = out.images[0]

    b64 = encode_image(out_img)
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")
    
    return {"image_base64": b64}


if __name__ == "__main__":
    if os.getenv("RUNPOD_LOCAL_TEST"):
        from fastapi import FastAPI
        import uvicorn

        app = FastAPI()

        @app.post("/")
        async def run_job(job: dict):
            return handler(job)

        port = int(os.getenv("RP_PORT", "3000"))
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        # Start runpod serverless handler
        runpod.serverless.start({"handler": handler})
