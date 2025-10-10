import base64
import io
import os
from typing import Any, Dict, Optional, Tuple

import runpod
from PIL import Image

import torch
from diffusers import QwenImageEditPipeline
from diffusers.utils import load_image

from nunchaku import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_gpu_memory, get_precision
from transformers import Qwen2_5_VLForConditionalGeneration

# Model locations baked into the container at build-time
MODELS_DIR = os.getenv("MODELS_DIR", "/models")
PIPELINE_DIR = os.path.join(MODELS_DIR, "Qwen-Image-Edit")  # diffusers pipeline
TRANSFORMER_PATH = os.path.join(MODELS_DIR, "transformer.safetensors")  # nunchaku quantized

DEFAULT_STEPS = int(os.getenv("DEFAULT_STEPS", "30"))
DEFAULT_SCALE = float(os.getenv("DEFAULT_SCALE", "4.0"))
DEFAULT_RANK = int(os.getenv("DEFAULT_RANK", "32"))  # informational only; used at build time


# Lazy globals
pipe: Optional[QwenImageEditPipeline] = None


def load_pipeline() -> QwenImageEditPipeline:
    global pipe
    if pipe is not None:
        return pipe

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(TRANSFORMER_PATH)

    # Manually load text encoder to avoid diffusers passing unsupported kwargs
    te_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    text_encoder_path = os.path.join(PIPELINE_DIR, "text_encoder")
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        text_encoder_path,
        torch_dtype=te_dtype,
        local_files_only=True,
        low_cpu_mem_usage=False,
    )

    # Ensure text encoder params are in the expected dtype (de-quantize fp8 weights if present)
    try:
        text_encoder.to(dtype=te_dtype)
        # Some float8 tensors may not convert with a single .to; enforce per-param cast
        for p in text_encoder.parameters():
            if p.dtype != te_dtype:
                p.data = p.data.to(te_dtype)
        for b_name, b in text_encoder.named_buffers():
            try:
                if hasattr(b, 'dtype') and b.dtype != te_dtype:
                    # Replace buffer in-place
                    setattr(text_encoder, b_name, b.to(te_dtype))
            except Exception:
                pass
    except Exception:
        pass

    pipe_local = QwenImageEditPipeline.from_pretrained(
        PIPELINE_DIR,
        transformer=transformer,
        text_encoder=text_encoder,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        local_files_only=True,
        low_cpu_mem_usage=False,
    )

    if device == "cuda":
        if get_gpu_memory() > 18:
            pipe_local.enable_model_cpu_offload()
        else:
            transformer.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=1)
            pipe_local._exclude_from_cpu_offload.append("transformer")
            pipe_local.enable_sequential_cpu_offload()
    pipe = pipe_local
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
    prompt = inp.get("prompt")
    negative_prompt = inp.get("negative_prompt", " ")
    num_inference_steps = int(inp.get("num_inference_steps", DEFAULT_STEPS))
    true_cfg_scale = float(inp.get("true_cfg_scale", DEFAULT_SCALE))
    width = inp.get("width")
    height = inp.get("height")

    if not image_in or not prompt:
        return {"error": "Missing required fields: image, prompt"}

    pipe = load_pipeline()

    image = read_image(image_in)

    kwargs = {
        "image": image,
        "prompt": prompt,
        "true_cfg_scale": true_cfg_scale,
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_inference_steps,
    }
    if width:
        kwargs["width"] = int(width)
    if height:
        kwargs["height"] = int(height)

    out = pipe(**kwargs)
    out_img = out.images[0]

    b64 = encode_image(out_img)
    return {"image_base64": b64}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
