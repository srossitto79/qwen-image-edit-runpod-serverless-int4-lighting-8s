import base64
import io
import os
from typing import Any, Dict, Optional

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

DEFAULT_STEPS = int(os.getenv("DEFAULT_STEPS", "8"))
DEFAULT_SCALE = float(os.getenv("DEFAULT_SCALE", "4.0"))
DEFAULT_RANK = int(os.getenv("DEFAULT_RANK", "128"))  # informational only; used at build time


# Lazy globals
pipe: Optional[QwenImageEditPipeline] = None


def load_pipeline() -> QwenImageEditPipeline:
    global pipe
    if pipe is not None:
        return pipe

    #device = "cuda" if torch.cuda.is_available() else "cpu"

    transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(TRANSFORMER_PATH)

    # Manually load text encoder to avoid diffusers passing unsupported kwargs
    #te_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # Canonical text encoder path baked into the pipeline directory. During
    # build time we attempt to produce a BitsAndBytes-quantized text encoder
    text_encoder_path = os.path.join(PIPELINE_DIR, "text_encoder")

    # Load the text encoder from the canonical path. Keep trust_remote_code=True
    # to allow repository-provided code to run when loading the model.
    # text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     text_encoder_path,
    #     torch_dtype=te_dtype,
    #     local_files_only=True,
    #     low_cpu_mem_usage=False,
    #     trust_remote_code=True,
    # )

    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        text_encoder_path,
        device_map="auto",
        trust_remote_code=True
    )    


    # # Ensure text encoder params are in the expected dtype when appropriate.
    # # For BitsAndBytes-loaded models, avoid forcing dtype casts which can break
    # # BnB weight wrappers. Only coerce dtype for non-BnB models.
    # try:
    #     text_encoder.to(dtype=te_dtype)
    #     # Some float8 tensors may not convert with a single .to; enforce per-param cast
    #     for p in text_encoder.parameters():
    #         if p.dtype != te_dtype:
    #             p.data = p.data.to(te_dtype)
    #     for b_name, b in text_encoder.named_buffers():
    #         try:
    #             if hasattr(b, 'dtype') and b.dtype != te_dtype:
    #                 # Replace buffer in-place
    #                 setattr(text_encoder, b_name, b.to(te_dtype))
    #         except Exception:
    #             pass
    # except Exception:
    #     pass

    pipe_local = QwenImageEditPipeline.from_pretrained(
        PIPELINE_DIR,
        transformer=transformer,
        text_encoder=text_encoder,
        local_files_only=True,
        trust_remote_code=True,
        device_map="auto",
    )

    # Enable memory optimization based on available GPU memory
    if torch.cuda.is_available():
        gpu_memory_gb = get_gpu_memory()
        if gpu_memory_gb > 18:
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
