# Qwen-Image-Edit Serverless on Runpod (with Nunchaku)

This repository packages a Runpod Serverless worker that performs image editing using Qwen-Image-Edit with Nunchaku-quantized transformer weights embedded inside the Docker image (<= 20GB).

- Model: Qwen/Qwen-Image-Edit (Diffusers pipeline) + Nunchaku quantized transformer (INT4/FP4, rank selectable) + Lighting 8steps lora
- Runtime: Python worker using Runpod serverless
- Input: image (URL or base64), prompt, negative_prompt (optional), steps, scale
- Output: base64-encoded edited image

## Inputs

- image: string
  - Either an http(s) URL to an image, or a data URL / base64-encoded PNG/JPEG.
- prompt: string (required)
- negative_prompt: string (optional; default " ")
- num_inference_steps: int (optional; default 8)
- true_cfg_scale: float (optional; default 4.0)
- width: int (optional)
- height: int (optional)

## Quick local run (Docker)

You can run locally using uvicorn to sanity-check the handler:

1. Build the image (downloads and embeds weights). You can control these build args:

   - RANK: 32 or 128 (default 128)
   - LIGHTING: NONE, 4, or 8 (default NONE)
   - USE_ORIGINAL_TEXT_ENCODER: true/false or 1/0 or Y/N (default true = use original sharded encoder)

```cmd
docker build -t qwen-image-edit-serverless .
```

   Examples:

   - Use original sharded text encoder (default):
     ```cmd
     docker build -t qwen-image-edit-serverless:orig-te .
     ```

   - Use compact single-file text encoder instead:
     ```cmd
     docker build --build-arg USE_ORIGINAL_TEXT_ENCODER=false -t qwen-image-edit-serverless:compact-te .
     ```

   - Set rank and lighting variant (e.g., RANK=32, LIGHTING=8):
     ```cmd
     docker build --build-arg RANK=32 --build-arg LIGHTING=8 -t qwen-image-edit-serverless:r32-l8 .
     ```

2. Run container locally with a simple HTTP API (FastAPI) by setting RUNPOD_LOCAL_TEST=1:

```cmd
docker run --gpus all -p 3000:3000 -e RUNPOD_LOCAL_TEST=1 qwen-image-edit-serverless
```

3. Send a test job (replace URL as needed):

```cmd
curl -X POST http://localhost:3000/ -H "Content-Type: application/json" -d "{
  \"input\": {
    \"image\": \"https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/neon_sign.png\",
    \"prompt\": \"change the text to read 'Hello Runpod'\",
    \"num_inference_steps\": 20,
    \"true_cfg_scale\": 3.5
  }
}"
```

The response will contain a base64 field with the edited image.

## Build arguments at a glance

- RANK: 32 or 128 (default 128)
- LIGHTING: NONE, 4, or 8 (default NONE). Selects the corresponding Lightning 4/8 step transformer variant.
- USE_ORIGINAL_TEXT_ENCODER: true/false or 1/0 or Y/N (default true). When true, downloads the original sharded text encoder from Qwen/Qwen-Image-Edit. When false, downloads a compact single-file encoder from Comfy-Org/Qwen-Image_ComfyUI and removes any shard remnants.

Examples (Windows cmd):

```cmd
:: default (original sharded encoder)
docker build -t qwen-image-edit-serverless .

:: compact single-file encoder
docker build --build-arg USE_ORIGINAL_TEXT_ENCODER=false -t qwen-image-edit-serverless:compact-te .

:: rank 32 + lightning 8-step
docker build --build-arg RANK=32 --build-arg LIGHTING=8 -t qwen-image-edit-serverless:r32-l8 .
```

Resulting transformer filenames baked in the image:
- NONE: svdq-int4_r{RANK}-qwen-image-edit.safetensors
- 4: svdq-int4_r{RANK}-qwen-image-edit-lightningv1.0-4steps.safetensors
- 8: svdq-int4_r{RANK}-qwen-image-edit-lightningv1.0-8steps.safetensors

## Publishing to Runpod Hub

- Ensure `.runpod/hub.json` and `.runpod/tests.json` are set.
- Tag a GitHub release. The Hub indexes releases, not commits.

## Notes

- We use Nunchaku to load a quantized transformer from Hugging Face at build time and bake it into `/models` inside the image.
- The container size budget is 20GB; the selected quantized artifacts fit within this budget.
- CUDA 12.x base image with PyTorch is used for broad GPU support.

## Local test script

A small helper script `test_local_http_endpoint.py` posts a request to the local API (when RUNPOD_LOCAL_TEST=1 is used at container start):

1) Ensure the container is running locally:

```cmd
docker run --gpus all -p 3000:3000 -e RUNPOD_LOCAL_TEST=1 qwen-image-edit-serverless
```

2) On your host, install Python requests if needed and run the test:

```cmd
python -m pip install requests
python test_local_http_endpoint.py
```

The script reads `test_input.jpg` and writes `result.png`.

## Acknowledgments

This project stands on the shoulders of fantastic open-source work:
- Qwen-Image-Edit and the broader Qwen models by Alibaba Cloud (Qwen team)
- Hugging Face Transformers and Diffusers
- Nunchaku for efficient quantized transformers
- Runpod Serverless
- FastAPI and Uvicorn
- PyTorch and the broader ecosystem

## Author / Contact

Made with passion by Salvatore Rossitto.