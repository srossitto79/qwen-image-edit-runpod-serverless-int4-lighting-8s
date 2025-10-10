# Qwen-Image-Edit Serverless on Runpod (with Nunchaku)

This repository packages a Runpod Serverless worker that performs image editing using Qwen-Image-Edit with Nunchaku-quantized transformer weights embedded inside the Docker image (<= 20GB).

- Model: Qwen/Qwen-Image-Edit (Diffusers pipeline) + Nunchaku quantized transformer (INT4/FP4, rank selectable)
- Runtime: Python worker using Runpod serverless
- Input: image (URL or base64), prompt, negative_prompt (optional), steps, scale
- Output: base64-encoded edited image

## Inputs

- image: string
  - Either an http(s) URL to an image, or a data URL / base64-encoded PNG/JPEG.
- prompt: string (required)
- negative_prompt: string (optional; default " ")
- num_inference_steps: int (optional; default 30)
- true_cfg_scale: float (optional; default 4.0)
- width: int (optional)
- height: int (optional)

## Quick local run

You can run locally using uvicorn to sanity-check the handler:

1. Build the image (downloads and embeds weights):

```cmd
docker build -t qwen-image-edit-serverless .
```

2. Run container locally:

```cmd
docker run --gpus all -p 3000:3000 -e RP_HANDLER=handler -e RP_NUM_WORKERS=1 qwen-image-edit-serverless
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

## Publishing to Runpod Hub

- Ensure `.runpod/hub.json` and `.runpod/tests.json` are set.
- Tag a GitHub release. The Hub indexes releases, not commits.

## Notes

- We use Nunchaku to load a quantized transformer from Hugging Face at build time and bake it into `/models` inside the image.
- The container size budget is 20GB; the selected quantized artifacts fit within this budget.
- CUDA 12.x base image with PyTorch is used for broad GPU support.
