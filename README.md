# Qwen-Image-Edit Serverless on Runpod (with Nunchaku) - Enhanced Version

This repository packages a Runpod Serverless worker that performs image editing using Qwen-Image-Edit with Nunchaku-quantized transformer weights. This enhanced version features improved organization, automated model downloading, and flexible configuration.

## Key Improvements

- **Organized Model Structure**: Models are now organized in dedicated folders (`diffusion_models/`, `text_encoders/`, `loras/`)
- **Automated Downloads**: `download_models.py` script handles model downloading with smart caching
- **Enhanced Dockerfile**: Better layer caching, compression, and build optimization
- **Flexible Configuration**: Environment-based configuration for different model variants
- **Improved Error Handling**: Better logging and error reporting
- **Performance Monitoring**: Built-in timing and memory management

## Features

- **Model**: Qwen/Qwen-Image-Edit (Diffusers pipeline) + Nunchaku quantized transformer (INT4, configurable rank)
- **Text Encoder Options**: Original full-precision or compact FP8 quantized
- **Lightning Support**: 4-step and 8-step variants for faster inference
- **Memory Management**: Intelligent GPU memory management with offloading
- **Runtime**: Python worker using Runpod serverless with FastAPI local testing

## API Reference

### Input Parameters

#### Image Input (choose one)
- `image`: Single image input (URL or base64) - **optional if `images` provided**
- `images`: List of images (URLs or base64 strings) for multi-image composing - **optional if `image` provided**
  - Use this parameter for multi-image composition workflows
  - Note: At least one of `image` or `images` must be provided

#### Editing Parameters
- `prompt`: Edit instruction - **required**
- `negative_prompt`: Negative prompt (default: " ")
- `num_inference_steps`: Number of inference steps (default: 8, max: 50)
- `true_cfg_scale`: CFG scale for guidance strength (default: 4.0)
- `width`: Output width in pixels (optional, default: 1024)
- `height`: Output height in pixels (optional, default: 1024)

### Response

```json
{
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

### Example Requests

**Single Image Edit:**
```json
{
  "input": {
    "image": "https://example.com/image.jpg",
    "prompt": "make the sky purple",
    "num_inference_steps": 8,
    "true_cfg_scale": 4.0
  }
}
```

**Multi-Image Composition:**
```json
{
  "input": {
    "images": [
      "https://example.com/image1.jpg",
      "https://example.com/image2.jpg",
      "base64_encoded_image_data_here"
    ],
    "prompt": "blend the images together",
    "num_inference_steps": 8
  }
}
```

## Quick Start

### 1. Build the Image

Basic build with defaults (Rank 128, Lightning 8-step, original text encoder):
```cmd
docker build -t qwen-image-edit-nunchaku .
```

Optimized build with compact text encoder:
```cmd
docker build --build-arg USE_ORIGINAL_TEXT_ENCODER=false -t qwen-image-edit-nunchaku:compact .
```

### 2. Local Testing

Run with FastAPI for local testing:
```cmd
docker run --gpus all -p 3000:3000 -e RUNPOD_LOCAL_TEST=1 qwen-image-edit-nunchaku
```

Without docker (on Windows)
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
set RUNPOD_LOCAL_TEST=1 && python handler.py
```

Without docker (on linux)
```cmd
python -m venv .venv
source .venv\Scripts\activate
pip install -r requirements.txt
export RUNPOD_LOCAL_TEST=1 && python handler.py
```

### 3. Test the API

**Single image example:**
```cmd
curl -X POST http://localhost:3000/ -H "Content-Type: application/json" -d "{
  \"input\": {
    \"image\": \"https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/neon_sign.png\",
    \"prompt\": \"change the text to read 'Hello Runpod'\",
    \"num_inference_steps\": 8,
    \"true_cfg_scale\": 4.0
  }
}"
```

**Multi-image composition example:**
```cmd
curl -X POST http://localhost:3000/ -H "Content-Type: application/json" -d "{
  \"input\": {
    \"images\": [
      \"https://example.com/image1.jpg\",
      \"https://example.com/image2.jpg\"
    ],
    \"prompt\": \"blend these images together\",
    \"num_inference_steps\": 8
  }
}"
```

## Configuration Options

### Build Arguments

- `RANK`: SVD quantization rank (32, 64, 128) - default: 128
- `LIGHTING`: Lightning steps ("4", "8", "NONE") - default: "8"
- `USE_ORIGINAL_TEXT_ENCODER`: Use original vs compact text encoder - default: "true"

### Runtime Environment Variables

- `MODELS_DIR`: Model storage directory - default: "./models"
- `DEFAULT_STEPS`: Default inference steps - default: 8
- `DEFAULT_SCALE`: Default CFG scale - default: 4.0
- `RUNPOD_LOCAL_TEST`: Enable local FastAPI server for testing - default: not set

### Docker Build Examples

```cmd
# Standard build (Rank 128, Original Text Encoder)
docker build -t qwen-image-edit-nunchaku .

# Compact build (Rank 32, Compact Text Encoder, ~29-30GB)
docker build --build-arg RANK=32 --build-arg USE_ORIGINAL_TEXT_ENCODER=false -t qwen-image-edit-nunchaku:compact .

# Fast inference (4-step Lightning)
docker build --build-arg LIGHTING=4 -t qwen-image-edit-nunchaku:fast .

# High quality (Original settings + LoRA)
docker build --build-arg DOWNLOAD_LORA=true -t qwen-image-edit-nunchaku:quality .
```

## Model Structure

```
models/
├── diffusion_models/
│   └── transformer.safetensors    # Nunchaku quantized transformer
├── text_encoders/
│   └── qwen_2.5_vl_7b_fp8_scaled.safetensors  # Optional compact text encoder
└── Qwen-Image-Edit/               # Pipeline config and original text encoder
    ├── scheduler/
    ├── vae/
    ├── text_encoder/
    └── ...
```

## Acknowledgments

- **Qwen Team** (Alibaba Cloud) - Qwen-Image-Edit model
- **Nunchaku Team** - Quantization framework
- **Hugging Face** - Transformers and Diffusers
- **Runpod** - Serverless platform
- **Community Contributors** - Model variants and optimizations

## Author

Made with passion by Salvatore Rossitto

## Support & Documentation

- **Build Instructions**: See [BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md) for detailed build configurations
- **Quick Start**: See [QUICK_START.md](QUICK_START.md) for getting started quickly
- **Issues**: Check tests in `tests/` folder for usage examples
