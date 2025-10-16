# Quick Start Guide

Get up and running with Qwen Image Edit + Nunchaku in minutes!

## 🚀 TL;DR - Just Run It

```cmd
# Build the container
docker build -t qwen-nunchaku .

# Run locally for testing
docker run --gpus all -p 3000:3000 -e RUNPOD_LOCAL_TEST=1 qwen-nunchaku

# Test it (in another terminal)
curl -X POST http://localhost:3000/ -H "Content-Type: application/json" -d "{\"input\": {\"image\": \"https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/neon_sign.png\", \"prompt\": \"change the text to read 'Hello World'\"}}"
```

## 📋 Prerequisites

- ✅ Docker with GPU support
- ✅ NVIDIA GPU (8GB+ VRAM recommended)  
- ✅ 25GB+ free disk space
- ✅ Internet connection

## 🎯 Choose Your Build

### 🏃‍♂️ Quick & Easy (Recommended)
Best balance of speed and quality:
```cmd
docker build -t qwen-nunchaku:quick .
```

### 💾 Memory Optimized
For GPUs with limited VRAM:
```cmd
docker build --build-arg USE_ORIGINAL_TEXT_ENCODER=false --build-arg RANK=32 -t qwen-nunchaku:lite .
```

### 🎨 High Quality
Best quality, needs more VRAM:
```cmd
docker build --build-arg RANK=128 --build-arg DOWNLOAD_LORA=true -t qwen-nunchaku:quality .
```

## 🧪 Test It Out

### 1. Start the container
```cmd
docker run --gpus all -p 3000:3000 -e RUNPOD_LOCAL_TEST=1 qwen-nunchaku
```

### 2. Send a test request
```cmd
curl -X POST http://localhost:3000/ \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image": "https://example.com/image.jpg",
      "prompt": "make the sky purple"
    }
  }'
```

### 3. Or use the test script
```cmd
python test_local_http_endpoint.py
```

## 📊 Expected Results

- **Build time**: 15-30 minutes (depending on internet speed)
- **Container size**: 12-19GB (depending on configuration)
- **First inference**: 30-60 seconds (model loading)
- **Subsequent inference**: 3-8 seconds

## 🔧 Common Configurations

| Use Case | Build Command | Size | Speed | Quality |
|----------|---------------|------|-------|---------|
| Development | `docker build --build-arg USE_ORIGINAL_TEXT_ENCODER=false .` | ~12GB | Fast | Good |
| Production | `docker build .` | ~18GB | Fast | Best |
| Low Memory | `docker build --build-arg RANK=32 --build-arg USE_ORIGINAL_TEXT_ENCODER=false .` | ~8GB | Fast | Good |
| Max Quality | `docker build --build-arg RANK=128 --build-arg DOWNLOAD_LORA=true .` | ~19GB | Fastest | Best |

## 🐛 Something Not Working?

### Build Issues
```cmd
# Clean up Docker
docker system prune -a

# Check disk space
df -h

# Rebuild with verbose output
docker build --progress=plain -t qwen-nunchaku .
```

### Runtime Issues
```cmd
# Check GPU availability
nvidia-smi

# Check container logs
docker logs <container-id>

# Try with less memory usage
docker run --gpus all -p 3000:3000 \
  -e RUNPOD_LOCAL_TEST=1 \
  -e USE_ORIGINAL_TEXT_ENCODER=false \
  qwen-nunchaku
```

### Memory Errors
Try these in order:
1. Use compact text encoder: `USE_ORIGINAL_TEXT_ENCODER=false`
2. Lower rank: `RANK=32`
3. Restart container
4. Rebuild with memory-optimized settings

## 🎛️ Runtime Options

### API Parameters
```json
{
  "input": {
    "image": "url_or_base64",
    "prompt": "what to change",
    "negative_prompt": "what to avoid",
    "num_inference_steps": 8,
    "true_cfg_scale": 4.0,
    "width": 1024,
    "height": 1024
  }
}
```

### Environment Variables
```cmd
docker run --gpus all -p 3000:3000 \
  -e RUNPOD_LOCAL_TEST=1 \
  -e DEFAULT_STEPS=6 \
  -e DEFAULT_SCALE=3.5 \
  qwen-nunchaku
```

## 🚀 Deploy to RunPod

1. **Build and test locally first**
2. **Push to container registry**
3. **Create RunPod template**
4. **Deploy with GPU instances**

For detailed deployment instructions, see the main README.

## 📚 Next Steps

- Read [BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md) for advanced configuration
- Check [README.md](README.md) for complete documentation
- Explore different model configurations
- Deploy to production

## 💡 Tips

- 🔄 **Always test locally first** before deploying
- 📏 **Start with default settings** then optimize
- 💾 **Monitor memory usage** with `nvidia-smi`
- 🕐 **First run takes longer** due to model initialization
- 🎯 **Choose config based on your GPU** and requirements
- 📸 **Test with different images** to verify quality
- 📸 **For local Tests** remember to install torch before anything else (pip install torch==2.8 torchvision --index-url https://download.pytorch.org/whl/cu128)