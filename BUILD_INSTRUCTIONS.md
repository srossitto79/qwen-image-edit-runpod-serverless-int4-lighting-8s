# Build Instructions - Qwen Image Edit with Nunchaku

This document provides detailed instructions for building and configuring the enhanced Qwen Image Edit serverless container with Nunchaku quantization.

## Prerequisites

- Docker with NVIDIA GPU support
- At least 90GB free disk space for build (base image + models + build artifacts)
- Internet connectivity for model downloads
- NVIDIA GPU with >=8GB VRAM for runtime

## Build Configuration

### Environment Variables

The build process supports several environment variables for customization:

| Variable | Default | Description |
|----------|---------|-------------|
| `RANK` | `128` | SVD quantization rank (32, 64, 128) |
| `LIGHTING` | `8` | Lightning steps ("4", "8", "NONE") |
| `USE_ORIGINAL_TEXT_ENCODER` | `true` | Use original vs compact text encoder |

### Build Arguments

These can be passed to Docker build using `--build-arg`:

```cmd
docker build --build-arg RANK=32 --build-arg LIGHTING=8 -t my-image .
```

## Build Examples

### Basic Build (Recommended)
```cmd
docker build -t qwen-nunchaku:latest .
```
- Rank: 128
- Lightning: 8 steps
- Text Encoder: Original
- Size: ~49-50GB

### Optimized Build (Memory Efficient)
```cmd
docker build --build-arg USE_ORIGINAL_TEXT_ENCODER=false --build-arg RANK=32 -t qwen-nunchaku:compact .
```
- Rank: 32 (smaller model)
- Lightning: 8 steps
- Text Encoder: Compact FP8
- Size: ~29-30GB


### Development Build (Fast iterations)
```cmd
docker build --build-arg COMPRESS_FILES=false --build-arg USE_ORIGINAL_TEXT_ENCODER=false -t qwen-nunchaku:dev .
```
- No compression (faster build)
- Compact text encoder
- Good for development/testing

## Model Variants Matrix

| Rank | Lightning | Model Size | Quality | Speed | VRAM Usage |
|------|-----------|------------|---------|-------|------------|
| 32   | NONE      | Small      | Good    | Slow  | Low        |
| 32   | 8         | Small      | Good    | Fast  | Low        |
| 128  | NONE      | Large      | Best    | Slow  | High       |
| 128  | 8         | Large      | Best    | Fast  | High       |

## Text Encoder Options

### Original Text Encoder
- **Pros**: Best quality, full compatibility
- **Cons**: Larger size (~8GB), higher VRAM usage
- **Use when**: Quality is priority, sufficient VRAM available

### Compact FP8 Text Encoder
- **Pros**: Smaller size (~2GB), lower VRAM usage, faster loading
- **Cons**: Slightly reduced quality
- **Use when**: Memory is limited, speed is priority

## Build Process Details

### Stage 1: Base Setup
- Install system dependencies
- Set up Python environment
- Install requirements including Nunchaku

### Stage 2: Model Download
- Create organized model directory structure
- Download pipeline configuration files
- Download quantized transformer based on RANK and LIGHTING
- Optionally download compact text encoder
- Optionally download LoRA weights

### Stage 3: Optimization
- Clean up unnecessary files
- Set offline mode flags

## Runtime Configuration

### Environment Variables at Runtime

| Variable | Default | Description |
|----------|---------|-------------|
| `MODELS_DIR` | `./models` | Model storage directory |
| `DEFAULT_STEPS` | `8` | Default inference steps |
| `DEFAULT_SCALE` | `4.0` | Default CFG scale |
| `USE_ORIGINAL_TEXT_ENCODER` | `true` | Text encoder preference |
| `RUNPOD_LOCAL_TEST` | - | Enable local FastAPI server |

### Memory Management

The container automatically adjusts memory usage based on available GPU memory:

- **>18GB VRAM**: Full model CPU offloading
- **<18GB VRAM**: Block-wise GPU loading with sequential CPU offload

## Troubleshooting

### Build Errors

**Out of disk space**
- Ensure at least 90GB free space (base image 7-8GB + models + build artifacts + Docker overhead)
- Use `docker system prune` to clean up

**Network timeouts**
- Check internet connectivity
- Model downloads can be large (25-35GB)
- Consider using a build machine with good bandwidth

**CUDA compatibility**
- Ensure NVIDIA drivers are up to date
- Check Docker NVIDIA runtime installation (CUDA >=2.8)

### Runtime Issues

**Out of memory**
- Try compact text encoder: `USE_ORIGINAL_TEXT_ENCODER=false`
- Reduce rank: `RANK=32`
- Ensure `--gpus all` flag is used

**Model loading errors**
- Check model files exist in container
- Verify offline mode environment variables
- Check container logs for specific errors

### Performance Optimization

**Faster inference**
- Use Lightning variants: `LIGHTING=8` or `LIGHTING=4`
- Use compact text encoder for memory efficiency

**Better quality**
- Use higher rank: `RANK=128`
- Use original text encoder: `USE_ORIGINAL_TEXT_ENCODER=true`
- Increase inference steps at runtime

## Advanced Configuration

### Custom Model Paths

You can mount custom models at runtime:

```cmd
docker run --gpus all -v /path/to/models:/models -e MODELS_DIR=/models qwen-nunchaku
```

### Development Mode

For development with hot reloading:

```cmd
docker run --gpus all -v $(pwd):/app -e RUNPOD_LOCAL_TEST=1 qwen-nunchaku
```

### Production Deployment

For production on RunPod:

1. Build with optimized settings
2. Test locally first
3. Push to registry
4. Deploy with appropriate GPU instances

## File Size Reference

| Component | Size |
|-----------|------|
| Base Image (runpod/pytorch) | ~7-8GB |
| Transformer (INT4 R128, any LIGHTING variant) | ~12.7GB |
| Transformer (INT4 R32, any LIGHTING variant) | ~11.5GB |
| Transformer (FP4 R128, any LIGHTING variant) | ~13.1GB |
| Transformer (FP4 R32, any LIGHTING variant) | ~11.9GB |
| Text Encoder (Original, full precision) | ~16GB |
| Text Encoder (Compact, FP8) | ~9GB |
| Pipeline Config (with text encoder) | ~16GB |
| Pipeline Config (without text encoder) | ~340MB |
| **Total (INT4 R128 + Original TE)** | **~49-50GB** |
| **Total (INT4 R128 + Compact TE)** | **~29-30GB** |
| **Total (INT4 R32 + Compact TE)** | **~28-29GB** |
| **Total (INT4 R32 only)** | **~19-20GB** |

## Best Practices

1. **Always test locally** before deploying
2. **Use compact text encoder** for memory-constrained environments
3. **Enable compression** for production builds
4. **Choose appropriate rank** based on quality/speed tradeoffs
5. **Monitor VRAM usage** and adjust configuration accordingly
6. **Keep build args consistent** across environments
7. **Document your specific configuration** for reproducibility

## Support

For issues and questions:
- Check the troubleshooting section above
- Review container logs for specific errors
- Test with different configuration options
- Ensure hardware requirements are met