# Docker Quick Start Guide

Quick reference for building and pushing your DotsOCR Docker image.

## Prerequisites

1. **Docker Desktop** installed and running
2. **Docker Hub account** created at https://hub.docker.com

## Quick Commands

### 1. Build the Image

```bash
# Navigate to project directory
cd /path/to/dots_ocr

# Build (replace YOUR_USERNAME with your Docker Hub username)
docker build -t YOUR_USERNAME/dots-ocr:latest .

# Example:
# docker build -t johnsmith/dots-ocr:latest .
```

**⏱️ Build time:** 30-60 minutes (downloads 10GB+ model)

### 2. Login to Docker Hub

```bash
docker login
```

Enter your Docker Hub username and password when prompted.

### 3. Push to Docker Hub

```bash
# Push (replace YOUR_USERNAME)
docker push YOUR_USERNAME/dots-ocr:latest

# Example:
# docker push johnsmith/dots-ocr:latest
```

**⏱️ Push time:** 20-40 minutes (uploads ~15GB image)

### 4. Verify

Visit: https://hub.docker.com/r/YOUR_USERNAME/dots-ocr

You should see your image with the "latest" tag.

---

## Testing Locally (Optional)

### Test the Image Before Pushing

```bash
# Run locally
docker run --rm --gpus all -p 7860:7860 YOUR_USERNAME/dots-ocr:latest

# Test in browser
# Visit: http://localhost:7860/health
```

**Note:** Requires NVIDIA GPU and nvidia-docker2 installed.

---

## Troubleshooting

### Docker Desktop Not Running

**Mac/Windows:** Open Docker Desktop app and wait for it to fully start

**Linux:**
```bash
sudo systemctl start docker
```

### Build Failed - Out of Space

```bash
# Clean up old images
docker system prune -a

# Check available space
docker system df
```

### Push Failed - Authentication Required

```bash
# Login again
docker logout
docker login
```

### Image Too Large

The image is ~15GB because it includes:
- CUDA runtime (~5GB)
- Python packages (~3GB)
- AI model weights (~7GB)

This is normal and expected!

---

## Next Steps

Once your image is pushed to Docker Hub:

1. Go to `RUNPOD_DEPLOYMENT_GUIDE.md`
2. Follow from **Step 3: Sign Up for RunPod**
3. Use your Docker Hub image URL: `YOUR_USERNAME/dots-ocr:latest`

---

## Image Details

**Base Image:** `nvidia/cuda:12.1.0-runtime-ubuntu22.04`

**Key Components:**
- Python 3.10
- PyTorch 2.4.0 with CUDA support
- Flash Attention 2.8.0
- DotsOCR model (pre-downloaded)
- FastAPI + Uvicorn

**Ports:** 7860 (HTTP)

**Volumes:** `/data` (for uploads and results)

**GPU Required:** Yes (CUDA 12.1+)

---

## Advanced: Multi-Platform Build

To build for different architectures (not usually needed):

```bash
# Enable buildx
docker buildx create --use

# Build for multiple platforms
docker buildx build --platform linux/amd64,linux/arm64 \
  -t YOUR_USERNAME/dots-ocr:latest \
  --push .
```

**Note:** This takes 2-3x longer and is usually not necessary for RunPod.

---

## Updating Your Image

When you update `app.py` or other files:

```bash
# 1. Rebuild
docker build -t YOUR_USERNAME/dots-ocr:latest .

# 2. Push
docker push YOUR_USERNAME/dots-ocr:latest

# 3. On RunPod:
#    - Stop your pod
#    - Start it again (it will pull the new image)
```

**Pro Tip:** Use version tags for better control:

```bash
# Build with version
docker build -t YOUR_USERNAME/dots-ocr:v1.0.1 .
docker push YOUR_USERNAME/dots-ocr:v1.0.1

# Also tag as latest
docker tag YOUR_USERNAME/dots-ocr:v1.0.1 YOUR_USERNAME/dots-ocr:latest
docker push YOUR_USERNAME/dots-ocr:latest
```

---

## Cost Optimization

### Reduce Image Size (Advanced)

Create a `.dockerignore` file (already done!) to exclude unnecessary files:
- ✅ Already configured in your project
- Saves ~500MB in build time

### Use Docker Layer Caching

When rebuilding, Docker reuses layers that haven't changed:
- Requirements are installed first (rarely change)
- Model is downloaded early (never changes)
- App code is copied last (changes frequently)

This means rebuilds after code changes take only **2-3 minutes**!

---

## Quick Reference: Docker Commands

| Command | Purpose |
|---------|---------|
| `docker images` | List all local images |
| `docker ps` | List running containers |
| `docker ps -a` | List all containers |
| `docker logs <container>` | View container logs |
| `docker stop <container>` | Stop a container |
| `docker rm <container>` | Remove a container |
| `docker rmi <image>` | Remove an image |
| `docker system prune -a` | Clean up everything |

---

**Ready to deploy?** Continue with `RUNPOD_DEPLOYMENT_GUIDE.md`!
