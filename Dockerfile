# DotsOCR RunPod Dockerfile
# This builds a container with pre-downloaded models for fast startup

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Install flash-attn (this takes a while, so we do it separately)
RUN pip3 install --no-cache-dir flash-attn==2.8.0.post2 --no-build-isolation

# Clone dots.ocr repository
RUN git clone https://github.com/rednote-hilab/dots.ocr.git /app/dots.ocr

# Install dots.ocr
RUN pip3 install -e /app/dots.ocr

# Pre-download the model (this saves ~10 minutes on first run)
# This is the key optimization for RunPod!
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download( \
        repo_id='rednote-hilab/dots.ocr', \
        local_dir='/app/dots.ocr/weights/DotsOCR', \
        local_dir_use_symlinks=False \
    ); \
    print('✅ Model pre-downloaded successfully')"

# Create data directories
RUN mkdir -p /data/uploads /data/results

# Copy the application code
COPY app.py /app/app.py

# Expose the port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run the application
CMD ["python3", "app.py"]
