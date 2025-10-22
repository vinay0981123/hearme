# GPU-ready base image
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/.cache/huggingface
ENV TORCH_HOME=/workspace/.cache/torch
ENV PORT=8000
ENV HF_HUB_ENABLE_HF_TRANSFER=0

# Install system dependencies + full ffmpeg codecs
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    ffmpeg git ca-certificates tzdata \
  && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY hearme/ /app/hearme/
COPY requirements.txt /app/requirements.txt

# Upgrade pip and install Python dependencies
RUN python -m pip install --upgrade pip
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

# Create persistent cache directories
RUN mkdir -p /workspace/.cache/huggingface /workspace/.cache/torch \
    && chown -R 1000:1000 /workspace || true

# Expose port
EXPOSE $PORT

# Run FastAPI app directly
CMD ["uvicorn", "hearme.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
