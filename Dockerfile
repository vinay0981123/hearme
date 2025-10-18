# GPU-ready base image
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/.cache/huggingface
ENV TORCH_HOME=/workspace/.cache/torch
ENV PORT=8000

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git ca-certificates tzdata \
  && rm -rf /var/lib/apt/lists/*

# Use /app for code to avoid being overwritten by RunPod's /workspace mount
WORKDIR /app

# Copy project files
COPY hearme/ /app/hearme/
COPY requirements.txt /app/requirements.txt
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Install dependencies
RUN python -m pip install --upgrade pip
RUN python -m pip install -r /app/requirements.txt

# Create cache directories (on persistent volume)
RUN mkdir -p /workspace/.cache/huggingface /workspace/.cache/torch \
    && chown -R 1000:1000 /workspace || true

EXPOSE 8000

# Run the app
CMD ["/app/start.sh"]
