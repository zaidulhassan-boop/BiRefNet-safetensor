# Use lightweight PyTorch GPU image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Install curl (needed to fetch model)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy all project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Preload BiRefNet model (download from Hugging Face)
RUN curl -L -o BiRefNet_model.safetensors \
    https://huggingface.co/zaidulhassan79/load/resolve/main/model.safetensors

# Set environment variable for RunPod
ENV PORT=8000

# Run handler
CMD ["python", "handler.py"]
