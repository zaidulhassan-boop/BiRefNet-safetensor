# Use lightweight PyTorch image with CUDA runtime
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Copy all project files into container
COPY . .

# Install wget (needed for model download) and clean up after
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Preload model from Hugging Face to avoid cold-start delay
RUN wget -O BiRefNet_model.safetensors \
    https://huggingface.co/zaidulhassan79/BiRefNet/resolve/main/BiRefNet_model.safetensors

# Expose port for RunPod
ENV PORT=8000

# Run the handler script when container starts
CMD ["python", "handler.py"]
