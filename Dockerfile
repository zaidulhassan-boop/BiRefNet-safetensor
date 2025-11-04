# Use official PyTorch runtime image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /workspace

# Install curl and dependencies
RUN apt-get update && apt-get install -y curl

# Copy all project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Preload BiRefNet model (use curl)
RUN curl -L -o BiRefNet_model.safetensors \
    https://huggingface.co/zaidulhassan79/BiRefNet/resolve/main/BiRefNet_model.safetensors

# Set environment variable for handler
ENV PYTHONUNBUFFERED=1

# Run handler
CMD ["python", "handler.py"]
