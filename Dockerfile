# Use lightweight PyTorch GPU image.
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /workspace

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Preload BiRefNet model (use curl instead of wget)
RUN curl -L -o BiRefNet_model.safetensors \
    https://huggingface.co/zaidulhassan79/BiRefNet/resolve/main/BiRefNet_model.safetensors

# Set environment variable for RunPod
ENV PORT=8000

# Start handler
CMD ["python", "handler.py"]
