# Use lightweight PyTorch GPU image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /workspace

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Preload model from Hugging Face (your direct resolve link)
RUN wget -O BiRefNet_model.safetensors \
    https://huggingface.co/zaidulhassan79/BiRefNet/resolve/main/BiRefNet_model.safetensors

# Set environment variable for RunPod
ENV PORT=8000

# Command for RunPod serverless
CMD ["python", "handler.py"]
