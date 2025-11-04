import io
import base64
import torch
from PIL import Image
import requests
from models.birefnet import BiRefNet
import torchvision.transforms as transforms
import os
import runpod  # ✅ Make sure this is also in requirements.txt

# === Download model if not exists ===
model_path = "BiRefNet_model.safetensors"
if not os.path.exists(model_path):
    url = "https://huggingface.co/zaidulhassan79/BiRefNet/resolve/main/BiRefNet_model.safetensors"
    print("Downloading BiRefNet model from Hugging Face...")
    r = requests.get(url)
    with open(model_path, "wb") as f:
        f.write(r.content)
    print("Model downloaded successfully.")

# === Load model ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model = BiRefNet()
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# === Helper: preprocess image ===
def load_image_from_input(input_data):
    if input_data.startswith("http"):
        response = requests.get(input_data)
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
    else:
        image_bytes = base64.b64decode(input_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return image

# === Main handler ===
def handler(event):
    try:
        input_data = event.get("input", {}).get("image")
        if not input_data:
            return {"error": "Missing image input"}

        image = load_image_from_input(input_data)
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(input_tensor)[0][0]
            mask = (pred > 0.5).float().cpu()

        # Convert to transparent PNG
        mask_img = transforms.ToPILImage()(mask)
        image = image.resize(mask_img.size)
        image.putalpha(mask_img)

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        base64_output = base64.b64encode(buf.getvalue()).decode("utf-8")

        # ✅ Return using key RunPod expects
        return {"output": base64_output}

    except Exception as e:
        return {"error": str(e)}

# === ✅ Required by RunPod Serverless ===
runpod.serverless.start({"handler": handler})
