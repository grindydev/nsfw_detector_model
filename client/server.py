"""
FastAPI backend for NSFW Detector.

Loads ONNX model, accepts image uploads, returns predictions.
Usage:  uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# ==================== CONFIG ====================
# Adjust these based on which model you exported to ONNX
INPUT_SIZE = 128                        # 128 for SimpleCNN, 224 for TunedCNN/Residual/ResNet
MEAN = [0.5973, 0.5313, 0.5066]         # NSFW dataset (for SimpleCNN)
STD = [0.2896, 0.2808, 0.2854]          # Use [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225] for TunedCNN/Residual
MODEL_PATH = Path(__file__).parent.parent / "models" / "nsfw_detector.onnx"
CLASS_NAMES = ["drawings", "hentai", "neutral", "porn", "sexy"]
# ================================================

app = FastAPI(title="NSFW Detector API")

# Allow React frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ONNX model on startup
print(f"Loading ONNX model from: {MODEL_PATH}")
if not MODEL_PATH.exists():
    print(f"⚠️  Model not found! Run export_onnx.py first to create {MODEL_PATH}")
    session = None
else:
    session = ort.InferenceSession(str(MODEL_PATH))
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    print(f"✅ Model loaded")
    print(f"   Input:  {input_info.name} {input_info.shape}")
    print(f"   Output: {output_info.name} {output_info.shape}")


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize, normalize, and convert image to ONNX input format."""
    # Resize to model's expected input size
    image = image.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)

    # Convert to numpy array [0, 1]
    img_array = np.array(image, dtype=np.float32) / 255.0

    # Normalize with mean/std
    mean = np.array(MEAN, dtype=np.float32)
    std = np.array(STD, dtype=np.float32)
    img_array = (img_array - mean) / std

    # HWC → CHW (height, width, channels → channels, height, width)
    img_array = img_array.transpose(2, 0, 1)

    # Add batch dimension: (C, H, W) → (1, C, H, W)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def softmax(logits: np.ndarray) -> np.ndarray:
    """Convert raw logits to probabilities."""
    exp_logits = np.exp(logits - np.max(logits))  # numerical stability
    return exp_logits / exp_logits.sum()


@app.get("/")
def root():
    """API info and model status."""
    return {
        "name": "NSFW Detector API",
        "model_loaded": session is not None,
        "model_path": str(MODEL_PATH),
        "input_size": INPUT_SIZE,
        "classes": CLASS_NAMES,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Accept an image file, return NSFW classification."""
    if session is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run export_onnx.py first.")

    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Read image
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    # Preprocess
    input_array = preprocess_image(image)

    # Run ONNX inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_array})

    # Get probabilities
    logits = outputs[0][0]
    probabilities = softmax(logits)

    # Build response
    top_idx = int(np.argmax(probabilities))
    probs_dict = {
        CLASS_NAMES[i]: round(float(probabilities[i]), 4)
        for i in range(len(CLASS_NAMES))
    }

    return {
        "prediction": CLASS_NAMES[top_idx],
        "confidence": round(float(probabilities[top_idx]), 4),
        "probabilities": probs_dict,
        "image_size": f"{image.width}x{image.height}",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
