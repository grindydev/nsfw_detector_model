"""
FastAPI backend for NSFW Detector.

Loads ONNX model, accepts image/video uploads, returns predictions.
Usage:  uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

import base64
import io
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image

# ==================== CONFIG ====================
# Adjust these based on which model you exported to ONNX
INPUT_SIZE = 128                        # 128 for SimpleCNN, 224 for TunedCNN/Residual/ResNet
MEAN = [0.5973, 0.5313, 0.5066]         # NSFW dataset (for SimpleCNN)
STD = [0.2896, 0.2808, 0.2854]          # Use [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225] for TunedCNN/Residual
MODEL_PATH = Path(__file__).parent.parent / "models" / "nsfw_detector.onnx"
CLASS_NAMES = ["drawings", "hentai", "neutral", "porn", "sexy"]
NSFW_CLASSES = {"hentai", "porn", "sexy"}
VIDEO_FRAME_INTERVAL = 1       # extract 1 frame per second
NSFW_THRESHOLD = 0.5           # flag if confidence > 50% for NSFW class
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


@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):
    """Accept a video file, stream frame-by-frame NSFW classification via SSE."""
    if session is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run export_onnx.py first.")

    # Validate file type
    content_type = file.content_type or ""
    if not (content_type.startswith("video/") or content_type in ("application/octet-stream",)):
        raise HTTPException(status_code=400, detail="File must be a video")

    # Check ffmpeg is available
    if not shutil.which("ffmpeg"):
        raise HTTPException(status_code=500, detail="ffmpeg is not installed. Install it to enable video scanning.")

    # Save uploaded video to temp file
    tmp_dir = tempfile.mkdtemp()
    ext = Path(file.filename or "video.mp4").suffix or ".mp4"
    video_path = os.path.join(tmp_dir, f"input{ext}")
    frames_dir = os.path.join(tmp_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    with open(video_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    def sse(data: dict) -> str:
        return f"data: {json.dumps(data)}\n\n"

    def encode_frame_thumbnail(frame_path: str, max_size: int = 160) -> str:
        """Resize frame and return as base64 JPEG."""
        img = Image.open(frame_path).convert("RGB")
        img.thumbnail((max_size, max_size))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=50)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def event_stream():
        input_name = session.get_inputs()[0].name
        flagged = []

        try:
            # 1. Tell client we're extracting frames
            yield sse({"type": "extracting", "message": "Extracting frames from video..."})

            # 2. Get video duration
            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "csv=p=0", video_path],
                capture_output=True, text=True, timeout=10
            )
            try:
                duration = float(probe.stdout.strip())
            except ValueError:
                duration = 0.0

            # 3. Extract frames with ffmpeg
            subprocess.run(
                ["ffmpeg", "-i", video_path,
                 "-vf", f"fps=1/{VIDEO_FRAME_INTERVAL}",
                 "-q:v", "2",
                 os.path.join(frames_dir, "frame_%04d.jpg")],
                capture_output=True, timeout=120
            )

            frame_files = sorted(Path(frames_dir).glob("frame_*.jpg"))
            if not frame_files:
                yield sse({"type": "error", "message": "Could not extract frames from video"})
                return

            total = len(frame_files)

            # 4. Send frame count info
            yield sse({"type": "info", "total_frames": total, "duration": round(duration, 1)})

            # 5. Classify each frame and stream results
            for i, frame_path in enumerate(frame_files):
                try:
                    image = Image.open(frame_path).convert("RGB")
                    input_array = preprocess_image(image)
                    logits = session.run(None, {input_name: input_array})[0][0]
                    probabilities = softmax(logits)
                    top_idx = int(np.argmax(probabilities))

                    probs_dict = {
                        CLASS_NAMES[j]: round(float(probabilities[j]), 4)
                        for j in range(len(CLASS_NAMES))
                    }

                    frame_result = {
                        "type": "frame",
                        "frame": i + 1,
                        "total": total,
                        "timestamp": round(i * VIDEO_FRAME_INTERVAL, 1),
                        "prediction": CLASS_NAMES[top_idx],
                        "confidence": round(float(probabilities[top_idx]), 4),
                        "probabilities": probs_dict,
                    }

                    # Check if NSFW
                    is_nsfw = any(
                        probs_dict[cls] > NSFW_THRESHOLD
                        for cls in NSFW_CLASSES
                    )
                    if is_nsfw:
                        frame_result["nsfw"] = True
                        frame_result["nsfw_classes"] = {
                            cls: probs_dict[cls]
                            for cls in NSFW_CLASSES
                            if probs_dict[cls] > NSFW_THRESHOLD
                        }
                        frame_result["image"] = encode_frame_thumbnail(str(frame_path))
                        flagged.append(frame_result)

                    yield sse(frame_result)

                except Exception as e:
                    yield sse({"type": "error", "frame": i + 1, "message": str(e)})

            # 6. Send final summary
            yield sse({
                "type": "done",
                "total_frames": total,
                "flagged_frames": len(flagged),
                "is_nsfw": len(flagged) > 0,
            })

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
