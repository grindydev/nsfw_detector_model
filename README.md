# NSFW Detector

A deep learning project that classifies images into 5 categories: `drawings`, `hentai`, `neutral`, `porn`, `sexy`. Built as a learning exercise covering the full PyTorch pipeline from scratch to deployment.

## Results

All models evaluated on the same 5,600 image test set (20% of 28K dataset).

| Model | Test Accuracy | Val Accuracy | Size | Params |
|-------|:------------:|:------------:|-----:|-------:|
| SimpleCNN (3 layers) | **80.18%** | 78.45% | 645 KB | 162K |
| ResNet18 Transfer (fine-tune) | 79.43% | 87.71% | 43 MB | 11.7M |
| FlexibleCNN (Optuna) | 79.27% | 78.21% | 814 KB | 210K |
| TunedCNN (5 layers, Optuna) | 77.20% | 87.98% | 2.3 MB | 593K |
| ResidualTunedCNN (skip connections) | 75.66% | 88.19% | 6.1 MB | 1.6M |

**Key insights:**
- **SimpleCNN (80.18%)** generalizes best — right-sized model for 28K images wins
- TunedCNN and ResidualTunedCNN overfit (val 88% → test 75-77%) — too complex for this dataset
- ResNet18 Transfer also overfits (val 87% → test 79%) despite pretrained weights

## Dataset

Download the dataset from HuggingFace (zip ~2GB):

```bash
# Replace {token} with your HuggingFace access token
# Generate one at: https://huggingface.co/settings/tokens

mkdir -p data
wget --header="Authorization: Bearer {token}" \
     https://huggingface.co/datasets/deepghs/nsfw_detect/resolve/main/nsfw_dataset_v1.zip

unzip nsfw_dataset_v1.zip -d data/
rm nsfw_dataset_v1.zip
```

Expected structure after extraction:
```
data/
└── nsfw_dataset_v1/
    ├── drawings/
    ├── hentai/
    ├── neutral/
    ├── porn/
    └── sexy/
```

28,000 images across 5 classes, already balanced.

## Quick Start

### Web App (Recommended)

```bash
# 1. Export model to ONNX (one time)
#    Auto-detects model type (SimpleCNN / TunedCNN / ResidualTunedCNN / ResNet18)
cd src/
python export_onnx.py

# 2. Start the app
cd ../client/
pip install -r requirements.txt
python start.py
```

Open http://localhost:3000 — upload an **image** to classify, or a **video** to scan frame-by-frame for NSFW content.

**Prerequisite:** `ffmpeg` must be installed for video scanning.
```bash
brew install ffmpeg      # Mac
sudo apt install ffmpeg  # Linux
```

### Train a Model

```bash
cd src/

# Quick test (5 epochs)
python main.py

# Full training (GPU, 40 epochs)
# Edit CONFIG in main.py: "mode": "train"
python main.py

# Evaluate any model
python evaluate.py
```

### Hyperparameter Search

```bash
cd src/
python tuning.py          # Optuna searches best architecture
```

## Project Structure

```
├── src/
│   ├── main.py                   # SimpleCNN training pipeline
│   ├── cnn.py                    # SimpleCNN model (3 layers)
│   ├── cnn_tuned.py              # TunedCNN (5 layers, Optuna-optimized)
│   ├── residual_cnn_tuned.py     # ResidualTunedCNN (skip connections)
│   ├── tuning.py                 # Optuna hyperparameter search
│   ├── evaluate.py               # Test evaluation (works for all models)
│   ├── grad_cam.py               # Grad-CAM visualization
│   ├── export_onnx.py            # Export to ONNX
│   ├── prune.py                  # Pruning + Quantization pipeline
│   ├── transfer_cnn.py           # ResNet18 Strategy 1 (freeze all)
│   ├── transfer_cnn_finetune.py  # ResNet18 Strategy 2 (fine-tune)
│   ├── transfer_cnn_fulltrain.py # ResNet18 Strategy 3 (full retrain)
│   ├── train_tuned.py            # TunedCNN training
│   ├── train_residual.py         # ResidualTunedCNN training
│   ├── data_loader.py            # Dataset, transforms, DataLoaders
│   └── helper_utils.py           # Progress bars, plotting, utilities
│
├── client/
│   ├── server.py                 # FastAPI backend (image + video inference)
│   ├── start.py                  # One-command launcher
│   └── frontend/                 # React frontend (image + video upload)
│
├── documents/
│   └── phase5_residual_connections.md
│
├── models/                       # Saved checkpoints + ONNX
├── roadmap.md                    # Full learning roadmap (7 phases)
└── data/nsfw_dataset_v1/         # Dataset (not included in repo)
```

## Requirements

### Training (src/)
- Python 3.10+
- PyTorch 2.0+
- torchvision
- scikit-learn
- optuna
- mlflow
- joblib

### Deployment (client/)
- fastapi
- uvicorn
- onnxruntime
- Pillow
- ffmpeg (system install, for video scanning)
- Node.js 18+ (frontend)

## What I Learned

This project covers the full ML pipeline through 7 phases:

1. **Build & Train** — Custom CNN, data augmentation, mixed precision, early stopping
2. **Evaluate** — Confusion matrix, per-class F1, MLflow tracking
3. **Tuning** — Optuna hyperparameter search, flexible CNN architecture
4. **Transfer Learning** — ResNet18 with 3 strategies (freeze/fine-tune/full retrain)
5. **Skip Connections** — Residual blocks, vanishing gradients
6. **Interpretability** — Grad-CAM heatmaps, model debugging
7. **Deployment** — ONNX export, pruning, quantization, FastAPI + React web app

See [roadmap.md](roadmap.md) for detailed notes on each phase.

## License

Educational project — not intended for production use.
