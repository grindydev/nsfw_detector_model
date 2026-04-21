# NSFW Detector — Learning Roadmap

A hands-on exercise that practices the full PyTorch image classification pipeline, from building a custom CNN to deploying an optimized model.

---

## Dataset

- **Source:** `data/nsfw_dataset_v1/` (28,000 images, 5 classes)
- **Classes:** `drawings`, `hentai`, `neutral`, `porn`, `sexy`
- **Image sizes:** Width 112–3300 (mean 670), Height 147–5400 (mean 756)
- **Aspect ratios:** 61% portrait/tall, 20% landscape, 10% square, rest wide
- **Input size:** 128×128 (resize to square — preserves all content, slight distortion acceptable at this resolution)
- **Split:** 65% train / 15% val / 20% test

---

## Project Structure

```
nsfw_detector/
├── ROADMAP.md              ← You are here
├── main.py                 ← Config-driven training pipeline
├── cnn.py                  ← SimpleCNN model definition
├── data_loader.py          ← Custom Dataset, transforms, DataLoaders
├── helper_utils.py         ← Plotting, utilities
├── optuna.py               ← (Phase 3) Hyperparameter tuning
├── evaluate.py             ← (Phase 2) Test-set evaluation, metrics
├── grad_cam.py             ← (Phase 6) Grad-CAM visualization
├── predict.py              ← (Phase 7) Single image inference
├── transfer_cnn.py         ← (Phase 5) Transfer learning model
├── export_onnx.py          ← (Phase 7) ONNX export
└── data/
    └── nsfw_dataset_v1/    ← Dataset (on Ubuntu training machine)
```

---

## Learning Path

```
 ✅ Phase 1 — Build & Train SimpleCNN
    │
    ▼
 🔲 Phase 2 — Evaluate Baseline (test accuracy, confusion matrix, F1)
    │           → Know your starting point
    │           → Discover which classes are problematic
    │
    ▼
 🔲 Phase 3 — Optuna Tuning (push SimpleCNN to its limit)
    │           → Use Phase 2 insights to guide search space
    │           → Compare with Phase 2 baseline
    │
    ▼
 🔲 Phase 4 — Transfer Learning (biggest accuracy jump)
    │           → ResNet18/MobileNetV3 pretrained on ImageNet
    │           → Compare with Phase 2 + Phase 3 results
    │
    ▼
 🔲 Phase 5 — ResNet Skip Connections
    │           → Add residual connections to your CNNBlock
    │           → Understand why deeper networks work
    │
    ▼
 🔲 Phase 6 — Model Interpretability (Grad-CAM)
    │           → See where the model looks to make predictions
    │           → Debug misclassifications visually
    │
    ▼
 🔲 Phase 7 — Export & Deployment (ONNX → Prune → Quantize)
                → Export to ONNX, shrink model, benchmark speed
```

---

## Phase 1 — Build & Train SimpleCNN ✅

**Status:** Done

**What you practiced:**

| Concept | Where in your code | Course reference |
|---------|-------------------|-----------------|
| Custom Dataset with `__len__`, `__getitem__` | `data_loader.py` — `NSFWDataset` | L1-M3 `data_management/main.py` |
| Train/val/test split with different transforms | `data_loader.py` — `SubsetWithTransform` | L1-M3 `data_management/main.py` |
| Data augmentation (flip, rotate, color jitter) | `data_loader.py` — `get_transformations()` | L1-M3 `data_management/main.py` |
| Computing dataset mean/std | `data_loader.py` — `get_mean_std()` | L1-M2 `transform_dataset.py` |
| CNN blocks (Conv2d → BatchNorm → ReLU → MaxPool) | `cnn.py` — `CNNBlock` | L1-M4 `cnn/cnn_block.py` |
| AdaptiveAvgPool2d (input-size agnostic) | `cnn.py` — `SimpleCNN.classifier` | L1-M4 `cnn/main.py` |
| Config-driven training pipeline | `main.py` — `CONFIG` dict | — |
| Mixed precision (AMP) | `main.py` — `autocast`, `GradScaler` | — |
| Early stopping + Cosine LR scheduler | `main.py` — `training_loop()` | L2-M1 `scheduler/main.py` |
| Best model checkpointing | `main.py` — save on improvement | L1-M4 `cnn/main.py` |
| Device-aware (CUDA / MPS / CPU) | `main.py` — auto-detect | — |

**Key design decisions:**
- Input 128×128 — fast iteration on GTX 1650 (4GB VRAM), adequate for NSFW patterns
- `Resize((128, 128))` — forces square, keeps all content (distortion minimal at this resolution)
- `label_smoothing=0.1` — prevents overconfident predictions
- `AdamW` with `weight_decay=0.05` — better regularization than plain Adam

---

## Phase 2 — Evaluate Baseline ✅

**Goal:** Measure your starting point. You can't improve what you don't measure.

**Built in:** `evaluate.py`

### What you practiced

| Concept | Where in your code | What you learned |
|---------|-------------------|-----------------|
| Test set is separate from validation | `evaluate.py` — loads model, runs on test_loader only | Test set is NEVER touched during training → gives honest, unbiased performance number |
| Overall accuracy as a single number | `evaluate.py` — `test_accuracy = 100.0 * correct / total` | 64.36% is the baseline, every future change compares to this |
| Confusion matrix (sklearn) | `evaluate.py` + `helper_utils.py` — `confusion_matrix()`, `plot_confusion_matrix()` | Shows WHICH classes get confused: neutral ↔ sexy ↔ porn is the main problem area |
| Per-class precision/recall/F1 | `evaluate.py` — `classification_report()` | Accuracy hides problems — porn (F1=0.73) vs neutral (F1=0.59) shows huge gap between classes |
| MLflow experiment tracking | `main.py` — `mlflow.log_params()`, `mlflow.log_metric()`, `mlflow.log_artifact()` | Every training run is automatically logged with params + metrics, no more manual spreadsheets |
| Config-driven hyperparameters | `main.py` — `lr`, `weight_decay`, `label_smoothing` in CONFIG dict | No more magic numbers hardcoded — single source of truth at the top of the file |
| `torch.compile()` portability fix | `main.py` — strip `_orig_mod.` prefix before saving | Training on CUDA with compile wraps model — must clean state_dict keys for Mac/Linux compatibility |

### Baseline results (5 epochs, test mode)

```
┌──────────────────────────────────────────┐
│ BASELINE (SimpleCNN)                     │
│ Test Accuracy:  64.36%                   │
│ Best class:     porn    (F1=0.73)        │
│ Worst class:    neutral (F1=0.59)        │
│ Main confusion: neutral ↔ sexy ↔ porn    │
│ Hentai recall:  0.57 (misses many)       │
│ Model status:   underfitting             │
└──────────────────────────────────────────┘
```

**Full classification report:**

```
              precision    recall  f1-score   support

    drawings       0.64      0.61      0.62      1099
      hentai       0.78      0.57      0.66      1099
     neutral       0.57      0.61      0.59      1129
        porn       0.71      0.75      0.73      1121
        sexy       0.58      0.68      0.62      1152

    accuracy                           0.64      5600
   macro avg       0.66      0.64      0.64      5600
weighted avg       0.65      0.64      0.64      5600
```

### Key takeaways for Phase 3
- Model is **underfitting** — 3 conv blocks may be too shallow
- `neutral` ↔ `sexy` ↔ `porn` confusion → deeper model may help distinguish
- `hentai` recall is low (0.57) → model misses many hentai images
- Optuna should search: more conv blocks, wider channels, lower dropout

---

## Phase 3 — Optuna Hyperparameter Tuning 🔲

**Goal:** Push SimpleCNN to its limit by finding the best hyperparameters.

**Build this file:** `optuna.py`

### What to tune

| Parameter | Search space | Impact |
|-----------|-------------|--------|
| Learning rate | `1e-4` to `1e-2` (log scale) | 🔴 Highest — too high diverges, too low crawls |
| Weight decay | `1e-5` to `1e-1` (log scale) | 🟡 Regularization strength |
| Dropout | `0.2` to `0.7` | 🟡 Classifier regularization |
| Batch size | `32, 64, 128` | 🟡 Affects gradient quality |
| Num conv blocks | `2, 3, 4, 5` | 🟡 Model depth |
| Channels | `16→32→64` vs `32→64→128` vs `64→128→256` | 🟡 Model width |
| Kernel size | `3, 5` | 🟢 Receptive field |

### How Optuna works

```python
def objective(trial):
    # 1. Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.2, 0.7)
    n_blocks = trial.suggest_int("n_blocks", 2, 5)
    
    # 2. Build model with these params
    model = FlexibleCNN(num_blocks=n_blocks, dropout=dropout, ...)
    
    # 3. Train for a few epochs
    # 4. Return validation accuracy
    return val_accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
```

### Phase 2 insights guide Phase 3

After seeing the confusion matrix, you know:
- If `sexy` ↔ `porn` confusion is the main problem → try deeper model (more blocks)
- If model overfits (train loss ↓ val loss ↑) → increase dropout/weight decay
- If model underfits (both losses high) → more channels, lower dropout

**Course reference:** `L2-M1 optuna/main.py` — **exact match**, shows flexible CNN + Optuna search

### After Optuna — record results

```
┌──────────────────────────────┐
│ OPTUNA (Tuned SimpleCNN)     │
│ Test Accuracy:  XX.XX%       │
│ Best params:    lr=?, ...    │
│ Improvement:    +X.X%        │
└──────────────────────────────┘
```

---

## Phase 4 — Transfer Learning 🔲

**Goal:** Use a pretrained model for the biggest accuracy jump.

**Build this file:** `transfer_cnn.py`

### Why transfer learning helps

Your SimpleCNN learns edges → textures → patterns from 28K NSFW images. A pretrained ResNet18 learned edges → textures → objects → scenes from **1.2 million ImageNet images**. You reuse that knowledge.

### Three strategies (try all, compare)

```python
import torchvision.models as models

# Load pretrained ResNet18
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Strategy 1: Feature Extraction (fastest)
# Freeze everything, replace only the final layer
for param in resnet.parameters():
    param.requires_grad = False
resnet.fc = nn.Linear(resnet.fc.in_features, 5)  # 5 NSFW classes

# Strategy 2: Fine-tuning (best balance)
# Freeze early layers, train later layers + new head
for name, param in resnet.named_parameters():
    if "layer4" not in name and "fc" not in name:
        param.requires_grad = False

# Strategy 3: Full Retraining (slowest, needs most data)
# Replace head, train everything with small LR
resnet.fc = nn.Linear(resnet.fc.in_features, 5)
# Use lr=1e-4 (not 1e-3, or you destroy pretrained weights)
```

### Important: preprocessing must match

Pretrained models expect specific input preprocessing. ResNet18 expects 224×224 with ImageNet normalization:

```python
# NOT your custom mean/std — use ImageNet values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
```

You'll need to change input size from 128→224 for transfer learning. Or try MobileNetV3 which also supports 128×128.

**Course reference:**
- `L2-M2 transfer_learning/main.py` — **exact match**, shows all 3 strategies
- `L2-M2 pre_processing/main.py` — preprocessing for pretrained models

### After transfer learning — record results

```
┌──────────────────────────────────────┐
│ TRANSFER LEARNING (ResNet18)         │
│ Strategy:        fine-tuning         │
│ Test Accuracy:   XX.XX%              │
│ Improvement vs   +X.X% vs Optuna     │
│ Improvement vs   +X.X% vs baseline   │
└──────────────────────────────────────┘
```

---

## Phase 5 — ResNet Skip Connections 🔲

**Goal:** Understand WHY deeper models work better by adding residual connections.

**Modify:** `cnn.py`

### The problem without skip connections

In deep networks, gradients vanish as they flow backward through many layers. Layer 1 barely receives a signal from the loss. The network "forgets" early features.

### The solution: skip connections

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # If channels change, we need a projection for the skip
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)        # save input (with projection if needed)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual                     # ← skip connection: add input back!
        out = F.relu(out)
        return out
```

**Course reference:** `L3-M1 resnet/main.py` — residual connections explained

---

## Phase 6 — Model Interpretability 🔲

**Goal:** See what the model has learned and where it looks.

**Build this file:** `grad_cam.py`

### What you'll visualize

| Visualization | What it shows | Answers the question |
|--------------|---------------|---------------------|
| **Conv filters** | Patterns each filter detects (edges, textures) | What has layer 1 learned? |
| **Feature maps** | How an image transforms at each layer | How does data flow through the model? |
| **Grad-CAM** | Heatmap of important regions for a prediction | Why did the model predict "porn" for this image? |

### Grad-CAM for debugging

When your model misclassifies `sexy` as `porn`:
- If Grad-CAM focuses on face/hands → model learned person features (reasonable)
- If Grad-CAM focuses on random background → model learned spurious correlation (fix needed)
- If Grad-CAM focuses on body → model is using the right features but boundary is unclear

**Course reference:**
- `L3-M2 interpreting/main.py` — filter + feature map visualization
- `L3-M2 saliency_and_class_activation_map/main.py` — Grad-CAM implementation

---

## Phase 7 — Export & Deployment 🔲

**Goal:** Ship your model — export, shrink, and benchmark.

### Step 7a — ONNX export

**Build this file:** `export_onnx.py`

```python
# Load your best model
model = SimpleCNN(num_classes=5)
model.load_state_dict(torch.load("best_simple_cnn_train.pth")["model_state_dict"])
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 128, 128)

# Export
torch.onnx.export(
    model, dummy_input,
    "nsfw_detector.onnx",
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}
)
```

**Course reference:** `L3-M4 ONNX/main.py`

### Step 7b — Pruning (remove unnecessary weights)

Remove weights that contribute little to predictions:

```python
import torch.nn.utils.prune as prune

# Prune 30% of smallest weights in each conv layer
for module in model.modules():
    if isinstance(module, nn.Conv2d):
        prune.l1_unstructured(module, name="weight", amount=0.3)
```

**Course reference:** `L3-M4 pruning/main.py`

### Step 7c — Quantization (FP32 → INT8)

Make model 4× smaller and 2-4× faster:

```python
import torch.quantization as quant

model_quantized = quant.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
```

**Course reference:** `L3-M4 quantization/main.py`

### Step 7d — Inference script

**Build this file:** `predict.py`

```python
# Usage: python predict.py path/to/image.jpg
# Output: "porn (95.2% confidence)"
```

Load ONNX model with `onnxruntime`, preprocess image, run inference, print result.

### Step 7e — Full pipeline (prune → quantize → benchmark)

**Course reference:** `L3-M4 metro_city/main.py` — end-to-end optimization pipeline

### After deployment — record final results

```
┌──────────────────────────────────────────────┐
│ DEPLOYMENT                                   │
│ Model format:     ONNX                       │
│ Original size:    XX MB (.pth)               │
│ ONNX size:        XX MB (.onnx)              │
│ Quantized size:   XX MB (INT8)               │
│ Inference speed:  XX ms per image            │
│ Accuracy kept:    XX.XX% (same as training?) │
└──────────────────────────────────────────────┘
```

---

## Progress Tracker

| Phase | Description | File to build | Course reference | Status |
|-------|------------|---------------|-----------------|--------|
| 1 | Build & train SimpleCNN | `main.py`, `cnn.py`, `data_loader.py` | L1-M2, L1-M3, L1-M4 | ✅ Done |
| 2a | Test-set evaluation | `evaluate.py` | — | ✅ Done |
| 2b | Confusion matrix | `evaluate.py` | L3-M4 `MLflow/main.py` | ✅ Done |
| 2c | Per-class precision/recall/F1 | `evaluate.py` | L2-M1 `learning_rate/main.py` | ✅ Done |
| 3 | Optuna hyperparameter tuning | `optuna.py` | L2-M1 `optuna/main.py` | 🔲 |
| 4 | Transfer learning (ResNet18/MobileNetV3) | `transfer_cnn.py` | L2-M2 `transfer_learning/main.py` | 🔲 |
| 5 | ResNet skip connections | modify `cnn.py` | L3-M1 `resnet/main.py` | 🔲 |
| 6 | Grad-CAM interpretability | `grad_cam.py` | L3-M2 `saliency_and_class_activation_map/main.py` | 🔲 |
| 7a | ONNX export | `export_onnx.py` | L3-M4 `ONNX/main.py` | 🔲 |
| 7b | Pruning | modify `cnn.py` | L3-M4 `pruning/main.py` | 🔲 |
| 7c | Quantization | modify `cnn.py` | L3-M4 `quantization/main.py` | 🔲 |
| 7d | Inference script | `predict.py` | L3-M4 `metro_city/main.py` | 🔲 |

---

## Key Principle

```
Every change is measured against a known baseline.

Baseline → evaluate → tune → evaluate → upgrade → evaluate → deploy → evaluate
              ▲                                              |
              └──────────── compare at each step ────────────┘
```

Never move to the next phase without recording your current results. That's how you learn what actually works.
