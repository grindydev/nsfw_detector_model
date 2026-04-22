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
├── roadmap.md                        ← You are here
├── documents/
│   └── phase5_residual_connections.md ← Skip connections learning doc
├── client/
│   ├── server.py                     ← FastAPI backend (ONNX inference)
│   ├── start.py                      ← One-command launcher
│   └── frontend/                     ← React frontend (upload + results)
├── src/
│   ├── main.py                       ← Phase 1: SimpleCNN training
│   ├── cnn.py                        ← Phase 1: SimpleCNN model
│   ├── cnn_tuned.py                  ← Phase 3: TunedCNN (Optuna-optimized)
│   ├── residual_cnn_tuned.py         ← Phase 5: ResidualTunedCNN
│   ├── train_tuned.py                ← Phase 3: TunedCNN training
│   ├── train_residual.py             ← Phase 5: ResidualTunedCNN training
│   ├── transfer_cnn.py               ← Phase 4: ResNet18 Strategy 1 (freeze all)
│   ├── transfer_cnn_finetune.py      ← Phase 4: ResNet18 Strategy 2 (fine-tune)
│   ├── transfer_cnn_fulltrain.py     ← Phase 4: ResNet18 Strategy 3 (full retrain)
│   ├── tuning.py                     ← Phase 3: Optuna hyperparameter search
│   ├── evaluate.py                   ← Phase 2: Test-set evaluation (all models)
│   ├── grad_cam.py                   ← Phase 6: Grad-CAM visualization
│   ├── export_onnx.py                ← Phase 7: ONNX export
│   ├── prune.py                      ← Phase 7: Pruning + Quantization pipeline
│   ├── data_loader.py                ← Shared data loading + transforms
│   └── helper_utils.py               ← Shared utilities + progress bars
└── data/
    └── nsfw_dataset_v1/              ← Dataset
```

---

## Learning Path

```
 ✅ Phase 1 — Build & Train SimpleCNN
    │
    ▼
 ✅ Phase 2 — Evaluate Baseline (test accuracy, confusion matrix, F1)
    │           → Know your starting point
    │           → Discover which classes are problematic
    │
    ▼
 ✅ Phase 3 — Optuna Tuning (push SimpleCNN to its limit)
    │           → Use Phase 2 insights to guide search space
    │           → Compare with Phase 2 baseline
    │
    ▼
 ✅ Phase 4 — Transfer Learning (biggest accuracy jump)
    │           → ResNet18: 3 strategies (freeze/fine-tune/full retrain)
    │           → Compare with Phase 2 + Phase 3 results
    │
    ▼
 ✅ Phase 5 — ResNet Skip Connections
    │           → ResidualBlock: F(x) + x
    │           → Understand why deeper networks work
    │
    ▼
 ✅ Phase 6 — Model Interpretability (Grad-CAM)
    │           → See where the model looks to make predictions
    │           → Compare SimpleCNN vs ResNet18 heatmaps
    │
    ▼
 ✅ Phase 7 — Export & Deployment (ONNX → Prune → Quantize → Web App)
                → Export to ONNX, deploy with FastAPI + React
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
- Input 128×128 — fast iteration with limited VRAM, adequate for NSFW patterns
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
| `torch.compile()` portability fix | `main.py` — strip `_orig_mod.` prefix before saving | Training with compile wraps model — must clean state_dict keys for cross-device compatibility |
| Auto-detect model type | `evaluate.py` — detects SimpleCNN, TunedCNN, ResNet18 from checkpoint | One evaluate.py works for ALL models — no hardcoded model class |

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

### Key takeaways for Phase 3
- Model is **underfitting** — 3 conv blocks may be too shallow
- `neutral` ↔ `sexy` ↔ `porn` confusion → deeper model may help distinguish
- `hentai` recall is low (0.57) → model misses many hentai images
- Optuna should search: more conv blocks, wider channels, lower dropout

---

## Phase 3 — Optuna Hyperparameter Tuning ✅

**Goal:** Push SimpleCNN to its limit by finding the best hyperparameters.

**Built in:** `tuning.py`

### What you practiced

| Concept | Where in your code | What you learned |
|---------|-------------------|-----------------|
| Flexible CNN architecture | `tuning.py` — `FlexibleCNN` with dynamic layers, filters, kernels | Model structure doesn't have to be fixed — can be controlled by parameters |
| Optuna objective function | `tuning.py` — `objective(trial)` | Each trial samples hyperparameters, trains, returns a score — Optuna maximizes it |
| Search space design | `tuning.py` — `trial.suggest_int`, `suggest_float`, `suggest_categorical` | Different param types need different suggest methods; log scale for lr |
| `AdaptiveAvgPool2d` for variable-depth models | `tuning.py` — classifier starts with `AdaptiveAvgPool2d((1,1))` | Variable conv layers → variable spatial size → must squash to fixed size before Linear |
| MaxPool2d spatial shrinking | Bug: 5 layers × MaxPool2d → 2×2 → kernel doesn't fit → crash | Each MaxPool2d halves spatial size: 128→64→32→16→8→4. Too many layers = too small |
| BatchNorm in conv blocks | `tuning.py` — `nn.BatchNorm2d(out_channels)` | Stabilizes training, especially important when trying many different architectures |
| Optuna study + trials | `tuning.py` — `optuna.create_study`, `study.optimize` | Study = container for trials; direction='maximize' tells Optuna to find highest accuracy |
| Saving Optuna study | `tuning.py` — `joblib.dump(study, ...)` | Can reload later with `joblib.load()` to analyze without retraining |
| Retraining best model | `tuning.py` — rebuild with best params, train again, save checkpoint | Optuna only remembers hyperparameters, not the model. Must retrain with best params |
| Best accuracy per trial | `tuning.py` — track best val accuracy across epochs | Return peak accuracy, not final epoch — model might peak early then drop |
| GPU-friendly filters | `tuning.py` — FILTER_CHOICES = [32, 64, 128, 256] | Powers of 2 are optimized on GPU; random values like 37 waste compute |
| Ordered filter progression | Search space insight | Filters should increase monotonically: 32→64→128→256, not chaotic 16→128→64→128 |
| `num_workers=4` on CUDA | `data_loader.py` — `num_workers` parameter | CPU pre-loads batches in parallel while GPU trains → no idle GPU time |
| `train_fraction` for fast trials | `data_loader.py` — subset training data | Optuna doesn't need full data to rank configs; 30-50% is enough for comparison |
| Dropout must use sampled value | Bug: hardcoded `p=0.6` instead of `p=self.dropout_rate` | If dropout isn't wired to Optuna's suggestion, it wastes trials searching a value that's ignored |

### Key takeaways for Phase 4
- Optuna pushed SimpleCNN architecture to its limit
- Transfer learning (Phase 4) will likely give the biggest accuracy jump
- Compare Optuna's best vs transfer learning to see if custom CNN can compete with pretrained

---

## Phase 4 — Transfer Learning ✅

**Goal:** Use a pretrained model for the biggest accuracy jump.

**Built in:** `transfer_cnn.py`, `transfer_cnn_finetune.py`, `transfer_cnn_fulltrain.py`

### What you practiced

| Concept | Where in your code | What you learned |
|---------|-------------------|-----------------|
| Pretrained model loading | `transfer_cnn.py` — `tv_models.resnet18(weights=...)` | Don't train from scratch — reuse ImageNet knowledge (1.2M images) |
| Strategy 1: Feature Extraction | `transfer_cnn.py` — freeze all, train only fc | Fastest, trains only ~2.5K params, good baseline |
| Strategy 2: Fine-tuning | `transfer_cnn_finetune.py` — freeze early, train layer4 + fc | Best balance — reuses general features, adapts task-specific ones |
| Strategy 3: Full Retraining | `transfer_cnn_fulltrain.py` — train everything | All 11.7M params trainable, risk of overfitting on 28K images |
| Freezing layers | `transfer_cnn.py` — `param.requires_grad = False` | Frozen layers keep pretrained weights, unfrozen layers adapt to NSFW task |
| ImageNet preprocessing | All 3 files — 224×224, ImageNet mean/std | Pretrained weights expect specific input normalization — mismatch = garbage predictions |
| Model must be on device after replacing fc | Bug: `.to(device)` before fc replacement leaves new fc on CPU | New layers are created on CPU — must call .to(device) AFTER replacing layers |
| Lower LR for transfer | `transfer_cnn_finetune.py` — lr=1e-5 | High LR destroys pretrained ImageNet weights in first epoch |
| Overfitting detection | ResNet18 val=87% → test=79% (8.5% gap) | Bigger model + small dataset = overfitting. SimpleCNN (80% test) generalized better |

### Transfer learning results

```
┌──────────────────────────────────────────────────┐
│ TRANSFER LEARNING COMPARISON                    │
│                                                  │
│ Strategy 1 (freeze all):      75.93% val         │
│ Strategy 2 (fine-tune):       87.71% val         │
│ Strategy 3 (full retrain):    (results pending)  │
│                                                  │
│ ResNet18 test:                79.16% (overfits!)  │
│ SimpleCNN test:               80.32% (better!)    │
│                                                  │
│ Key insight: bigger model ≠ better on small data │
│ 11.7M params vs 28K images → overfitting         │
└──────────────────────────────────────────────────┘
```

### Key takeaway
- Transfer learning gave biggest val accuracy (87%) but overfit on test (79%)
- SimpleCNN (80%) generalized better — right-sized model for 28K images
- Optuna-tuned custom CNN could beat ResNet18 on this specific dataset

---

## Phase 5 — ResNet Skip Connections ✅

**Goal:** Understand WHY deeper networks work better by adding residual connections.

**Built in:** `residual_cnn_tuned.py`, `train_residual.py`

### What you practiced

| Concept | Where in your code | What you learned |
|---------|-------------------|-----------------|
| ResidualBlock with skip connection | `residual_cnn_tuned.py` — `ResidualBlock` | `out = F(x) + x` — model only learns the DELTA from identity |
| Two convolutions per block | `residual_cnn_tuned.py` — conv1 + conv2 | Standard ResNet pattern: richer residual before skip |
| MaxPool AFTER skip connection | Bug: MaxPool before skip caused shape mismatch crash | Skip operates on same spatial size, downsample after |
| Shortcut projection | `residual_cnn_tuned.py` — 1×1 conv when channels change | Can't add 32 channels to 64 channels — 1×1 conv matches them |
| Useless layers become identity | Residual theory | If F(x)=0, out=0+x=x — layer becomes transparent |
| Vanishing gradients | Theory from `documents/phase5_residual_connections.md` | Without skip: gradients fade to 0 by layer 1. With skip: direct gradient path to every layer |
| Going deeper is now possible | ResidualBlock unlocks depth | 5 layers with skip ~87%. 8-12 layers could reach 90%+ |

### Architecture

```
ResidualTunedCNN: 5 ResidualBlocks [32→64→128→128→256]
Each block: Conv→BN→ReLU→Conv→BN → (+skip) → ReLU → MaxPool
Dropout: 0.358 (from Optuna)
FC: 256→256→5
```

### Key takeaway
- Skip connections don't automatically improve accuracy at 5 layers
- They UNLOCK going deeper (8, 12, 18 layers) without vanishing gradients
- Your custom ResidualTunedCNN could compete with ResNet18 from scratch
- See `documents/phase5_residual_connections.md` for full learning document

---

## Phase 6 — Model Interpretability ✅

**Goal:** See what the model has learned and where it looks.

**Built in:** `grad_cam.py`

### What you practiced

| Concept | Where in your code | What you learned |
|---------|-------------------|-----------------|
| Grad-CAM heatmap generation | `grad_cam.py` — `GradCAM` class | Shows WHERE the model focuses for each prediction (red=important, blue=ignored) |
| Forward/backward hooks | `grad_cam.py` — `register_forward_hook`, `register_full_backward_hook` | Capture activations and gradients during forward/backward pass without modifying model |
| Comparing models visually | `grad_cam.py` — side-by-side SimpleCNN vs ResNet18 heatmaps | ResNet18 focuses on meaningful regions, SimpleCNN scatters attention |
| Per-class grid visualization | `grad_cam.py` — `generate_class_grid()` | 5 images per class × 3 columns (original, SimpleCAM, ResNet18 CAM) |
| Sensitive content handling | `grad_cam.py` — grayscale + adjustable blur | Can visualize model behavior without viewing explicit content |
| Target layer selection | SimpleCNN: `conv_block3.block[0]`, ResNet18: `layer4[1].conv2` | Must pick the last conv layer — that's where spatial info exists |

### Key takeaway
- Grad-CAM reveals WHY models make mistakes (looking at wrong regions)
- ResNet18's focused heatmaps explain its higher accuracy
- Your SimpleCNN's scattered heatmaps explain its confusion between similar classes

---

## Phase 7 — Export & Deployment ✅

**Goal:** Ship your model — export, shrink, and benchmark.

**Built in:** `export_onnx.py`, `prune.py`, `client/` (FastAPI + React)

### What you practiced

| Concept | Where in your code | What you learned |
|---------|-------------------|-----------------|
| ONNX export | `export_onnx.py` — `torch.onnx.export()` | .pth → .onnx makes model portable to any language/platform |
| ONNX is like PDF | `client/server.py` — `onnxruntime` loads model | .pth needs PyTorch (GB of deps), .onnx needs only onnxruntime (~50MB) |
| Pruning (L1 unstructured) | `prune.py` — `prune.l1_unstructured()` | Remove smallest weights → sparser model → faster inference |
| Quantization (FP32→INT8) | `prune.py` — `quantize_dynamic()` | 4× smaller weights, 2-4× faster, ~1% accuracy loss |
| Pruning on small models | 30% prune → 22% accuracy drop | Small models can't afford aggressive pruning — already minimal |
| Fine-tuning after pruning | `prune.py` — 3 epochs recovered 76% from 57% | Pruning damages learned features, fine-tuning recovers them |
| FastAPI backend | `client/server.py` — `/predict` endpoint | Standard industry pattern: React → FastAPI → ONNX → prediction |
| React frontend | `client/frontend/` — upload + confidence bars | Drag & drop upload, shows prediction + all class probabilities |
| One-command deployment | `client/start.py` | Spawns backend + frontend, handles Ctrl+C gracefully |
| Preprocessing must match training | `server.py` — INPUT_SIZE=128, NSFW mean/std | Wrong normalization = garbage predictions. #1 deployment bug |
| ONNX works in JavaScript too | onnxruntime-web, onnxruntime-node | ONNX is cross-platform: Python, JS, C++, Java, browser |
| Model size comparison | SimpleCNN=648KB, TunedCNN=2.3MB, ResNet18=43MB | Size grows quadratically with channel count, not linearly with layers |

### Deployment results

```
┌──────────────────────────────────────────────┐
│ DEPLOYMENT                                   │
│ Model format:     ONNX (29 KB)               │
│ Original size:    648 KB (.pth)               │
│ ONNX size:        29 KB (.onnx)               │
│ Pruned ONNX:      26 KB (30% pruned)          │
│ Accuracy:         80.32% (SimpleCNN test)     │
│ Frontend:         React (port 3000)           │
│ Backend:          FastAPI (port 8000)         │
│ Note:             Model too small for pruning │
│                   to provide meaningful gain  │
└──────────────────────────────────────────────┘
```

### Key takeaway
- Pruning/quantization matter for BIG models (>50MB). Small models are already optimal.
- ONNX + FastAPI + React is the standard ML deployment stack
- Always match training preprocessing in deployment

---

## Progress Tracker

| Phase | Description | File to build | Course reference | Status |
|-------|------------|---------------|-----------------|--------|
| 1 | Build & train SimpleCNN | `main.py`, `cnn.py`, `data_loader.py` | L1-M2, L1-M3, L1-M4 | ✅ Done |
| 2a | Test-set evaluation | `evaluate.py` | — | ✅ Done |
| 2b | Confusion matrix | `evaluate.py` | L3-M4 `MLflow/main.py` | ✅ Done |
| 2c | Per-class precision/recall/F1 | `evaluate.py` | L2-M1 `learning_rate/main.py` | ✅ Done |
| 3 | Optuna hyperparameter tuning | `tuning.py` | L2-M1 `optuna/main.py` | ✅ Done |
| 4 | Transfer learning (ResNet18) | `transfer_cnn.py`, `transfer_cnn_finetune.py`, `transfer_cnn_fulltrain.py` | L2-M2 `transfer_learning/main.py` | ✅ Done |
| 5 | ResNet skip connections | `residual_cnn_tuned.py`, `train_residual.py` | L3-M1 `resnet/main.py` | ✅ Done |
| 6 | Grad-CAM interpretability | `grad_cam.py` | L3-M2 `saliency_and_class_activation_map/main.py` | ✅ Done |
| 7a | ONNX export | `export_onnx.py` | L3-M4 `ONNX/main.py` | ✅ Done |
| 7b | Pruning | `prune.py` | L3-M4 `pruning/main.py` | ✅ Done |
| 7c | Quantization | `prune.py` | L3-M4 `quantization/main.py` | ✅ Done |
| 7d | Web deployment | `client/server.py`, `client/frontend/` | — | ✅ Done |

---

## Key Principle

```
Every change is measured against a known baseline.

Baseline → evaluate → tune → evaluate → upgrade → evaluate → deploy → evaluate
              ▲                                              |
              └──────────── compare at each step ────────────┘
```

Never move to the next phase without recording your current results. That's how you learn what actually works.

---

## Final Results Summary

```
┌──────────────────────────────────────────────────────────┐
│ MODEL COMPARISON (Test Accuracy)                         │
│                                                          │
│ SimpleCNN baseline (3 layers, 5 epochs):      64.36%    │
│ SimpleCNN trained (3 layers, 40 epochs):      80.32%    │
│ TunedCNN (5 layers, Optuna):                  ~87% val  │
│ ResidualTunedCNN (5 residual blocks):         (pending) │
│ ResNet18 Strategy 1 (freeze all):             75.93% val │
│ ResNet18 Strategy 2 (fine-tune):              87.71% val │
│ ResNet18 test:                                79.16%    │
│                                                          │
│ Key insight: SimpleCNN (80.32% test) beat ResNet18       │
│ (79.16% test) on this dataset. Right-sized model wins.   │
│                                                          │
│ Deployed as ONNX (29 KB) via FastAPI + React web app.    │
└──────────────────────────────────────────────────────────┘
```
