# ============================================================
# main.py - CONFIG-DRIVEN VERSION (Test Mode + Train Mode)
# ============================================================
#
# EDUCATIONAL VERSION - Designed for you to learn PyTorch training!
#
# This script is now a complete, clean, and flexible training pipeline.
# Everything important is controlled from the CONFIG dictionary at the top.
# Every major section has detailed comments explaining:
#   • WHAT the code does
#   • WHY we do it this way
#   • HOW it helps learning / performance
#
# Key learning concepts covered:
#   • Config-driven design (easy to switch between testing and full training)
#   • Immediate best-model checkpointing
#   • Device-aware training (Mac Mini MPS vs Ubuntu GTX 1650)
#   • Mixed Precision (AMP) for speed
#   • Subset training for fast testing
#   • Early stopping + Cosine LR scheduler

import copy
import torch
from torch import nn
from torch import optim
from torch.amp import autocast, GradScaler
from torch.utils.data import Subset, DataLoader

from data_loader import get_dataloaders
from cnn import SimpleCNN
import helper_utils
import mlflow

# ==================== CONFIG (EDIT ONLY THIS SECTION) ====================
# This is the only place you need to change settings.
# It makes the code very readable and reusable across your Mac Mini and Ubuntu PC.
CONFIG = {
    "mode": "test",                    # ← Change to "train" when you want full training on Ubuntu
                                       # "test"  = fast development on Mac Mini
                                       # "train" = full serious training on GTX 1650

    "device": "auto",                  # "auto", "cuda", "mps", or "cpu"
                                       # "auto" is recommended - it picks the best available hardware

    "val_fraction": 0.15,              # Fraction of dataset used for validation (15%)
    "test_fraction": 0.2,              # Fraction of dataset used for testing (20%)

    "lr": 1e-3,                        # Learning rate
    "weight_decay": 0.05,               # L2 regularization (AdamW handles this properly)
    "label_smoothing": 0.1,             # Prevents overconfident predictions
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR",

    # Settings used when mode = "test" (fast development)
    "test": {
        "num_epochs": 5,               # Very few epochs so you can test changes quickly
        "train_data_fraction": 0.05,   # Use only 5% of training images → much faster on Mac
        "batch_size": 64,              # Smaller batch fits easily in Mac memory
        "patience": 3                  # Stop early if no improvement
    },

    # Settings used when mode = "train" (full training)
    "train": {
        "num_epochs": 40,              # Enough epochs for the model to learn well
        "train_data_fraction": 1.0,    # Use 100% of the training data
        "batch_size": 64,              # Larger batch = better GPU utilization on GTX 1650
        "patience": 8                  # More patience during long training
    }
}

# ==================== APPLY CONFIG (you don't need to change anything below) ====================
MODE = CONFIG["mode"]
SETTINGS = CONFIG[MODE]

NUM_EPOCHS = SETTINGS["num_epochs"]
TRAIN_DATA_FRACTION = SETTINGS["train_data_fraction"]
BATCH_SIZE = SETTINGS["batch_size"]
PATIENCE = SETTINGS["patience"]
VAL_FRACTION = CONFIG["val_fraction"]
TEST_FRACTION = CONFIG["test_fraction"]
LR = CONFIG["lr"]
WEIGHT_DECAY = CONFIG["weight_decay"]
LABEL_SMOOTHING = CONFIG["label_smoothing"]

# File where the best model will be saved (updated live during training)
BEST_MODEL_PATH = f"models/best_simple_cnn_{MODE}.pth"

print(f"🔧 CONFIG LOADED → Running in **{MODE.upper()} MODE**")
print(f"   Best model file: {BEST_MODEL_PATH} (saved immediately when improved)")

# ==================== DEVICE SETUP ====================
# Why this matters: You can develop on Mac Mini (MPS) and train on Ubuntu (CUDA)
# without changing any code except the CONFIG.
if CONFIG["device"] == "auto":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        is_cuda = True
        torch.backends.cudnn.benchmark = True          # Speeds up convolution operations on NVIDIA
        print("🚀 Auto-detected NVIDIA GPU (GTX 1650) → using CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        is_cuda = False
        print("🍎 Auto-detected Apple Silicon → using MPS")
    else:
        device = torch.device("cpu")
        is_cuda = False
        print("⚠️  Auto-detected CPU only")
else:
    # User forced a specific device from CONFIG
    device = torch.device(CONFIG["device"])
    is_cuda = (CONFIG["device"] == "cuda")
    print(f"✅ Using forced device from CONFIG: {device}")

# ==================== LOAD DATA ====================
# get_dataloaders() is from your data_loader.py
# We pass the fractions from CONFIG so you can control dataset split easily.
train_loader, val_loader, test_loader, num_classes = get_dataloaders(
    batch_size=BATCH_SIZE,
    val_fraction=VAL_FRACTION,
    test_fraction=TEST_FRACTION
)

print(f"✅ Final training set size: {len(train_loader.dataset)} images")

# ==================== MODEL, LOSS, OPTIMIZER, SCHEDULER ====================
model = SimpleCNN(num_classes=num_classes)

# CrossEntropyLoss with label smoothing prevents the model from becoming over-confident
loss_function = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

# AdamW is better than Adam when using weight decay (stronger regularization)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# Cosine Annealing gradually reduces learning rate → helps converge better
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

# ==================== MIXED PRECISION SETUP ====================
# Mixed Precision (AMP) makes training much faster and uses less memory on GPU
use_amp = is_cuda                                      # Full AMP only works reliably on CUDA
scaler = GradScaler() if use_amp else None

# torch.compile() can give extra speed (PyTorch 2.0+ feature)
try:
    if is_cuda:
        model = torch.compile(model, mode="reduce-overhead")
        print("⚡ Model compiled with torch.compile() for extra speed")
except Exception:
    pass

# ==================== TRAINING FUNCTIONS ====================
def train_epoch(model, train_loader, loss_function, optimizer, device, scaler, use_amp):
    model.train()                                      # Enable Dropout + BatchNorm training behavior
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)          # Faster than zero_grad()

        if use_amp:
            # Automatic Mixed Precision: most math runs in FP16 → faster + less VRAM
            with autocast(device_type="cuda"):
                outputs = model(images)
                loss = loss_function(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Stability
            scaler.step(optimizer)
            scaler.update()
        else:
            # Normal full-precision training (MPS or CPU)
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(train_loader.dataset)


def validate_epoch(model, val_loader, loss_function, device):
    model.eval()                                       # Disable Dropout, use running BatchNorm stats
    running_val_loss = 0.0
    correct = total = 0

    with torch.no_grad():                              # No gradients = faster and less memory
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            with autocast(device_type=device.type if device.type != "mps" else "cpu"):
                outputs = model(images)
                val_loss = loss_function(outputs, labels)

            running_val_loss += val_loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return (running_val_loss / len(val_loader.dataset)), (100.0 * correct / total)


# ==================== SETUP MLFLOW ====================
mlflow.set_experiment("NSFW_Detector")

# ==================== TRAINING LOOP WITH IMMEDIATE BEST-MODEL SAVING ====================
def training_loop(model, train_loader, val_loader, loss_function, optimizer, scheduler,
                  num_epochs, device, scaler, use_amp):
    model.to(device)
    best_val_accuracy = 0.0
    best_model_state = None
    best_epoch = 0
    patience_counter = 0

    train_losses, val_losses, val_accuracies = [], [], []

    print("\n" + "="*70)
    print(f"🚀 TRAINING STARTED - {MODE.upper()} MODE")
    print(f"Device: {device} | Epochs: {num_epochs} | Best model saved live to {BEST_MODEL_PATH}")
    print("="*70)

    for epoch in range(num_epochs):
        epoch_loss = train_epoch(model, train_loader, loss_function, optimizer, device, scaler, use_amp)
        epoch_val_loss, epoch_accuracy = validate_epoch(model, val_loader, loss_function, device)

        train_losses.append(epoch_loss)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_accuracy)

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
              f"Train Loss: {epoch_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Val Acc: {epoch_accuracy:6.2f}% | LR: {current_lr:.6f}")

        mlflow.log_metric("train_loss", epoch_loss, step=epoch)
        mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", epoch_accuracy, step=epoch)

        scheduler.step()

        # === IMPORTANT LEARNING POINT: Save best model IMMEDIATELY ===
        # We save the model to disk every time validation accuracy improves.
        # This is called "checkpointing". It means you always have the best version,
        # even if training is interrupted or you stop early.
        if epoch_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_accuracy
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())

            # Strip _orig_mod. prefix from torch.compile() so checkpoint
            # is portable to machines without compile (Mac MPS/CPU)
            clean_state = {
                k.replace("_orig_mod.", ""): v
                for k, v in model.state_dict().items()
            }

            # Save to disk right now
            torch.save({
                'model_state_dict': clean_state,           # clean weights, no compile prefix
                'num_classes': num_classes,
                'val_accuracy': best_val_accuracy,
                'epoch': best_epoch,
                'mode': MODE,
            }, BEST_MODEL_PATH)

            print(f"  → New best model saved to {BEST_MODEL_PATH} "
                  f"({best_val_accuracy:.2f}% at epoch {best_epoch})")

            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n⏹️ Early stopping after {patience_counter} epochs without improvement")
                break

    # Load the best weights back into the model for plotting and further use
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, [train_losses, val_losses, val_accuracies]


# ==================== RUN TRAINING ====================
with mlflow.start_run(run_name=f"SimpleCNN_{MODE}"):
    mlflow.log_params({
        "mode": MODE,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "label_smoothing": LABEL_SMOOTHING,
        "patience": PATIENCE,
        "optimizer": CONFIG["optimizer"],
        "scheduler": CONFIG["scheduler"],
    })

    trained_model, training_metrics = training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=NUM_EPOCHS,
        device=device,
        scaler=scaler,
        use_amp=use_amp
    )

    helper_utils.plot_training_metrics(training_metrics)
    mlflow.log_artifact(BEST_MODEL_PATH)

    print(f"\n✅ Training finished in {MODE.upper()} mode!")
    print(f"   Best Validation Accuracy: {max(training_metrics[2]):.2f}%")
    print(f"   The best model was saved live to: {BEST_MODEL_PATH}")