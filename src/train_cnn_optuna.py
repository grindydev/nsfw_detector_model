# ============================================================
# train_cnn_optuna.py - Training pipeline for Optuna-tuned model
# ============================================================
#
# Based on train results from Optuna study:
#   layers=5 | filters=[16, 16, 128, 64, 128] | lr=0.000345
#   dropout=0.358 | fc_size=256 | batch_size=16 | weight_decay=0.01
#
# Same structure as main.py but uses OptunaSimpleCNN + Optuna params

import copy
import torch
from torch import nn
from torch import optim
from torch.amp import autocast, GradScaler
from torch.utils.data import Subset, DataLoader

from data_loader import get_dataloaders
from cnn_optuna import OptunaSimpleCNN
import helper_utils
import mlflow

# ==================== CONFIG (EDIT ONLY THIS SECTION) ====================
CONFIG = {
    "mode": "test",                    # "test" = fast dev on Mac | "train" = full training on GTX 1650

    "device": "auto",

    "val_fraction": 0.15,
    "test_fraction": 0.2,

    # Optuna best params
    "lr": 0.000345,
    "weight_decay": 0.01,
    "label_smoothing": 0.1,
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR",

    # Settings used when mode = "test"
    "test": {
        "num_epochs": 5,
        "train_data_fraction": 0.05,
        "batch_size": 16,
        "patience": 3
    },

    # Settings used when mode = "train"
    "train": {
        "num_epochs": 40,
        "train_data_fraction": 1.0,
        "batch_size": 16,
        "patience": 8
    }
}

# ==================== APPLY CONFIG ====================
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

BEST_MODEL_PATH = f"models/best_optuna_cnn_{MODE}.pth"

print(f"🔧 CONFIG LOADED → Running in **{MODE.upper()} MODE**")
print(f"   Best model file: {BEST_MODEL_PATH} (saved immediately when improved)")

# ==================== DEVICE SETUP ====================
if CONFIG["device"] == "auto":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        is_cuda = True
        torch.backends.cudnn.benchmark = True
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
    device = torch.device(CONFIG["device"])
    is_cuda = (CONFIG["device"] == "cuda")
    print(f"✅ Using forced device from CONFIG: {device}")

# ==================== LOAD DATA ====================
train_loader, val_loader, test_loader, num_classes = get_dataloaders(
    batch_size=BATCH_SIZE,
    val_fraction=VAL_FRACTION,
    test_fraction=TEST_FRACTION,
    num_workers=4 if is_cuda else 1
)

print(f"✅ Final training set size: {len(train_loader.dataset)} images")

# ==================== MODEL, LOSS, OPTIMIZER, SCHEDULER ====================
model = OptunaSimpleCNN(num_classes=num_classes)

loss_function = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

# ==================== MIXED PRECISION SETUP ====================
use_amp = is_cuda
scaler = GradScaler() if use_amp else None

try:
    if is_cuda:
        model = torch.compile(model, mode="reduce-overhead")
        print("⚡ Model compiled with torch.compile() for extra speed")
except Exception:
    pass

# ==================== TRAINING FUNCTIONS ====================
def train_epoch(model, train_loader, loss_function, optimizer, device, scaler, use_amp):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast(device_type="cuda"):
                outputs = model(images)
                loss = loss_function(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(train_loader.dataset)


def validate_epoch(model, val_loader, loss_function, device):
    model.eval()
    running_val_loss = 0.0
    correct = total = 0

    with torch.no_grad():
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
mlflow.set_experiment("NSFW_Detector_Optuna")

# ==================== TRAINING LOOP ====================
def training_loop(model, train_loader, val_loader, loss_function, optimizer, scheduler,
                  num_epochs, device, scaler, use_amp):
    model.to(device)
    best_val_accuracy = 0.0
    best_model_state = None
    best_epoch = 0
    patience_counter = 0

    train_losses, val_losses, val_accuracies = [], [], []

    print("\n" + "="*70)
    print(f"🚀 TRAINING STARTED — {MODE.upper()} MODE (OptunaSimpleCNN)")
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

        if epoch_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_accuracy
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())

            clean_state = {
                k.replace("_orig_mod.", ""): v
                for k, v in model.state_dict().items()
            }

            torch.save({
                'model_state_dict': clean_state,
                'num_classes': num_classes,
                'val_accuracy': best_val_accuracy,
                'epoch': best_epoch,
                'mode': MODE,
                'model_type': 'OptunaSimpleCNN',
            }, BEST_MODEL_PATH)

            print(f"  → New best model saved to {BEST_MODEL_PATH} "
                  f"({best_val_accuracy:.2f}% at epoch {best_epoch})")

            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n⏹️ Early stopping after {patience_counter} epochs without improvement")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, [train_losses, val_losses, val_accuracies]


# ==================== RUN TRAINING ====================
with mlflow.start_run(run_name=f"OptunaSimpleCNN_{MODE}"):
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
        "model": "OptunaSimpleCNN",
        "architecture": "5 layers [16,16,128,64,128] dropout=0.358 fc=256",
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
