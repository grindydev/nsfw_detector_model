# ============================================================
# test_model.py - Load model trained on Ubuntu GTX 1650 → Run on Mac Mini M1
# ============================================================

import torch
from src.cnn import SimpleCNN
from src.data_loader import get_dataloaders
from pathlib import Path

# ==================== PATH TO YOUR DOWNLOADED MODEL ====================
MODEL_PATH = Path.cwd() / 'data/models/best_simple_cnn.pth'

# ==================== LOAD CHECKPOINT ====================
# map_location='cpu' is important when loading a CUDA-trained model on Mac
checkpoint = torch.load(MODEL_PATH, map_location='cpu')

print(f"✅ Model loaded successfully from: {MODEL_PATH.name}")
print(f"   Classes      : {checkpoint['num_classes']}")
print(f"   Val Accuracy : {checkpoint['val_accuracy']:.2f}%")
print(f"   Best epoch   : {checkpoint['epoch']}")

# ==================== RECREATE MODEL & LOAD WEIGHTS ====================
model = SimpleCNN(num_classes=checkpoint['num_classes'])

# This is the most important line you were missing!
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()                                      # Important: set to evaluation mode
print("   Model is now in eval mode (ready for inference)")

# ==================== DEVICE (Auto for Mac Mini M1) ====================
device = torch.device("cuda" if torch.cuda.is_available() else 
                     "mps" if torch.backends.mps.is_available() else "cpu")

model.to(device)
print(f"   Using device: {device}")

# ==================== TEST ON FULL TEST SET ====================
_, _, test_loader, _ = get_dataloaders(
    batch_size=32,
    val_fraction=0.15,
    test_fraction=0.2
)

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100.0 * correct / total
print(f"\n📊 Test Accuracy on full test set: {test_accuracy:.2f}%")