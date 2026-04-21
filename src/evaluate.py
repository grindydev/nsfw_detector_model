# ============================================================
# test_model.py - Load model trained on Ubuntu GTX 1650 → Run on Mac Mini M1
# ============================================================

import torch
import torch.nn as nn
import torchvision.models as tv_models
from cnn import SimpleCNN
from data_loader import get_dataloaders
from pathlib import Path

from helper_utils import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report 

# ==================== PATH TO YOUR DOWNLOADED MODEL ====================
MODEL_PATH = Path.cwd() / 'models/best_simple_cnn_train.pth'

# ==================== LOAD CHECKPOINT ====================
# map_location='cpu' is important when loading a CUDA-trained model on Mac

checkpoint = torch.load(MODEL_PATH, map_location='cpu')

print(f"✅ Model loaded successfully from: {MODEL_PATH.name}")
print(f"   Classes      : {checkpoint['num_classes']}")
print(f"   Val Accuracy : {checkpoint['val_accuracy']:.2f}%")
print(f"   Best epoch   : {checkpoint['epoch']}")

# ==================== RECREATE MODEL & LOAD WEIGHTS ====================
state_dict = checkpoint['model_state_dict']
num_classes = checkpoint['num_classes']

# Detect model type from checkpoint keys
if 'best_params' in checkpoint:
    # Optuna FlexibleCNN — need to rebuild with best params
    from tuning import FlexibleCNN
    best = checkpoint['best_params']
    n_layers = best['n_layers']
    n_filters = [best[f'n_filters_{i}'] for i in range(n_layers)]
    kernel_sizes = [3] * n_layers
    model = FlexibleCNN(n_layers, n_filters, kernel_sizes,
                        best['dropout_rate'], best['fc_size'], num_classes)
    print(f"   Model type  : FlexibleCNN (Optuna)")
elif 'conv1.weight' in state_dict:
    # ResNet18 (transfer learning)
    model = tv_models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    print(f"   Model type  : ResNet18 (Transfer Learning)")
else:
    # SimpleCNN
    model = SimpleCNN(num_classes=num_classes)
    print(f"   Model type  : SimpleCNN")

model.load_state_dict(state_dict)

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

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100.0 * correct / total
print(f"\n📊 Test Accuracy on full test set: {test_accuracy:.2f}%")

cm = confusion_matrix(all_labels, all_preds)
CLASS_NAMES = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']                                                                                                                                       

print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES)) 

plot_confusion_matrix(cm, CLASS_NAMES)


