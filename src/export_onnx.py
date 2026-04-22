import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
import torchvision.models as tv_models
from cnn import SimpleCNN
import helper_utils
from pathlib import Path


# ==================== RECREATE MODEL & LOAD WEIGHTS ====================
MODEL_PATH = Path.cwd() / 'models/best_simple_cnn_train.pth'

checkpoint = torch.load(MODEL_PATH, map_location='cpu')

print(f"✅ Model loaded successfully from: {MODEL_PATH.name}")
print(f"   Classes      : {checkpoint['num_classes']}")
print(f"   Val Accuracy : {checkpoint['val_accuracy']:.2f}%")
print(f"   Best epoch   : {checkpoint['epoch']}")

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

print("Model: ", model)

# ==================== EXPORT MODEL ====================
# Create dummy input
dummy_input = torch.randn(1, 3, 128, 128, device='cpu')

# Export
torch.onnx.export(
    model, dummy_input,
    "nsfw_detector.onnx",
    export_params=True,                         # save trained weights
    opset_version=13,                           # stable version (avoids old bugs)
    do_constant_folding=True,                   # optimization
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={                              # allow different batch sizes
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
print(" Exported successfully as nsfw_detector.onnx")

