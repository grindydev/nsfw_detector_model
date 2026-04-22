import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
import torchvision.models as tv_models
from cnn import SimpleCNN
from cnn_tuned import TunedCNN
from residual_cnn_tuned import ResidualTunedCNN
import helper_utils
from pathlib import Path


# ==================== CONFIG ====================
import sys

# Default model to export, or pass as argument:
#   python export_onnx.py                          → best model
#   python export_onnx.py best_transfer_cnn.pth    → specific model
#   python export_onnx.py best_simple_cnn_train.pth
#   python export_onnx.py best_optuna_cnn_train.pth
MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'

if len(sys.argv) > 1:
    MODEL_FILENAME = sys.argv[1]
else:
    # List available models and pick the best one
    candidates = ['best_optuna_cnn_train.pth', 'best_transfer_cnn.pth',
                  'best_simple_cnn_train.pth', 'best_residual_cnn_train.pth']
    MODEL_FILENAME = next(
        (c for c in candidates if (MODELS_DIR / c).exists()),
        None
    )
    if MODEL_FILENAME is None:
        # Fall back to any .pth file
        pth_files = list(MODELS_DIR.glob('*.pth'))
        if not pth_files:
            print(f"❌ No .pth files found in {MODELS_DIR}")
            sys.exit(1)
        MODEL_FILENAME = pth_files[0].name

MODEL_PATH = MODELS_DIR / MODEL_FILENAME

# List available models
print("Available models:")
for f in sorted(MODELS_DIR.glob('*.pth')):
    marker = '  ← selected' if f.name == MODEL_FILENAME else ''
    print(f"  {f.name} ({f.stat().st_size / 1024:.0f} KB){marker}")
print()

checkpoint = torch.load(MODEL_PATH, map_location='cpu')

print(f"✅ Model loaded successfully from: {MODEL_PATH.name}")
print(f"   Classes      : {checkpoint['num_classes']}")
print(f"   Val Accuracy : {checkpoint['val_accuracy']:.2f}%")
print(f"   Best epoch   : {checkpoint['epoch']}")

state_dict = checkpoint['model_state_dict']
num_classes = checkpoint['num_classes']

# Detect model type from checkpoint keys
if 'best_params' in checkpoint:
    # Optuna FlexibleCNN — rebuild with best params
    from tuning import FlexibleCNN
    best = checkpoint['best_params']
    n_layers = best['n_layers']
    n_filters = [best[f'n_filters_{i}'] for i in range(n_layers)]
    kernel_sizes = [3] * n_layers
    model = FlexibleCNN(n_layers, n_filters, kernel_sizes,
                        best['dropout_rate'], best['fc_size'], num_classes)
    input_size = 128
    model_type = "FlexibleCNN (Optuna)"
    print(f"   Model type  : {model_type}")
elif 'conv1.weight' in state_dict:
    # ResNet18 (transfer learning)
    model = tv_models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    input_size = 224
    model_type = "ResNet18 (Transfer)"
    print(f"   Model type  : {model_type}")
elif any('shortcut' in k for k in state_dict):
    # ResidualTunedCNN (has skip connection projections)
    model = ResidualTunedCNN(num_classes=num_classes)
    input_size = 224
    model_type = "ResidualTunedCNN"
    print(f"   Model type  : {model_type}")
elif 'conv_block5.block.0.weight' in state_dict:
    # TunedCNN (5 layers)
    model = TunedCNN(num_classes=num_classes)
    input_size = 224
    model_type = "TunedCNN (Optuna)"
    print(f"   Model type  : {model_type}")
else:
    # SimpleCNN (3 layers)
    model = SimpleCNN(num_classes=num_classes)
    input_size = 128
    model_type = "SimpleCNN"
    print(f"   Model type  : {model_type}")

model.load_state_dict(state_dict)
model.eval()

print(f"\n{model}\n")

# ==================== EXPORT MODEL ====================
OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'models'
OUTPUT_DIR.mkdir(exist_ok=True)
onnx_path = OUTPUT_DIR / 'nsfw_detector.onnx'

print(f"Exporting to ONNX (input_size={input_size})...")

# Create dummy input matching model's expected size
dummy_input = torch.randn(1, 3, input_size, input_size, device='cpu')

torch.onnx.export(
    model, dummy_input,
    str(onnx_path),
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

# Write model type as ONNX metadata (so server/UI knows what model is loaded)
import onnx
onnx_model = onnx.load(str(onnx_path), load_external_data=False)
meta = onnx_model.metadata_props.add()
meta.key = "model_type"
meta.value = model_type
meta2 = onnx_model.metadata_props.add()
meta2.key = "input_size"
meta2.value = str(input_size)
onnx.save(onnx_model, str(onnx_path))

total_kb = onnx_path.stat().st_size / 1024
data_path = onnx_path.parent / (onnx_path.name + '.data')
if data_path.exists():
    total_kb += data_path.stat().st_size / 1024
print(f"✅ Exported: {onnx_path.name} ({total_kb:.0f} KB total, {model_type})")

