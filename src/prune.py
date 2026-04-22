# ============================================================
# prune.py — Pruning + Quantization (Phase 7b + 7c)
# ============================================================
#
# Learn these techniques here, use them on bigger models later.
# Your SimpleCNN is only 29KB so the gains are minimal,
# but the code pattern works for any model size.
#
# What this script does:
#   1. Load trained model
#   2. Measure baseline accuracy
#   3. Apply pruning (remove small weights)
#   4. Measure pruned accuracy
#   5. Fine-tune to recover accuracy (if needed)
#   6. Apply quantization (FP32 → INT8)
#   7. Measure quantized accuracy
#   8. Export to ONNX
#   9. Benchmark: size, speed, accuracy comparison

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import time
import warnings
from pathlib import Path

from cnn import SimpleCNN
from data_loader import get_dataloaders

warnings.filterwarnings("ignore")  # suppress PyTorch/ONNX warnings

# ==================== CONFIG ====================
CHECKPOINT_PATH = Path.cwd() / 'models' / 'best_simple_cnn_train.pth'
OUTPUT_DIR = Path.cwd() / 'models'

# Pruning settings
PRUNE_AMOUNT = 0.3         # fraction of weights to prune (0.3 = 30%)

# Quantization settings
QUANTIZE = True             # set False to skip quantization

# Fine-tuning settings (to recover accuracy after pruning)
FINE_TUNE_EPOCHS = 3
FINE_TUNE_LR = 1e-4

# Benchmark settings
BENCHMARK_BATCHES = 20      # number of batches for speed test
# ================================================

device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
print(f"Device: {device}\n")


# ==================== HELPER FUNCTIONS ====================

def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = SimpleCNN(num_classes=checkpoint['num_classes'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"✅ Model loaded from: {checkpoint_path.name}")
    print(f"   Val Accuracy: {checkpoint['val_accuracy']:.2f}%")
    return model


def measure_accuracy(model, data_loader, device):
    """Measure accuracy on a dataset."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100.0 * correct / total
    return accuracy


def measure_inference_speed(model, data_loader, device, num_batches=20):
    """Measure average inference time per batch."""
    model.eval()
    times = []
    with torch.no_grad():
        for i, (images, _) in enumerate(data_loader):
            if i >= num_batches:
                break
            images = images.to(device)
            start = time.perf_counter()
            model(images)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
    avg_ms = (sum(times) / len(times)) * 1000
    return avg_ms


def count_zero_weights(model):
    """Count total and zero weights in the model."""
    total = 0
    zeros = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            total += module.weight.nelement()
            zeros += (module.weight == 0).sum().item()
    return total, zeros


def apply_pruning(model, amount=0.3):
    """Apply L1 unstructured pruning to Conv2d and Linear layers.

    L1 unstructured = remove the smallest absolute values individually
    (not structured = removing entire channels/filters)

    Args:
        model: the neural network
        amount: fraction of weights to prune (0.3 = 30%)
    """
    print(f"\n{'='*60}")
    print(f"✂️  PRUNING — removing {amount*100:.0f}% of smallest weights")
    print(f"{'='*60}")

    pruned_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name="weight", amount=amount)
            pruned_layers.append(name)

    total, zeros = count_zero_weights(model)
    sparsity = zeros / total * 100
    print(f"  Pruned {len(pruned_layers)} layers: {', '.join(pruned_layers)}")
    print(f"  Total weights: {total:,} | Zero: {zeros:,} ({sparsity:.1f}%)")
    return model


def make_pruning_permanent(model):
    """Remove pruning masks and make zeros permanent."""
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            try:
                prune.remove(module, 'weight')
            except ValueError:
                pass
    return model


def fine_tune(model, train_loader, loss_fn, device, num_epochs=3, lr=1e-4):
    """Fine-tune pruned model to recover accuracy."""
    print(f"\n{'='*60}")
    print(f"🔄 FINE-TUNING — recovering accuracy ({num_epochs} epochs)")
    print(f"{'='*60}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        avg_loss = running_loss / len(train_loader.dataset)
        print(f"  Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

    return model


def apply_quantization(model):
    """Apply dynamic quantization (FP32 → INT8).

    Dynamic quantization:
      - Converts Linear layer weights to INT8 at runtime
      - Activations are quantized dynamically during inference
      - Simplest form of quantization, no calibration data needed
      - Works best for models with large Linear layers

    Note: Quantization only works reliably on CPU.
          For GPU, you'd need ONNX Runtime quantization instead.
    """
    print(f"\n{'='*60}")
    print(f"📦 QUANTIZATION — FP32 → INT8 (dynamic)")
    print(f"{'='*60}")

    # Move to CPU for quantization
    model.cpu()

    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},          # quantize only Linear layers
        dtype=torch.qint8     # 8-bit integer
    )

    # Measure size reduction
    original_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    print(f"  Original size (FP32):  {original_size / 1024:.1f} KB")
    print(f"  Quantized layers:      {len([m for m in quantized_model.modules() if 'QuantizedLinear' in str(type(m))])}")
    print(f"  Note: Conv2d layers stay FP32 (dynamic quantization only affects Linear)")

    return quantized_model


def export_onnx(model, output_path, input_size=128):
    """Export model to ONNX format."""
    model.eval()
    model.cpu()

    dummy_input = torch.randn(1, 3, input_size, input_size)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=["image"],
            output_names=["logits"],
            opset_version=18,
        )
    size_kb = output_path.stat().st_size / 1024
    print(f"  ✅ {output_path.name} ({size_kb:.1f} KB)")


# ==================== MAIN PIPELINE ====================

if __name__ == "__main__":
    print("=" * 60)
    print("PRUNING + QUANTIZATION PIPELINE")
    print("=" * 60)

    # Load data for evaluation
    _, _, test_loader, _ = get_dataloaders(
        batch_size=32,
        val_fraction=0.15,
        test_fraction=0.2
    )
    loss_fn = nn.CrossEntropyLoss()

    # --- Step 1: Load model and measure baseline ---
    model = load_model(CHECKPOINT_PATH, device)

    print(f"\n{'='*60}")
    print(f"📊 BASELINE MEASUREMENT")
    print(f"{'='*60}")

    baseline_accuracy = measure_accuracy(model, test_loader, device)
    baseline_speed = measure_inference_speed(model, test_loader, device, BENCHMARK_BATCHES)
    baseline_total, baseline_zeros = count_zero_weights(model)

    print(f"  Test Accuracy:     {baseline_accuracy:.2f}%")
    print(f"  Inference speed:   {baseline_speed:.2f} ms/batch")
    print(f"  Total weights:     {baseline_total:,}")
    print(f"  Zero weights:      {baseline_zeros:,}")
    print(f"  Sparsity:          {baseline_zeros/baseline_total*100:.1f}%")

    baseline_onnx_path = OUTPUT_DIR / "baseline.onnx"
    export_onnx(model, baseline_onnx_path)

    # --- Step 2: Apply pruning ---
    model = apply_pruning(model, amount=PRUNE_AMOUNT)
    make_pruning_permanent(model)
    model.to(device)  # pruning may move weights to CPU, move back

    pruned_accuracy = measure_accuracy(model, test_loader, device)
    accuracy_loss = baseline_accuracy - pruned_accuracy

    print(f"\n  Pruned accuracy:   {pruned_accuracy:.2f}% (loss: {accuracy_loss:+.2f}%)")

    # --- Step 3: Fine-tune if accuracy dropped ---
    if accuracy_loss > 0.5:
        print(f"\n  Accuracy dropped {accuracy_loss:.2f}% — fine-tuning to recover...")
        model.to(device)
        train_loader, _, _, _ = get_dataloaders(
            batch_size=32,
            val_fraction=0.15,
            test_fraction=0.2
        )
        model = fine_tune(model, train_loader, loss_fn, device,
                          num_epochs=FINE_TUNE_EPOCHS, lr=FINE_TUNE_LR)

        recovered_accuracy = measure_accuracy(model, test_loader, device)
        print(f"  Recovered accuracy: {recovered_accuracy:.2f}%")
    else:
        print(f"  Accuracy loss < 0.5% — no fine-tuning needed")
        recovered_accuracy = pruned_accuracy

    pruned_speed = measure_inference_speed(model, test_loader, device, BENCHMARK_BATCHES)

    pruned_onnx_path = OUTPUT_DIR / "pruned.onnx"
    export_onnx(model, pruned_onnx_path)

    # --- Step 4: Quantization ---
    if QUANTIZE:
        try:
            quantized_model = apply_quantization(model)

            # Quantized model is on CPU — measure accuracy on CPU
            quantized_accuracy = measure_accuracy(quantized_model, test_loader, torch.device('cpu'))

            print(f"  Quantized accuracy: {quantized_accuracy:.2f}% (loss: {recovered_accuracy - quantized_accuracy:+.2f}%)")

            quantized_onnx_path = OUTPUT_DIR / "quantized.onnx"
            export_onnx(quantized_model, quantized_onnx_path)
        except Exception as e:
            print(f"  ⚠️ Quantization failed: {e}")
            print(f"  This is a known issue on MPS / newer PyTorch versions.")
            print(f"  Quantization works best on Linux + CUDA with onnxruntime.")
            quantized_accuracy = None
            quantized_onnx_path = None

    # --- Step 5: Final comparison ---
    print(f"\n{'='*60}")
    print(f"📊 FINAL COMPARISON")
    print(f"{'='*60}")

    baseline_size = baseline_onnx_path.stat().st_size / 1024
    pruned_size = pruned_onnx_path.stat().st_size / 1024

    print(f"""
  ┌──────────────────┬──────────┬──────────┬──────────┐
  │                  │ Baseline │ Pruned   │ Quantized│
  ├──────────────────┼──────────┼──────────┼──────────┤
  │ Accuracy         │ {baseline_accuracy:6.2f}%  │ {pruned_accuracy:6.2f}%  │ {quantized_accuracy if quantized_accuracy else '  N/A':>6}  │
  │ ONNX Size        │ {baseline_size:6.1f}KB │ {pruned_size:6.1f}KB │ {(quantized_onnx_path.stat().st_size/1024) if quantized_onnx_path else '  N/A':>6} │
  │ Speed (ms/batch) │ {baseline_speed:6.2f}   │ {pruned_speed:6.2f}   │ {'  N/A':>6} │
  └──────────────────┴──────────┴──────────┴──────────┘

  Baseline:   {baseline_onnx_path}
  Pruned:     {pruned_onnx_path}
  Quantized:  {quantized_onnx_path if quantized_onnx_path else 'N/A'}
""")
