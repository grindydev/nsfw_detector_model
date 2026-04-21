import torch
import torch.nn.functional as F
import torchvision.models as tv_models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms

from cnn import SimpleCNN

CLASS_NAMES = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']

device = torch.device('cuda' if torch.cuda.is_available()
                      else 'mps' if torch.backends.mps.is_available()
                      else 'cpu')


# ==================== GRAD-CAM IMPLEMENTATION ====================

class GradCAM:
    """Generates Grad-CAM heatmap showing where the model looks to make a prediction.

    How it works:
      1. Forward pass → capture activations from target conv layer
      2. Backward pass → capture gradients of the predicted class
      3. Weight each activation by how important it is (average gradient)
      4. Combine → heatmap of important regions
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # Hooks: functions that run automatically during forward/backward pass
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Runs during forward pass — saves the feature maps."""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Runs during backward pass — saves the gradients."""
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """Generate Grad-CAM heatmap for one image.

        Args:
            input_tensor: preprocessed image tensor, shape (1, 3, H, W)
            target_class: which class to explain. None = use predicted class

        Returns:
            heatmap: numpy array, shape (H, W), values 0-1
            predicted_class: int
            probabilities: tensor of class probabilities
        """
        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass — gradient of target class score only
        self.model.zero_grad()
        output[0, target_class].backward()

        # Global average pool the gradients → one weight per feature map
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of feature maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam)  # keep only positive contributions

        # Normalize to 0-1
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize heatmap to match input image size
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)

        probabilities = F.softmax(output, dim=1).cpu().detach()

        return cam.squeeze().cpu().numpy(), target_class, probabilities


# ==================== HELPER FUNCTIONS ====================

def load_simplecnn(checkpoint_path):
    """Load SimpleCNN model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = SimpleCNN(num_classes=checkpoint['num_classes'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"✅ SimpleCNN loaded | Val Accuracy: {checkpoint['val_accuracy']:.2f}%")
    return model


def load_resnet18(checkpoint_path):
    """Load ResNet18 transfer learning model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = tv_models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, checkpoint['num_classes'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"✅ ResNet18 loaded | Val Accuracy: {checkpoint['val_accuracy']:.2f}%")
    return model


def preprocess_image(image_path, input_size, mean, std):
    """Load and preprocess an image for the model."""
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)  # (1, 3, H, W)
    return image, tensor


def generate_single_cam(simplecnn_model, resnet18_model, image_path, device):
    """Generate Grad-CAM heatmaps for one image. Returns data for plotting."""
    NSFW_MEAN = [0.5973, 0.5313, 0.5066]
    NSFW_STD = [0.2896, 0.2808, 0.2854]
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    orig_image, simplecnn_input = preprocess_image(image_path, 128, NSFW_MEAN, NSFW_STD)
    _, resnet18_input = preprocess_image(image_path, 224, IMAGENET_MEAN, IMAGENET_STD)

    true_label = get_true_label(image_path)
    gray_image = orig_image.convert('L').convert('RGB')

    blur_level = 16  # ← adjust: 4=sharp, 8=light, 16=medium, 32=heavy
    if CLASS_NAMES[true_label] in ['hentai', 'sexy', 'porn']:
        w, h = orig_image.size
        small_w = max(1, w // blur_level)
        small_h = max(1, h // blur_level)
        gray_image = gray_image.resize((small_w, small_h), Image.NEAREST)
        gray_image = gray_image.resize((w, h), Image.NEAREST)

    simplecnn_cam = GradCAM(simplecnn_model, simplecnn_model.conv_block3.block[0])
    resnet18_cam = GradCAM(resnet18_model, resnet18_model.layer4[1].conv2)

    with torch.enable_grad():
        simple_heatmap, simple_pred, simple_probs = simplecnn_cam.generate(simplecnn_input)
        resnet_heatmap, resnet_pred, resnet_probs = resnet18_cam.generate(resnet18_input)

    return {
        'gray_image': gray_image,
        'true_label': true_label,
        'simple_heatmap': simple_heatmap,
        'simple_pred': simple_pred,
        'simple_conf': simple_probs[0, simple_pred].item() * 100,
        'resnet_heatmap': resnet_heatmap,
        'resnet_pred': resnet_pred,
        'resnet_conf': resnet_probs[0, resnet_pred].item() * 100,
    }


def generate_class_grid(simplecnn_model, resnet18_model, image_paths, class_name, save_path=None):
    """Generate a grid of Grad-CAM comparisons for one class.

    Layout: num_images rows × 3 columns
      Column 1: Original (grayscale)
      Column 2: SimpleCNN heatmap
      Column 3: ResNet18 heatmap
    """
    num_images = len(image_paths)
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))

    # Handle single image case (axes is 1D)
    if num_images == 1:
        axes = axes.reshape(1, -1)

    for row, img_path in enumerate(image_paths):
        print(f"  [{row+1}/{num_images}] {Path(img_path).name}")
        result = generate_single_cam(simplecnn_model, resnet18_model, img_path, device)

        true_label = result['true_label']

        # Column 1: Grayscale image
        axes[row, 0].imshow(result['gray_image'])
        if row == 0:
            axes[row, 0].set_title('Original (Grayscale)', fontsize=12)
        axes[row, 0].set_ylabel(f'Image {row+1}', fontsize=10)
        axes[row, 0].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

        # Column 2: SimpleCNN heatmap
        simple_resized = result['gray_image'].resize((128, 128))
        axes[row, 1].imshow(simple_resized, cmap='gray')
        axes[row, 1].imshow(result['simple_heatmap'], cmap='jet', alpha=0.5)
        pred_color = 'green' if result['simple_pred'] == true_label else 'red'
        axes[row, 1].set_title(
            f'SimpleCNN → {CLASS_NAMES[result["simple_pred"]]} ({result["simple_conf"]:.1f}%)' if row == 0 else '',
            fontsize=11, color=pred_color
        )
        axes[row, 1].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

        # Column 3: ResNet18 heatmap
        resnet_resized = result['gray_image'].resize((224, 224))
        axes[row, 2].imshow(resnet_resized, cmap='gray')
        axes[row, 2].imshow(result['resnet_heatmap'], cmap='jet', alpha=0.5)
        pred_color = 'green' if result['resnet_pred'] == true_label else 'red'
        axes[row, 2].set_title(
            f'ResNet18 → {CLASS_NAMES[result["resnet_pred"]]} ({result["resnet_conf"]:.1f}%)' if row == 0 else '',
            fontsize=11, color=pred_color
        )
        axes[row, 2].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

    fig.suptitle(f'Class: {class_name} — Grad-CAM Comparison', fontsize=15, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.show()
    plt.close()


def get_true_label(image_path):
    """Extract true label from image path (folder name = class name)."""
    class_name = Path(image_path).parent.name
    return CLASS_NAMES.index(class_name)


def find_sample_images(data_dir, num_per_class=1, seed=42):
    """Find random sample images, grouped by class."""
    import random
    random.seed(seed)
    samples = {}  # {class_name: [image_paths]}
    for class_name in CLASS_NAMES:
        class_dir = Path(data_dir) / class_name
        if not class_dir.exists():
            continue
        all_images = list(class_dir.glob('*'))
        selected = random.sample(all_images, min(num_per_class, len(all_images)))
        samples[class_name] = [str(p) for p in selected]
    return samples


# ==================== MAIN ====================

if __name__ == "__main__":
    DATA_DIR = Path.cwd() / 'data' / 'nsfw_dataset_v1'
    OUTPUT_DIR = Path.cwd() / 'grad_cam_results'
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load both models
    simplecnn = load_simplecnn('models/best_simple_cnn_train.pth')
    resnet18 = load_resnet18('models/best_transfer_cnn.pth')

    # Pick sample images grouped by class (5 per class)
    samples_by_class = find_sample_images(DATA_DIR, num_per_class=2)

    print(f"\n{'='*60}")
    print(f"Generating Grad-CAM grids for {len(samples_by_class)} classes")
    print(f"{'='*60}")

    for class_name, image_paths in samples_by_class.items():
        print(f"\n📷 Class: {class_name} ({len(image_paths)} images)")

        save_path = OUTPUT_DIR / f"grid_{class_name}.png"
        generate_class_grid(simplecnn, resnet18, image_paths, class_name, save_path=str(save_path))

    print(f"\n✅ All Grad-CAM results saved to {OUTPUT_DIR}/")
