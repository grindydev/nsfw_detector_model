import torch
from torch import optim
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms

from data_loader import get_dataloaders, get_transformations
import helper_utils


device = torch.device('cuda' if torch.cuda.is_available()
                      else 'mps' if torch.backends.mps.is_available()
                      else 'cpu')

# ==================== DATA PREPARATION ====================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

main_transform, transform_with_augmentation = get_transformations(IMAGENET_MEAN, IMAGENET_STD, size=(224, 224))


train_loader, val_loader, test_loader, num_classes = get_dataloaders(
    batch_size=64,
    val_fraction=0.15,
    test_fraction=0.2,
    main_transform=main_transform,
    augmentation_transform=transform_with_augmentation
    )

# ==================== EXPLORE PRE-TRAINED MODELS ====================
resnet18_model = tv_models.resnet18(tv_models.ResNet18_Weights.DEFAULT)

print('Restnet18: ', resnet18_model)

fc_layer = resnet18_model.fc
num_features = fc_layer.in_features

new_fc_layer = nn.Linear(in_features=num_features, out_features=num_classes)
resnet18_model.fc = new_fc_layer
resnet18_model.to(device)


print("Model's New Output Layer:")
print(resnet18_model.fc)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, resnet18_model.parameters()), lr=1e-5, weight_decay=0.05)

# ==================== TRAINING ====================
print(f"\n{'='*60}")
print(f"Transfer Learning — Strategy 3: Full Retraining")
print(f"  train={len(train_loader.dataset)} images | val={len(val_loader.dataset)} images")
print(f"  batch_size=64 | lr=0.00001 | optimizer=AdamW")
print(f"  input=224×224 | normalization=ImageNet")
print(f"{'='*60}")

best_accuracy = 0.0
num_epochs = 5

for epoch in range(num_epochs):
    helper_utils.train_model(
        model=resnet18_model,
        train_dataloader=train_loader,
        n_epochs=1,
        loss_fcn=loss_function,
        optimizer=optimizer,
        device=device
    )

    val_accuracy = helper_utils.evaluate_accuracy(resnet18_model, val_loader, device)
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save({
           'model_state_dict': resnet18_model.state_dict(),
           'num_classes': num_classes,
           'val_accuracy': best_accuracy * 100,
           'epoch': epoch + 1,
        }, 'models/best_transfer_cnn_fulltrain.pth')
        print(f"  → New best model saved ({best_accuracy*100:.2f}%)")

print(f"\n✅ Training finished! Best accuracy: {best_accuracy*100:.2f}%")
print(f"   Model saved to: models/best_transfer_cnn_fulltrain.pth")

