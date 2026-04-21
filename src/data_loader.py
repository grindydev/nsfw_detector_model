# data_loader.py

import os
from typing import List, Tuple

from torch.utils.data import Dataset, random_split, Subset, DataLoader
from pathlib import Path
from torchvision import transforms
import torch
from tqdm.auto import tqdm
from PIL import Image

path_dataset = Path.cwd() / 'data/nsfw_dataset_v1'
# helper_utils.print_data_folder_structure(path_dataset, max_depth=1)

class SubsetWithTransform(Dataset):
    def __init__(self, subset: Subset, transform = None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, index):
        image, label = self.subset[index]

        if self.transform:
            image = self.transform(image)

        return image, label

class NSFWDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.classes = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples: List[Tuple[str, int]] = self._make_dataset()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]

        with Image.open(img_path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        
        # label = self.get_class_name(label)

        return image, label


    def _make_dataset(self) -> List[Tuple[str, int]]:
        """Scan directories and collect (image_path, label) pairs."""
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            for entry in os.scandir(class_dir):
                if entry.is_file() and entry.name.lower().endswith(
                    ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif')
                ):
                    samples.append((entry.path, self.class_to_idx[class_name]))
        return samples
    
    def get_class_name(self, index):
        return self.classes[index]


def get_mean_std(dataset: Dataset):
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    total_pixels = 0
    sum_pixels = torch.zeros(3)

    mean_loader = tqdm(dataset, desc="Pass 1/2: Computing Mean")

    for img, _ in mean_loader:
        img_tensor = preprocess(img)
        # Reshape to [3, num_pixels] so we can sum per channel
        pixels = img_tensor.view(3, -1)
        sum_pixels += pixels.sum(dim=1)       # Sum of all pixel values per channel
        total_pixels += pixels.size(1)         # Count total pixels

    mean = sum_pixels / total_pixels

    # --- Pass 2: Compute Standard Deviation ---
    sum_squared_diff = torch.zeros(3)

    std_loader = tqdm(dataset, desc="Pass 2/2: Computing Std")

    for img, _ in std_loader:
        img_tensor = preprocess(img)
        pixels = img_tensor.view(3, -1)
        diff = pixels - mean.unsqueeze(1)       # Subtract mean from each pixel
        sum_squared_diff += (diff ** 2).sum(dim=1)  # Sum of squared differences

    std = torch.sqrt(sum_squared_diff / total_pixels)

    return mean, std


nswf_dataset = NSFWDataset(root_dir=path_dataset, transform=None)

MEAN = [0.5973, 0.5313, 0.5066]
MEAN_STD = [0.2896, 0.2808, 0.2854]

# mean, std = get_mean_std(nswf_dataset)
# print(f"\nMean: {mean}")
# print(f" Std: {std}")


def get_transformations(mean, std, size=(128, 128)):
    main_tfs = [
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]

    augmentation_tfs = [
        transforms.Resize(size),                                                                                                                                                                
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),                                                                                                                                                        
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),                                                                                                                         
        transforms.ToTensor(),                                                                                                                                                                        
        transforms.Normalize(mean, std),    
    ]

    main_transform = transforms.Compose(main_tfs)
    augmentation_transform = transforms.Compose(augmentation_tfs)

    return main_transform, augmentation_transform

main_transform, transform_with_augmentation = get_transformations(MEAN, MEAN_STD)


def get_dataloaders(batch_size, val_fraction, test_fraction, dataset=nswf_dataset,
                    main_transform=main_transform,
                    augmentation_transform=transform_with_augmentation,
                    train_fraction=1.0):
    
    dataset=nswf_dataset
    total_size = len(dataset)
    val_size = int(total_size * val_fraction)
    test_size = int(total_size * test_fraction)
    train_size = total_size - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Use only a fraction of training data (for fast Optuna trials)
    if train_fraction < 1.0:
        subset_size = int(len(train_dataset) * train_fraction)
        train_dataset = Subset(train_dataset, range(subset_size))

    train_dataset = SubsetWithTransform(subset=train_dataset, transform=augmentation_transform)
    val_dataset = SubsetWithTransform(subset=val_dataset, transform=main_transform)
    test_dataset = SubsetWithTransform(subset=test_dataset, transform=main_transform)


    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    num_classes = len(dataset.classes)

    return train_loader, val_loader, test_loader, num_classes

# Example
# train_loader, val_loader, test_loader = get_dataloaders(
#     dataset=nswf_dataset,
#     batch_size=32,
#     val_fraction=0.15,
#     test_fraction=0.2,
#     main_transform=main_transform,
#     augmentation_transform=transform_with_augmentation
#     )




