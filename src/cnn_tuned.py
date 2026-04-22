# cnn_tuned.py — TunedCNN architecture
# Optimized by Optuna hyperparameter search

import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):

        super(CNNBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            )
        )

    def forward(self, x):
        x = self.block(x)
        return x


class TunedCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(TunedCNN, self).__init__()
        self.classess = num_classes

        # Architecture found by Optuna: 5 layers, filters [32, 64, 128, 128, 256]
        self.conv_block1 = CNNBlock(in_channels=3, out_channels=32)
        self.conv_block2 = CNNBlock(in_channels=32, out_channels=64)
        self.conv_block3 = CNNBlock(in_channels=64, out_channels=128)
        self.conv_block4 = CNNBlock(in_channels=128, out_channels=128)
        self.conv_block5 = CNNBlock(in_channels=128, out_channels=256)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(p=0.358),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.classifier(x)
        return x


