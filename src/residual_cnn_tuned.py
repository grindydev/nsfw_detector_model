import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """ResNet-style residual block with skip connection.

    Two conv layers + shortcut for channel mismatch.
    MaxPool goes AFTER the skip connection so dimensions match.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1×1 conv to match channels if they change
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        residual = self.shortcut(x)              # match channels if needed
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual                           # skip connection — same spatial size
        out = F.relu(out)
        out = self.pool(out)                      # downsample after skip
        return out


class ResidualTunedCNN(nn.Module):
    def __init__(self, num_classes=5, dropout=0.358):
        super(ResidualTunedCNN, self).__init__()
        self.num_classes = num_classes

        # 5 residual blocks: 32 → 64 → 128 → 128 → 256
        self.conv_block1 = ResidualBlock(3, 32)
        self.conv_block2 = ResidualBlock(32, 64)
        self.conv_block3 = ResidualBlock(64, 128)
        self.conv_block4 = ResidualBlock(128, 128)
        self.conv_block5 = ResidualBlock(128, 256)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
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
