import torch.nn.functional as F
from torch import nn
import torch


class DropPath(nn.Module):
    """
    DropPath (Stochastic Depth)

    During training, randomly drops entire residual branches.
    Instead of dropping individual neurons (like Dropout),
    this drops the whole residual transformation.

    Used in:
    - ResNet variants
    - EfficientNet
    - ConvNeXt
    - Vision Transformers

    Args:
        drop_prob (float): Probability of dropping residual path.
    """
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        # If no dropping or we're in evaluation mode → do nothing
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob

        # Create binary mask per sample (not per pixel!)
        # Shape: (batch_size, 1, 1, 1)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        random_tensor = keep_prob + torch.rand(shape, device=x.device)
        binary_mask = torch.floor(random_tensor)

        # Scale surviving paths to keep expectation consistent
        return x / keep_prob * binary_mask


class ResidualConvBlock(nn.Module):
    """
    Basic ResNet-style residual block.

    Structure:
        Conv3x3(stride) → BN → ReLU
        Conv3x3(1) → BN
        + Skip connection
        → ReLU

    Args:
        in_channels (int)
        out_channels (int)
        stride (int): controls spatial downsampling
        drop_path_prob (float): stochastic depth probability
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        drop_path_prob=0.0
    ):
        super().__init__()

        # ---- Main branch ----
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,  # IMPORTANT: always 1
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # ---- Skip branch ----
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()

        # ---- Stochastic depth ----
        self.drop_path = DropPath(drop_path_prob)

    def forward(self, x):
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.drop_path(out)

        out += identity
        out = F.relu(out, inplace=True)

        return out


class CustomFlowerCNN(nn.Module):
    """
    Custom CNN for Oxford Flowers (102 classes).

    Architecture:
        Stem (7x7 Conv, stride=2)
        4 Residual Blocks (channel increasing)
        Global Average Pool
        Dropout
        Linear classifier

    This is structurally similar to ResNet-18,
    but simplified and customizable.

    Args:
        num_classes (int): Number of output classes.
        channels (tuple): Channel sizes for each block.
        drop_path_rate (float): Maximum stochastic depth rate.
    """

    def __init__(
        self,
        num_classes=102,
        channels=(64, 128, 256, 512),
        drop_path_rate=0.1
    ):
        super().__init__()

        # ----------------------
        # STEM
        # ----------------------
        # Initial feature extraction.
        # Large kernel to capture low-level spatial structure.
        self.stem = nn.Sequential(
            nn.Conv2d(
                3,                  # RGB input
                channels[0],        # output channels
                kernel_size=7,
                stride=2,           # downsample 224 → 112
                padding=3,
                bias=False
            ),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        # ----------------------
        # Stochastic depth schedule
        # ----------------------
        # Gradually increase drop rate deeper in network
        dp_rates = torch.linspace(
            0,
            drop_path_rate,
            steps=len(channels)
        ).tolist()

        blocks = []
        in_ch = channels[0]

        # ----------------------
        # Residual Stages
        # ----------------------
        for i, out_ch in enumerate(channels):

            # First block keeps resolution
            # Later blocks downsample
            stride = 1 if i == 0 else 2

            blocks.append(
                ResidualConvBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    stride=stride,
                    drop_path_prob=dp_rates[i]
                )
            )

            in_ch = out_ch  # update input channels

        self.features = nn.Sequential(*blocks)

        # ----------------------
        # Classification Head
        # ----------------------
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global Average Pool
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(channels[-1], num_classes)
        )

    def forward(self, x):
        """
        Full forward pass.
        """

        # Initial convolution
        x = self.stem(x)

        # Residual blocks
        x = self.features(x)

        # Classification head
        x = self.head(x)

        return x

