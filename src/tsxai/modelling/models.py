"""Time series classification models.

This module implements ResNet and InceptionTime architectures for time series
classification tasks. Both implementations are adapted from the tsai library
(https://github.com/timeseriesAI/tsai) and optimized for time series data.

The implementations maintain the core architecture while providing a standalone
version without external dependencies beyond PyTorch.

References:
    - ResNet implementation adapted from tsai.models.ResNet
    - InceptionTime implementation adapted from tsai.models.InceptionTime
"""

from typing import List, Optional, Union

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """1D Convolutional block with batch normalization and activation.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolving kernel
        stride (int, optional): Stride of convolution. Defaults to 1.
        padding (Union[str, int], optional): Padding added to both sides of input.
            Defaults to 'same'.
        act (Optional[nn.Module], optional): Activation function. Defaults to ReLU.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[str, int] = "same",
        act: Optional[nn.Module] = nn.ReLU(),
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class ResBlock(nn.Module):
    """Residual block for time series data.

    Contains three convolutional blocks with decreasing kernel sizes and a shortcut
    connection. If input and output channels differ, the shortcut includes a 1x1
    convolution to match dimensions.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_sizes (List[int], optional): Kernel sizes for the three conv blocks.
            Defaults to [7, 5, 3].
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_sizes: List[int] = [7, 5, 3]
    ):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_sizes[0])
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_sizes[1])
        self.conv3 = ConvBlock(out_channels, out_channels, kernel_sizes[2], act=None)

        # Shortcut connection with optional channel matching
        self.shortcut = (
            nn.BatchNorm1d(in_channels)
            if in_channels == out_channels
            else ConvBlock(in_channels, out_channels, 1, act=None)
        )

        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x + self.shortcut(identity)  # Residual connection
        x = self.act(x)

        return x


class ResNet(nn.Module):
    """ResNet architecture for time series classification.

    A simple ResNet architecture consisting of three residual blocks followed by
    global average pooling and a fully connected layer.

    Implementation adapted from the tsai library's ResNet model.

    Args:
        in_channels (int): Number of input channels (features)
        num_classes (int): Number of output classes
        base_filters (int, optional): Number of base filters. Doubles after first
            residual block. Defaults to 64.
        kernel_sizes (List[int], optional): Kernel sizes for conv blocks.
            Defaults to [7, 5, 3].

    Example:
        >>> model = ResNet(in_channels=1, num_classes=2)
        >>> x = torch.randn(32, 1, 100)  # (batch_size, channels, time_steps)
        >>> output = model(x)
        >>> output.shape
        torch.Size([32, 2])
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_filters: int = 64,
        kernel_sizes: List[int] = [7, 5, 3],
    ):
        super().__init__()

        nf = base_filters
        self.resblock1 = ResBlock(in_channels, nf, kernel_sizes)
        self.resblock2 = ResBlock(nf, nf * 2, kernel_sizes)
        self.resblock3 = ResBlock(nf * 2, nf * 2, kernel_sizes)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(nf * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)

        x = self.pool(x)
        x = x.squeeze(-1)  # Remove the last dimension
        x = self.fc(x)

        return x


class InceptionModule(nn.Module):
    """Inception module for time series data.

    Implements parallel convolutions with different kernel sizes and max pooling,
    following the InceptionTime architecture.

    Implementation adapted from the tsai library's InceptionTime model.

    Args:
        in_channels (int): Number of input channels
        n_filters (int): Number of filters for each convolution path
        kernel_size (int, optional): Base kernel size. Will be divided by powers of 2.
            Defaults to 40.
        use_bottleneck (bool, optional): Whether to use bottleneck layer.
            Defaults to True.
    """

    def __init__(
        self,
        in_channels: int,
        n_filters: int,
        kernel_size: int = 40,
        use_bottleneck: bool = True,
    ):
        super().__init__()

        # Calculate kernel sizes for parallel convolutions
        kernel_sizes = [kernel_size // (2**i) for i in range(3)]
        # Ensure odd kernel sizes for proper padding
        kernel_sizes = [k if k % 2 != 0 else k - 1 for k in kernel_sizes]

        # Bottleneck layer if requested and input channels > 1
        self.use_bottleneck = use_bottleneck and in_channels > 1
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, n_filters, 1, bias=False)

        # Parallel convolution paths
        bottleneck_channels = n_filters if self.use_bottleneck else in_channels
        self.conv_paths = nn.ModuleList(
            [
                nn.Conv1d(
                    bottleneck_channels,
                    n_filters,
                    kernel_size=k,
                    padding="same",
                    bias=False,
                )
                for k in kernel_sizes
            ]
        )

        # MaxPool path
        self.maxpool_path = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_channels, n_filters, 1, bias=False),
        )

        # Final BatchNorm and activation
        self.bn = nn.BatchNorm1d(n_filters * 4)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply bottleneck if present
        if self.use_bottleneck:
            x_conv = self.bottleneck(x)
        else:
            x_conv = x

        # Process through parallel paths
        conv_outputs = [conv(x_conv) for conv in self.conv_paths]
        maxpool_output = self.maxpool_path(x)

        # Concatenate all paths
        x = torch.cat([*conv_outputs, maxpool_output], dim=1)

        # Apply BatchNorm and activation
        x = self.bn(x)
        x = self.relu(x)

        return x


class InceptionBlock(nn.Module):
    """Block of stacked Inception modules with residual connections.

    Args:
        in_channels (int): Number of input channels
        n_filters (int, optional): Number of filters per path. Defaults to 32.
        use_residual (bool, optional): Whether to use residual connections.
            Defaults to True.
        depth (int, optional): Number of Inception modules. Defaults to 6.
        kernel_size (int, optional): Base kernel size. Defaults to 40.
        use_bottleneck (bool, optional): Whether to use bottleneck. Defaults to True.
    """

    def __init__(
        self,
        in_channels: int,
        n_filters: int = 32,
        use_residual: bool = True,
        depth: int = 6,
        kernel_size: int = 40,
        use_bottleneck: bool = True,
    ):
        super().__init__()

        self.use_residual = use_residual
        self.depth = depth

        # Stack of Inception modules
        self.inception_modules = nn.ModuleList()
        for d in range(depth):
            self.inception_modules.append(
                InceptionModule(
                    in_channels if d == 0 else n_filters * 4,
                    n_filters,
                    kernel_size=kernel_size,
                    use_bottleneck=use_bottleneck,
                )
            )

        # Residual connections
        if self.use_residual:
            self.shortcut_layers = nn.ModuleList()
            for d in range(depth // 3):
                n_in = in_channels if d == 0 else n_filters * 4
                n_out = n_filters * 4
                self.shortcut_layers.append(
                    nn.BatchNorm1d(n_in)
                    if n_in == n_out
                    else nn.Sequential(
                        nn.Conv1d(n_in, n_out, 1, bias=False), nn.BatchNorm1d(n_out)
                    )
                )

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for d in range(self.depth):
            x = self.inception_modules[d](x)
            if self.use_residual and d % 3 == 2:
                residual = x = self.relu(x + self.shortcut_layers[d // 3](residual))
        return x


class InceptionTime(nn.Module):
    """InceptionTime model for time series classification.

    Implementation based on the paper "InceptionTime: Finding AlexNet for Time Series
    Classification" by Fawaz et al.

    Args:
        in_channels (int): Number of input channels (features)
        num_classes (int): Number of output classes
        n_filters (int, optional): Number of filters per Inception module path.
            Defaults to 32.
        depth (int, optional): Depth of the Inception block. Defaults to 6.
        kernel_size (int, optional): Base kernel size. Defaults to 40.
        use_residual (bool, optional): Whether to use residual connections.
            Defaults to True.
        use_bottleneck (bool, optional): Whether to use bottleneck. Defaults to True.

    Example:
        >>> model = InceptionTime(in_channels=1, num_classes=2)
        >>> x = torch.randn(32, 1, 100)  # (batch_size, channels, time_steps)
        >>> output = model(x)
        >>> output.shape
        torch.Size([32, 2])
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        n_filters: int = 32,
        depth: int = 6,
        kernel_size: int = 40,
        use_residual: bool = True,
        use_bottleneck: bool = True,
    ):
        super().__init__()

        self.inception_block = InceptionBlock(
            in_channels=in_channels,
            n_filters=n_filters,
            depth=depth,
            kernel_size=kernel_size,
            use_residual=use_residual,
            use_bottleneck=use_bottleneck,
        )

        # Global average pooling and final classification
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(n_filters * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inception_block(x)
        x = self.gap(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x
