import torch.nn.functional as F
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, filters, layers_per_block=2, kernel_size=3):
        """
        Sequential Convolutional Block with BatchNorm and ReLU activation.
        Args:
          in_channels (int): Number of input channels.
          filters (int): Number of output channels (filters).
          layers_per_block (int): Number of convolutional layers in the block.
          kernel_size (int): Size of the convolutional kernel.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        # padding = kernel_size - stride // 2
        # since stride is 1, padding = kernel_size // 2
        # For example, for kernel_size=3, padding=1; for kernel_size=5, padding=2
        padding = kernel_size // 2

        self.layers.append(nn.Conv2d(in_channels, filters, kernel_size=kernel_size, padding=padding))
        self.layers.append(nn.BatchNorm2d(filters))
        self.layers.append(nn.ReLU(inplace=False))

        for _ in range(layers_per_block - 1):
            self.layers.append(nn.Conv2d(filters, filters, kernel_size=kernel_size, padding=padding))
            self.layers.append(nn.BatchNorm2d(filters))
            self.layers.append(nn.ReLU(inplace=False))

        self.block = nn.Sequential(*self.layers)

    def forward(self, x):
        """
        Forward pass through the convolutional block.
        Args:
          x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
          torch.Tensor: Output tensor of shape (batch_size, filters, height, width).
        """
        return self.block(x)