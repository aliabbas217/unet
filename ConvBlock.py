import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
  def __init__(self, in_channels=3, filters = 32, layers_per_block=3, kernel_size = 2):
    """
    Sequential Convolutional Block with ReLU activation function.
    Args:
      in_channels (int): Number of input channels.
      filters (int): Number of filters in the convolutional layer.
      layers_per_block (int): Number of convolutional layers in the block.
      kernel_size (int): Size of the convolutional kernel.
    """
    super(ConvBlock, self).__init__()
    self.kernel_size = kernel_size
    self.in_channels = in_channels
    self.filters = filters
    self.layers_per_block = layers_per_block
    self.first_conv_layer = nn.Conv2d(
      in_channels = self.in_channels, out_channels = self.filters,
      padding = "same", kernel_size=kernel_size
    )
    self.seq = nn.Sequential(*(
    nn.Sequential(
        nn.Conv2d(
          in_channels=self.filters, out_channels=self.filters,
          padding="same", kernel_size=kernel_size
        ),
        nn.ReLU(inplace=True)
    )
    for _ in range(self.layers_per_block - 1)
    ))

  def forward(self, x):
    """
    Forward pass through the convolutional block.
    Args:
      x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
    Returns:
      None
    """
    x = F.relu(self.first_conv_layer(x))
    x = self.seq(x)

    return x
