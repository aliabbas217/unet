import torch
import torch.nn as nn
import torch.nn.functional as F
from Encoder import Encoder
from Decoder import Decoder


class UNetBase(nn.Module):
  def __init__(self, in_channels=3, init_filters = 32, layers_per_block = 4, levels = 4, kernel_size= 2):
    """
    UNet Base model for image segmentation.
    Args:
        in_channels (int): Number of input channels.
        init_filters (int): Number of filters in the first layer.
        layers_per_block (int): Number of layers per block.
        levels (int): Number of levels in the U-Net architecture.
        kernel_size (int): Size of the convolutional kernel.
    """
    super(UNetBase, self).__init__()
    self.in_channels = in_channels
    self.init_filters = init_filters
    self.layers_per_block = layers_per_block
    self.kernel_size = kernel_size
    self.levels = levels
    self.encoder = Encoder(
        in_channels=self.in_channels, init_filters=self.init_filters, layers_per_block=self.layers_per_block,
        levels=self.levels, conv_block_kernel_size=self.kernel_size
    )
    self.decoder = Decoder(
        levels_in_encoder=self.levels, init_filters=self.init_filters, layers_per_block=self.layers_per_block,
        conv_block_kernel_size=self.kernel_size
    )

  def forward(self, x):
    """
    Forward pass through the U-Net model.
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
    Returns:
        torch.Tensor: Output tensor after passing through the U-Net model.
    """
    encoder_outputs = self.encoder(x)
    decoder_outputs = self.decoder(encoder_outputs)

    return decoder_outputs