import torch
import torch.nn as nn
import torch.nn.functional as F
from UNetBase import UNetBase


class UNet(nn.Module):
  def __init__(self, in_channels=3, init_filters = 32, layers_per_block = 4, num_classes = 3, levels = 4, kernel_size= 2, 
               base_model = None, freeze_base_model = False
               ):
    super(UNet, self).__init__()
    """
    UNet model for image segmentation.
    Args:
        in_channels (int): Number of input channels.
        init_filters (int): Number of filters in the first layer.
        layers_per_block (int): Number of layers per block.
        num_classes (int): Number of output classes.
        kernel_size (int): Size of the convolutional kernel.
        levels (int): Number of levels in the U-Net architecture.
        base_model (torch.nn.Module, optional): Base model to use for feature extraction. Defaults to None.
        freeze_base_model (bool): Whether to freeze the base model's parameters. Defaults to False.
    """
    self.in_channels = in_channels
    self.init_filters = init_filters
    self.layers_per_block = layers_per_block
    self.num_classes = num_classes
    self.kernel_size = kernel_size
    self.levels = levels
    self.base_model = base_model
    self.freeze_base_model = freeze_base_model
    if self.base_model is None:
      self.base_model = UNetBase(
          in_channels=self.in_channels, init_filters=self.init_filters,
          layers_per_block=self.layers_per_block, levels=self.levels, kernel_size=self.kernel_size
      )
    if self.freeze_base_model:
      for param in self.base_model.parameters():
        param.requires_grad = False

    self.final_conv = nn.Conv2d(in_channels=init_filters, out_channels=num_classes, kernel_size=1)


  def forward(self, x):
    """
    Forward pass through the U-Net model.
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
    Returns:
        torch.Tensor: Output tensor after passing through the U-Net model.
    """
    base_output = self.base_model(x)
    output = self.final_conv(base_output)
    return output