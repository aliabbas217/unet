import torch
import torch.nn as nn
import torch.nn.functional as F
from ConvBlock import ConvBlock
from Encoder import Encoder
from Decoder import Decoder
from UNet import UNet

#Random Test Tensor
X = torch.rand(2, 3, 256, 256)


# #Testing Encoder outputs
# ecnoder = Encoder(conv_block_kernel_size=3)
# print(encoder)
# encoder_outputs = model(X)
# for encoder_output in encoder_outputs:
#   print(encoder_output.shape)


# #Testing Decoder outputs
# decoder = Decoder()
# print(decoder)


#Testing Complete UNet
unet = UNet(kernel_size=3)
# print(unet)
unet_output = unet(X)
print(unet_output.shape)

def get_params_count(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(get_params_count(unet))