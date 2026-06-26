import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class ConvBlock(nn.Module):
  def __init__(self,in_channels,out_channels,kernel_size,stride,padding,use_sn=False):
    super().__init__()
    if use_sn:
      self.conv=spectral_norm(nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False))
    else:
      self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False)
    self.relu=nn.LeakyReLU(0.3)
  def forward(self,x):
    return self.relu(self.conv(x))

class UpsampleBlock(nn.Module):
  def __init__(self,in_channels,out_channels,use_sn=False):
    super().__init__()
    self.upsample=nn.Upsample(scale_factor=2,mode='nearest')
    self.conv_block=ConvBlock(in_channels,out_channels,kernel_size=3,stride=1,padding=1,use_sn=use_sn)
  def forward(self,x):
    return self.conv_block(self.upsample(x))

class DownsampleBlock(nn.Module):
  def __init__(self,in_channels,out_channels,use_sn=False):
    super().__init__()
    self.conv_block=ConvBlock(in_channels,out_channels,kernel_size=3,stride=1,padding=1,use_sn=use_sn)
    self.downsample=nn.AvgPool2d(kernel_size=2,stride=2)
  def forward(self,x):
    return self.downsample(self.conv_block(x))

class MiniBatchStdDev(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self,x):
    batch_size,channels,height,width=x.shape
    stddev=torch.sqrt(x.var(dim=0,keepdim=True)+1e-8)
    averaged_stddev=stddev.mean().view(1,1,1,1)
    replicated_stddev=averaged_stddev.repeat(batch_size,1,height,width)
    return torch.cat([x,replicated_stddev],dim=1)

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        # Calculate the square root of the mean of squares along the channel dimension
        # (batch_size, channels, height, width) -> (batch_size, 1, height, width)
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)
