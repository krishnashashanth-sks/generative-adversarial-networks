import torch.nn as nn
from layers import *

class Discriminator(nn.Module):
  def __init__(self,f_map_base=512,f_map_max=512,num_stages=8,in_channels=3):
    super().__init__()
    self.num_stages=num_stages
    self.in_channels=in_channels
    self.f_map_base=f_map_base
    self.f_map_max=f_map_max
    self.current_stage=0
    self.alpha=0.0
    def get_channels(stage):
      res_log2=stage+2
      return max(32,min(self.f_map_max,self.f_map_base//(2**(res_log2-2))))
    self.from_rgb_layers=nn.ModuleList()
    for stage in range(num_stages):
      out_ch=get_channels(stage)
      self.from_rgb_layers.append(nn.Conv2d(in_channels,out_ch,kernel_size=1,stride=1,padding=0))
    self.downsample_blocks=nn.ModuleList()
    for stage in range(1,num_stages):
      in_ch=get_channels(stage)
      out_ch=get_channels(stage-1)
      self.downsample_blocks.append(
          nn.Sequential(
              ConvBlock(in_ch,in_ch,kernel_size=3,stride=1,padding=1,use_sn=True),
              DownsampleBlock(in_ch,out_ch,use_sn=True)
          )
      )
    self.final_block=nn.Sequential(
        MiniBatchStdDev(),
        ConvBlock(get_channels(0)+1,get_channels(0),kernel_size=3,stride=1,padding=1,use_sn=True),
        nn.Conv2d(get_channels(0),1,kernel_size=4,stride=1,padding=0)
    )
  def forward(self,x):
    out_prev=None
    if self.current_stage>0 and self.alpha>0:
      out_prev=self.from_rgb_layers[self.current_stage-1](x)
      out_prev=nn.functional.avg_pool2d(out_prev,kernel_size=2,stride=2)
    x=self.from_rgb_layers[self.current_stage](x)
    for i in range(self.current_stage,-1,-1):
      if i==0:
        x=self.final_block(x)
      else:
        x=self.downsample_blocks[i-1](x)
    if out_prev is not None and self.alpha>0:
      x=self.alpha*x+(1-self.alpha)*out_prev
    return x
  def update_stage(self,new_stage,alpha):
    self.current_stage=new_stage
    self.alpha=alpha
    if self.current_stage>=self.num_stages:
      self.current_stage=self.num_stages-1
    if self.alpha>1.0:self.alpha=1.0
    if self.alpha<0.0:self.alpha=0.0