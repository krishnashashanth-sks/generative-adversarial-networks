import torch.nn as nn
from layers import *

class Generator(nn.Module):
  def __init__(self,latent_dim,f_map_base=512,f_map_max=512,num_stages=8,in_channels=3):
    super().__init__()
    self.latent_dim=latent_dim
    self.num_stages=num_stages
    self.in_channels=in_channels
    self.f_map_base=f_map_base
    self.f_map_max=f_map_max
    self.current_stage=0
    self.alpha=0.0
    def get_channels(stage):
      res_log2=stage+2
      return max(32,min(self.f_map_max,self.f_map_base//(2**(res_log2-2))))
    self.pixel_norm=PixelNorm()
    self.initial_block=nn.Sequential(
        nn.ConvTranspose2d(latent_dim,get_channels(0),kernel_size=4,stride=1,padding=0),
        PixelNorm(),
        nn.LeakyReLU(0.2),
        ConvBlock(get_channels(0),get_channels(0),kernel_size=3,stride=1,padding=1),
        PixelNorm()
    )
    self.upsample_blocks=nn.ModuleList()
    self.to_rgb_layers=nn.ModuleList()
    self.to_rgb_layers.append(nn.Conv2d(get_channels(0),in_channels,kernel_size=1,stride=1,padding=0))
    for stage in range(1,num_stages):
      in_ch=get_channels(stage-1)
      out_ch=get_channels(stage)
      self.upsample_blocks.append(nn.Sequential(
          UpsampleBlock(in_ch,out_ch),
          PixelNorm(),
          ConvBlock(out_ch,out_ch,kernel_size=3,stride=1,padding=1),
          PixelNorm()
      ))
      self.to_rgb_layers.append(nn.Conv2d(out_ch,in_channels,kernel_size=1,stride=1,padding=0))
  def forward(self,z):
    x = z.view(z.size(0), self.latent_dim, 1, 1)

    # Start with the initial block (stage 0, 4x4 resolution features)
    x = self.initial_block(x)

    # If we are only generating at stage 0, return directly
    if self.current_stage == 0:
        return torch.tanh(self.to_rgb_layers[0](x))

    # Features and RGB output from the previous (lower) resolution stage, for blending
    out_prev_rgb = None

    # Iterate through stages, applying upsampling blocks
    # We iterate from stage 1 up to the current_stage (inclusive)
    for i in range(1, self.current_stage + 1):
        # Check if we are at the target stage for blending
        if i == self.current_stage and self.alpha > 0:
            # We need the RGB output of the *previous* resolution (i-1)
            # which is currently stored in `x` before applying the upsample block `upsample_blocks[i-1]`
            out_prev_rgb = self.to_rgb_layers[i-1](x)
            out_prev_rgb = nn.functional.interpolate(out_prev_rgb, scale_factor=2, mode='nearest')

        # Apply the corresponding upsample block to get features for stage 'i'
        # upsample_blocks[0] transforms from stage 0 to stage 1.
        # upsample_blocks[i-1] transforms from stage (i-1) to stage i.
        x = self.upsample_blocks[i-1](x)

    # After the loop, x contains the features for `self.current_stage`
    # Convert these features to an RGB image
    img = self.to_rgb_layers[self.current_stage](x)

    # Apply blending if fade-in is active
    if out_prev_rgb is not None and self.alpha > 0:
        img = self.alpha * img + (1 - self.alpha) * out_prev_rgb

    return torch.tanh(img)
  def update_stage(self,new_stage,alpha):
    self.current_stage=new_stage
    self.alpha=alpha
    if self.current_stage>=self.num_stages:
      self.current_stage=self.num_stages-1
    if self.alpha>1.0:self.alpha=1.0
    if self.alpha<0.0:self.alpha=0.0