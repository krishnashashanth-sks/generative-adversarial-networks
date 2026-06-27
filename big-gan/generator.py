import torch.nn as nn
from layers import GenBlock
import torch
import torch.nn.functional as F

class BigGANGenerator(nn.Module):
  def __init__(self, latent_dim, num_classes, ch=64, img_size=128, attention_res=[64, 32]):
    super(BigGANGenerator, self).__init__()
    self.latent_dim = latent_dim
    self.num_classes = num_classes
    self.ch = ch
    self.img_size = img_size
    self.attention_res = attention_res

    self.initial_dim = 4
    # Calculate num_cond_vectors correctly based on img_size and initial_dim
    temp_res = self.initial_dim
    num_blocks = 0
    while temp_res < img_size:
      num_blocks += 1
      temp_res *= 2
    self.num_cond_vectors = num_blocks + 1 # +1 for the initial linear layer

    if self.latent_dim % self.num_cond_vectors != 0:
        raise ValueError(f"latent_dim ({self.latent_dim}) must be divisible by num_cond_vectors ({self.num_cond_vectors})")

    self.latent_dim_per_split = self.latent_dim // self.num_cond_vectors

    self.label_embedding = nn.Embedding(num_classes, self.latent_dim_per_split) # Corrected embedding dimension
    self.linear = nn.Linear(self.latent_dim_per_split, self.initial_dim * self.initial_dim * ch * 16)

    self.blocks = nn.ModuleList()
    current_res = self.initial_dim
    current_ch = ch * 16

    for i in range(num_blocks):
      upsample = True
      # Attention is applied when the *output* resolution of the block matches attention_res
      use_attention = (current_res * 2 in self.attention_res)

      block_out_ch = current_ch // 2 # Halve channels for each block by default
      if i == num_blocks - 1: # For the very last block, output channels should be 'ch'
          block_out_ch = ch

      block = GenBlock(current_ch, block_out_ch, self.latent_dim_per_split, attention=use_attention, upsample=upsample)
      self.blocks.append(block)
      current_ch = block_out_ch
      current_res *= 2

    self.final_bn = nn.BatchNorm2d(current_ch)
    self.final_conv = nn.Conv2d(current_ch, 3, kernel_size=3, padding=1)
    self.tanh = nn.Tanh()

  def forward(self, z, y):
    class_embed = self.label_embedding(y) # Shape: (batch_size, latent_dim_per_split)

    # Split z into num_cond_vectors parts, each of size latent_dim_per_split
    z_splits = torch.split(z, self.latent_dim_per_split, dim=1) # Each split will be (batch_size, latent_dim_per_split)

    c_initial = z_splits[0] + class_embed # Now dimensions match
    out = self.linear(c_initial).view(c_initial.size(0), self.ch * 16, self.initial_dim, self.initial_dim)

    for i, block in enumerate(self.blocks):
      c_block = z_splits[i + 1] + class_embed # Dimensions will match here too
      out = block(out, c_block)

    out = F.relu(self.final_bn(out))
    out = self.final_conv(out)
    return self.tanh(out)
