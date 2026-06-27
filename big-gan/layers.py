from torch.nn.utils import spectral_norm
import torch.nn as nn
import torch
import torch.nn.functional as F

# --- Corrected SelfAttention (adjusted reduced_channels calculation) ---
class SelfAttention(nn.Module):
  def __init__(self, in_channels):
    super(SelfAttention, self).__init__()
    self.in_channels = in_channels
    reduced_channels = max(1, in_channels // 8) # Ensure reduced_channels is at least 1
    self.query_conv = nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False)
    self.key_conv = nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False)
    self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False) # Value conv keeps original in_channels
    self.gamma = nn.Parameter(torch.zeros(1))

  def forward(self, x):
    batch_size, channels, height, width = x.size()
    proj_query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1) # B, HW, C'
    proj_key = self.key_conv(x).view(batch_size, -1, height * width) # B, C', HW
    energy = torch.bmm(proj_query, proj_key) # B, HW, HW
    attention = F.softmax(energy, dim=-1) # B, HW, HW

    proj_value = self.value_conv(x).view(batch_size, -1, height * width) # B, C, HW
    out = torch.bmm(proj_value, attention.permute(0, 2, 1)) # B, C, HW
    out = out.view(batch_size, channels, height, width)

    out = self.gamma * out + x
    return out

# --- ConditionalBatchNorm2d (no change needed from previous correct version) ---
class ConditionalBatchNorm2d(nn.Module):
  def __init__(self, num_features, embedding_dim):
    super(ConditionalBatchNorm2d, self).__init__()
    self.num_features = num_features
    self.embedding_dim = embedding_dim
    self.bn = nn.BatchNorm2d(num_features, affine=False)
    self.gamma_embed = nn.Linear(embedding_dim, num_features, bias=False)
    self.beta_embed = nn.Linear(embedding_dim, num_features, bias=False)
  def forward(self, x, c):
    out = self.bn(x)
    gamma = self.gamma_embed(c).view(c.size(0), self.num_features, 1, 1)
    beta = self.beta_embed(c).view(c.size(0), self.num_features, 1, 1)
    return out * (1 + gamma) + beta

# --- Corrected GenBlock (fixed residual connection logic and input `latent_dim_per_split`) ---
class GenBlock(nn.Module):
  def __init__(self, in_channels, out_channels, latent_dim_per_split, attention=False, upsample=True):
    super(GenBlock, self).__init__()
    self.upsample = upsample
    self.attention = attention

    # Main path
    self.cbn1 = ConditionalBatchNorm2d(in_channels, latent_dim_per_split)
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
    self.cbn2 = ConditionalBatchNorm2d(out_channels, latent_dim_per_split)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)

    # Shortcut path
    self.shortcut_conv = None
    if in_channels != out_channels or upsample:
      self.shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)

    if attention:
      self.self_attention = SelfAttention(out_channels)

  def forward(self, x, z_cond): # z_cond is the combination of z_split and class_embed
    # Residual connection
    x_shortcut = x
    if self.upsample:
      x_shortcut = F.interpolate(x_shortcut, scale_factor=2, mode='nearest')
    if self.shortcut_conv: # Apply 1x1 conv if needed for channel matching
      x_shortcut = self.shortcut_conv(x_shortcut)

    # Main path
    out = F.relu(self.cbn1(x, z_cond))
    if self.upsample:
      out = F.interpolate(out, scale_factor=2, mode='nearest')
    out = self.conv1(out)

    out = F.relu(self.cbn2(out, z_cond))
    out = self.conv2(out)

    out = out + x_shortcut

    if self.attention:
      out = self.self_attention(out)

    return out


# --- SNConv2d (no change needed from previous correct version) ---
class SNConv2d(nn.Conv2d):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    spectral_norm(self)

# --- SNLinear (no change needed from previous correct version) ---
class SNLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        spectral_norm(self)

# --- Corrected DisBlock (fixed residual connection logic and use of SNConv2d) ---
class DisBlock(nn.Module):
  def __init__(self, in_channels, out_channels, downsample=True, attention=False):
    super(DisBlock, self).__init__()
    self.downsample = downsample
    self.attention = attention

    self.conv1 = SNConv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
    self.conv2 = SNConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
    # Removed inplace=True from LeakyReLU
    self.relu = nn.LeakyReLU(0.2)

    self.shortcut_conv = None
    if in_channels != out_channels or downsample:
      self.shortcut_conv = SNConv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)

    if attention:
      self.self_attention = SelfAttention(out_channels)

  def forward(self, x):
    # Shortcut path
    x_shortcut = x
    if self.downsample:
        x_shortcut = F.avg_pool2d(x_shortcut, 2)
    if self.shortcut_conv:
        x_shortcut = self.shortcut_conv(x_shortcut)

    # Main path
    out = self.relu(self.conv1(self.relu(x)))
    out = self.conv2(out)

    if self.downsample:
      out = F.avg_pool2d(out, 2)

    out = out + x_shortcut

    if self.attention:
      out = self.self_attention(out)
    return out