import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as spectral_utils

class PixelNorm(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self,input):
    return input*torch.sqrt(torch.mean(input**2,dim=1,keepdim=True)+1e-8)

class MappingNetwork(nn.Module):
  def __init__(self,z_dim,w_dim,num_mapping_layers=8,num_ws=18):
    super().__init__()
    self.z_dim=z_dim
    self.w_dim=w_dim
    self.num_ws=num_ws
    layers=[PixelNorm()]
    for i in range(num_mapping_layers):
      in_features=z_dim if i==0 else w_dim
      layers.append(nn.Linear(in_features,w_dim))
      layers.append(nn.LeakyReLU(0.2))
    self.mapping_mlp=nn.Sequential(*layers)
    self.output_transform=nn.Linear(w_dim,w_dim*num_ws)
  def forward(self,z):
    if z.ndim!=2 or z.shape[1]!=self.z_dim:
      raise ValueError(f"Input latent vector z must have shape (batch_size, {self.z_dim}), but got {z.shape}")
    w_base=self.mapping_mlp(z)
    w_expanded=self.output_transform(w_base).view(-1,self.num_ws,self.w_dim)
    return w_expanded

# # --- Example Usage (adjust parameters as needed) ---

# # Define dimensions
# z_dim_val = 512       # Dimension of the input latent vector z
# w_dim_val = 512       # Dimension of the intermediate disentangled latent vector w
# # For a 64x64 resolution, typically 2*(log2(64)-1) = 2*(6-1) = 10 style vectors for 2D.
# # For video, this might be multiplied by num_frames or be more complex.
# # Let's assume a reasonable number for now.
# num_ws_val = 14 # Example: 2 blocks per resolution level, from 4x4 up to 64x64, + a few for temporal/frame control

# # Instantiate the mapping network
# mapping_net = MappingNetwork(z_dim=z_dim_val, w_dim=w_dim_val, num_ws=num_ws_val).to(device)
# print(f"MappingNetwork instantiated and moved to {device}.")

# # Create a dummy latent vector 'z'
# batch_size = 2
# dummy_z = torch.randn(batch_size, z_dim_val).to(device) # Batch size of 2

# # Pass through the mapping network
# dummy_w = mapping_net(dummy_z)

# print(f"Input z shape: {dummy_z.shape}")
# print(f"Output w (expanded) shape: {dummy_w.shape}") # Expected: (batch_size, num_ws, w_dim)

# print("Mapping network implemented and tested with dummy input.")

class ScaledLeakyReLU(nn.Module):
  def __init__(self,negative_slope=0.2):
    super().__init__()
    self.negative_slope=negative_slope
  def forward(self,x):
    return F.leaky_relu(x,negative_slope=self.negative_slope)
  
class ModulatedConv3d(nn.Module):
  def __init__(self,in_channels,out_channels,kernel_size,style_dim,
               demodulate=True,upsample=False,padding=1):
    super().__init__()
    self.eps=1e-8
    self.in_channels=in_channels
    self.out_channels=out_channels
    if isinstance(kernel_size, int):
      self.kernel_size = (1, kernel_size, kernel_size) # Default to spatial convs per frame
    else:
      self.kernel_size = kernel_size
    self.upsample=upsample
    self.demodulate=demodulate
    self.weight=nn.Parameter(torch.randn(1,out_channels,in_channels,*self.kernel_size))
    self.modulation=nn.Linear(style_dim,in_channels)
    self.bias=nn.Parameter(torch.zeros(1,out_channels,1,1,1)) if not demodulate else None # Corrected: Added an extra '1' dimension for temporal axis
    self.padding=padding
  def _get_upsample_filter(self,kernel_size):
    factor=2
    if kernel_size%2==1:
      pad=kernel_size//2
      res=torch.zeros(kernel_size,kernel_size,device=self.weight.device)
      res[pad,pad]=1
    else:
      res=torch.ones(kernel_size,kernel_size,device=self.weight.device)
    f=torch.zeros(factor,factor,device=self.weight.device)
    f[0,0]=1
    return f.view(1,1,factor,factor).repeat(self.out_channels,1,1,1)
  def forward(self,x,w):
    batch_size=x.size(0)
    s=self.modulation(w).view(batch_size,1,self.in_channels,1,1,1)
    weight=self.weight*s
    if self.demodulate:
      d=torch.rsqrt((weight**2).sum(dim=(2,3,4,5),keepdim=True)+self.eps)
      weight=weight*d
    weight=weight.view(batch_size*self.out_channels,self.in_channels,*self.kernel_size)
    x=x.view(1,batch_size*self.in_channels,*x.shape[2:])
    if self.upsample:
      x = F.interpolate(x, scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
    out=F.conv3d(x,weight,padding=self.padding,groups=batch_size)
    out=out.view(batch_size,self.out_channels,*out.shape[2:])
    if self.bias is not None:
      out=out+self.bias
    return out

class NoiseInjection(nn.Module):
  def __init__(self):
    super().__init__()
    self.weight=nn.Parameter(torch.zeros(1))
  def forward(self,image):
    noise=torch.randn(image.size(0),1,*image.shape[2:],device=image.device)
    return image+self.weight*noise
  
class SelfAttention3d(nn.Module):
  def __init__(self,in_channels):
    super().__init__()
    self.in_channels=in_channels
    self.query_conv=nn.Conv3d(in_channels,in_channels//8,kernel_size=1)
    self.key_conv=nn.Conv3d(in_channels,in_channels//8,kernel_size=1)
    self.value_conv=nn.Conv3d(in_channels,in_channels//8,kernel_size=1)
    self.gamma=nn.Parameter(torch.zeros(1))
    self.softmax=nn.Softmax(dim=-1)
  def forward(self,x):
    batch_size,C,T,H,W=x.size()
    proj_query=self.query_conv(x).view(batch_size,-1,T*H*W).permute(0,2,1)
    proj_key=self.key_conv(x).view(batch_size,-1,T*H*W)
    energy=torch.bmm(proj_query,proj_key)
    attention=self.softmax(energy)
    proj_value=self.value_conv(x).view(batch_size,-1,T*H*W)
    out=torch.bmm(proj_value,attention.permute(0,2,1))
    out=out.view(batch_size,C,T,H,W)
    return self.gamma*out+x

class SynthesisBlock(nn.Module):
  def __init__(self,in_channels,out_channels,style_dim,num_frames_initial,initial_block=False,
               add_noise=True,upsample=False,temporal_kernel_size=(1,3,3), use_attention=False):
    super().__init__()
    # self.resolution=resolution # No longer directly used as an arg, derived from output of ModulatedConv3d
    self.add_noise=add_noise
    self.initial_block=initial_block
    self.upsample=upsample
    self.lrelu=ScaledLeakyReLU()

    # Correct padding for temporal_kernel_size=(Kt, Kh, Kw)
    # If temporal_kernel_size is (Kt, Kh, Kw), padding should be (Kt//2, Kh//2, Kw//2)
    # Assuming standard (1,3,3) or (3,3,3) kernel, padding is (0 or 1, 1, 1)
    kernel_t, kernel_h, kernel_w = temporal_kernel_size
    padding_val = (kernel_t // 2, kernel_h // 2, kernel_w // 2)

    if initial_block:
      # For StyleGAN-V, initial constant input is 5D: (batch, C, T, H, W)
      self.constant_input=nn.Parameter(torch.randn(1,out_channels,num_frames_initial,4,4))
      # First conv in initial block usually doesn't upsample, but processes the constant
      self.conv1=ModulatedConv3d(out_channels,out_channels,kernel_size=temporal_kernel_size,style_dim=style_dim,upsample=False,padding=padding_val)
    else:
      # Subsequent blocks handle upsampling spatially
      self.conv1=ModulatedConv3d(in_channels,out_channels,kernel_size=temporal_kernel_size,style_dim=style_dim,upsample=upsample,padding=padding_val)

    self.noise1=NoiseInjection() if add_noise else None
    self.conv2=ModulatedConv3d(out_channels,out_channels,kernel_size=temporal_kernel_size,style_dim=style_dim,padding=padding_val)
    self.noise2=NoiseInjection() if add_noise else None

    self.attention = SelfAttention3d(out_channels) if use_attention else None

  def forward(self,x,w1,w2):
    if self.initial_block:
      x=self.constant_input.repeat(w1.size(0),1,1,1,1) # Repeat for batch and temporal dims

    x=self.conv1(x,w1)
    if self.noise1: x=self.noise1(x)
    x=self.lrelu(x)

    if self.attention: # Apply attention after first convolution block
        x = self.attention(x)

    x=self.conv2(x,w2)
    if self.noise2:x=self.noise2(x)
    return self.lrelu(x)
  
class MinibatchStdDev(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self,x):
    batch_size,channels,frames,height,width=x.shape
    stddev=torch.sqrt(x.var(dim=0,keepdim=True)+1e-8)
    stddev=stddev.mean([1,2,3,4],keepdim=True)
    stddev=stddev.repeat(batch_size,1,frames,height,width)
    return torch.cat([x, stddev],1)

