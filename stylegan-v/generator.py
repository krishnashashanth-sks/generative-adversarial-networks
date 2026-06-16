import torch
import torch.nn as nn
from layers import MappingNetwork,SynthesisBlock,ModulatedConv3d

class Generator(nn.Module):
  def __init__(self,z_dim,w_dim,num_frames,img_channels,img_resolution,
               num_mapping_layers=8,max_res_log2=6,temporal_kernel_size=(1,3,3),
               use_attention_at_res=None, truncation_psi=0.7, truncation_cutoff=8, **kwargs):
    super().__init__()
    self.z_dim=z_dim
    self.w_dim=w_dim
    self.num_frames=num_frames
    self.img_channels=img_channels
    self.img_resolution=img_resolution
    self.max_res_log2=max_res_log2
    self.temporal_kernel_size=temporal_kernel_size
    self.use_attention_at_res = use_attention_at_res if use_attention_at_res is not None else []

    # Truncation trick parameters
    self.truncation_psi = truncation_psi
    self.truncation_cutoff = truncation_cutoff
    self.register_buffer('mean_w', torch.zeros(w_dim)) # Will be updated during training

    # Update num_ws: Each SynthesisBlock has 2 modulated convs, plus 1 for to_rgb
    self.num_ws = (max_res_log2 * 2) + 1 # max_res_log2 blocks * 2 w's per block + 1 w for to_rgb
    self.mapping_network=MappingNetwork(z_dim,w_dim,num_mapping_layers=num_mapping_layers,num_ws=self.num_ws)
    self.synthesis_blocks=nn.ModuleList()

    current_channels=w_dim
    # Initial block (4x4 resolution)
    self.synthesis_blocks.append(SynthesisBlock(0,current_channels,w_dim,
                                                num_frames_initial=num_frames,initial_block=True,
                                                add_noise=True,temporal_kernel_size=temporal_kernel_size,
                                                use_attention=(4 in self.use_attention_at_res)))

    # Subsequent blocks (upsampling from 8x8 to img_resolution)
    for i in range(2,max_res_log2+1):
      res=2**i
      in_ch=current_channels
      out_ch=w_dim # Assuming constant channel width for simplicity in synthesis blocks
      self.synthesis_blocks.append(SynthesisBlock(in_ch,out_ch,w_dim,num_frames_initial=num_frames,
                                                  initial_block=False,upsample=True,add_noise=True,
                                                  temporal_kernel_size=temporal_kernel_size,
                                                  use_attention=(res in self.use_attention_at_res)))
      current_channels=out_ch

    self.to_rgb=ModulatedConv3d(current_channels,img_channels,kernel_size=(1,1,1),
                                style_dim=w_dim,demodulate=False,padding=0)

  def forward(self,z):
    w_styles=self.mapping_network(z)

    # Apply truncation trick
    if self.truncation_psi < 1.0:
        # The truncation cutoff determines at which layers the truncation is applied.
        # Each SynthesisBlock uses 2 w vectors.
        # For example, if truncation_cutoff=8, it applies to the first 8 w vectors, meaning the first 4 blocks.
        # This ensures that styles applied to coarser resolutions are truncated.
        truncated_w = self.mean_w.unsqueeze(0).unsqueeze(0).expand_as(w_styles)
        w_styles[:, :self.truncation_cutoff, :] = truncated_w[:, :self.truncation_cutoff, :] + \
                                                  self.truncation_psi * (w_styles[:, :self.truncation_cutoff, :] - truncated_w[:, :self.truncation_cutoff, :])

    x=None
    w_idx = 0
    for i,block in enumerate(self.synthesis_blocks):
      w1 = w_styles[:,w_idx,:]
      w2 = w_styles[:,w_idx+1,:]
      x=block(x,w1,w2)
      w_idx+=2
    final_rgb_w=w_styles[:,w_idx,:] # The last remaining w for to_rgb
    video_output=self.to_rgb(x,final_rgb_w) # Pass the feature map x along with final_rgb_w
    return torch.tanh(video_output)

# # --- Example Usage (similar to previous) ---
# # Define dimensions (ensure these match with MappingNetwork setup)
# z_dim = 512
# w_dim = 512
# num_frames = 8    # Number of frames in the generated video (e.g., 8 frames)
# img_channels = 3  # RGB video (3 channels)
# img_resolution = 32 # Spatial resolution of video frames (e.g., 32x32)
# max_res_log2 = 5  # log2(32) = 5 (i.e., resolutions 4, 8, 16, 32)

# # Instantiate the generator
# generator = Generator(z_dim=z_dim, w_dim=w_dim, num_frames=num_frames,
#                       img_channels=img_channels, img_resolution=img_resolution,
#                       max_res_log2=max_res_log2,
#                       temporal_kernel_size=(1,3,3),
#                       truncation_psi=0.7, # Example truncation strength
#                       truncation_cutoff=8 # Example: apply truncation to first 8 style vectors (4 blocks)
#                      ).to(device)
# print(f"Generator instantiated and moved to {device}.")

# # Create a dummy latent vector 'z'
# batch_size = 2
# dummy_z = torch.randn(batch_size, z_dim).to(device)

# # Pass through the generator
# dummy_video = generator(dummy_z)

# print(f"Input z shape: {dummy_z.shape}")
# print(f"Output video shape: {dummy_video.shape}") # Expected: (batch_size, channels, num_frames, height, width)