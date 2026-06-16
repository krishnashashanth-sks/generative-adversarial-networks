import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.nn.utils as spectral_utils
from layers import ScaledLeakyReLU,MinibatchStdDev
import torch.nn.utils as spectral_utils

class DiscriminatorBlock(nn.Module):
  def __init__(self,in_channels,out_channels,temporal_kernel_size=(1,3,3),downsample=True):
    super().__init__()
    self.downsample=downsample
    self.lrelu=ScaledLeakyReLU()
    pad_t=temporal_kernel_size[0]//2
    pad_h=temporal_kernel_size[1]//2
    pad_w=temporal_kernel_size[2]//2
    self.padding=(pad_t,pad_h,pad_w)
    self.conv1=spectral_utils.spectral_norm(
        nn.Conv3d(in_channels,out_channels,kernel_size=temporal_kernel_size,padding=self.padding)
    )
    self.conv2=spectral_utils.spectral_norm(
        nn.Conv3d(out_channels,out_channels,kernel_size=temporal_kernel_size,padding=self.padding)
    )
    if downsample:
      self.downsample_layer=nn.AvgPool3d(kernel_size=(1,2,2))
    self.skip=spectral_utils.spectral_norm(
        nn.Conv3d(in_channels,out_channels,kernel_size=(1,1,1))
    )
  def forward(self,x):
    h=self.lrelu(self.conv1(x))
    h=self.lrelu(self.conv2(h))
    if self.downsample:
      h=self.downsample_layer(h)
    x_skip=self.skip(x)
    if self.downsample:
      x_skip=self.downsample_layer(x_skip)
    return (h+x_skip)

class Discriminator(nn.Module):
    def __init__(self, img_channels, img_resolution, num_frames, max_res_log2,
                 temporal_kernel_size=(1, 3, 3), base_channels=32):
        super().__init__()
        self.img_channels = img_channels
        self.img_resolution = img_resolution
        self.max_res_log2 = max_res_log2
        self.temporal_kernel_size = temporal_kernel_size
        self.num_frames = num_frames # Store num_frames for final linear layer
        self.lrelu = ScaledLeakyReLU()

        # Define channel map for each resolution level (log2 value)
        # From 4x4 (res_log2=2) up to max_res_log2
        channel_map = {
            2**res_log2: min(512, base_channels * (2**(res_log2 - 2)))
            for res_log2 in range(2, max_res_log2 + 1)
        }

        # Initial from_rgb layer for the highest resolution, including activation
        highest_res_channels = channel_map[img_resolution]
        self.from_rgb = nn.Sequential(
            spectral_utils.spectral_norm(
                nn.Conv3d(img_channels, highest_res_channels, kernel_size=(1, 1, 1))
            ),
            self.lrelu
        )

        self.blocks = nn.ModuleList()
        in_ch = highest_res_channels

        # Iterate from max_res_log2 down to 3 (processing 32x32 -> 16x16 -> 8x8 blocks)
        for res_log2 in range(max_res_log2, 2, -1): # Example: for max_res_log2=5, this is 5, 4, 3
            out_ch = channel_map[2**(res_log2 - 1)] # Channels for the next lower resolution
            self.blocks.append(DiscriminatorBlock(in_ch, out_ch,
                                                  temporal_kernel_size=temporal_kernel_size,
                                                  downsample=True))
            in_ch = out_ch # Update in_ch for the next iteration

        # Final block for 4x4 resolution processing
        # At this point, in_ch is channel_map[4]
        self.final_block_part1 = nn.Sequential(
            MinibatchStdDev(), # Adds 1 channel, so input to next conv is in_ch + 1
            spectral_utils.spectral_norm(
                nn.Conv3d(in_ch + 1, in_ch, kernel_size=temporal_kernel_size, padding=temporal_kernel_size[0]//2)
            ),
            ScaledLeakyReLU()
        )
        self.final_block_part2 = nn.Sequential(
            # Replace Conv3d with AvgPool3d to explicitly reduce spatial dimensions to 1x1
            nn.AvgPool3d(kernel_size=(1,2,2)), # Reduce 2x2 to 1x1 spatially
            # Removed spectral_norm as AvgPool3d does not have learnable weights
            # ScaledLeakyReLU() # Often not used directly after pooling unless followed by another conv
            nn.Flatten(), # Flattens (batch, in_ch, num_frames, 1, 1) to (batch, in_ch * num_frames)
            nn.Linear(in_ch * self.num_frames, 1) # Output a single score
        )
        self.in_ch = in_ch # Store in_ch for debugging

    def forward(self, x):
        # x: (batch_size, img_channels, num_frames, img_resolution, img_resolution)
        x = self.from_rgb(x)

        for i, block in enumerate(self.blocks):
            x = block(x)

        x = self.final_block_part1(x)

        x = self.final_block_part2(x)

        return x
    
# # --- Example Usage for Discriminator ---

# # Define dimensions (ensure these match with Generator setup)
# img_channels = 3  # RGB video (3 channels)
# img_resolution = 32 # Spatial resolution of video frames (e.g., 32x32)
# max_res_log2 = 5  # log2(32) = 5 (i.e., resolutions 4, 8, 16, 32)
# num_frames_disc = 8 # Number of frames in the input video for discriminator

# # Re-instantiate the discriminator to ensure it uses the latest class definition
# discriminator = Discriminator(img_channels=img_channels, img_resolution=img_resolution,
#                               num_frames=num_frames_disc, # Added this argument
#                               max_res_log2=max_res_log2,
#                               temporal_kernel_size=(1,3,3),
#                               base_channels=32 # Start with 32 channels for 4x4 resolution
#                              ).to(device)
# print(f"Discriminator instantiated and moved to {device}.")

# # Create a dummy video input
# batch_size = 2
# dummy_video_input = torch.randn(batch_size, img_channels, num_frames_disc, img_resolution, img_resolution).to(device)

# # Pass through the discriminator
# dummy_output = discriminator(dummy_video_input)

# print(f"Input video shape: {dummy_video_input.shape}")
# print(f"Discriminator output shape: {dummy_output.shape}") # Expected: (batch_size, 1) for a realism score