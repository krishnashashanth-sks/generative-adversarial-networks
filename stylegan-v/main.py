# Install necessary libraries. PyTorch is commonly used for GAN implementations.
# !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# !pip install numpy scipy tqdm pillow matplotlib opencv-python
# !pip install ninja # Often needed for some PyTorch extensions or custom ops

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os
from layers import MappingNetwork
from generator import Generator
from discriminator import Discriminator
from inference import inference

# Check for GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

if device == 'cuda':
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"CUDA memory cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

# Define dimensions
z_dim_val = 512       # Dimension of the input latent vector z
w_dim_val = 512       # Dimension of the intermediate disentangled latent vector w
# For a 64x64 resolution, typically 2*(log2(64)-1) = 2*(6-1) = 10 style vectors for 2D.
# For video, this might be multiplied by num_frames or be more complex.
# Let's assume a reasonable number for now.
num_ws_val = 14 # Example: 2 blocks per resolution level, from 4x4 up to 64x64, + a few for temporal/frame control

# Instantiate the mapping network
mapping_net = MappingNetwork(z_dim=z_dim_val, w_dim=w_dim_val, num_ws=num_ws_val).to(device)
print(f"MappingNetwork instantiated and moved to {device}.")

# Define dimensions (ensure these match with MappingNetwork setup)
z_dim = 512
w_dim = 512
num_frames = 8    # Number of frames in the generated video (e.g., 8 frames)
img_channels = 3  # RGB video (3 channels)
img_resolution = 32 # Spatial resolution of video frames (e.g., 32x32)
max_res_log2 = 5  # log2(32) = 5 (i.e., resolutions 4, 8, 16, 32)
ema_beta = 0.999 # Exponential Moving Average beta parameter

# Instantiate the generator
generator = Generator(z_dim=z_dim, w_dim=w_dim, num_frames=num_frames,
                      img_channels=img_channels, img_resolution=img_resolution,
                      max_res_log2=max_res_log2,
                      temporal_kernel_size=(1,3,3),
                      truncation_psi=0.7, # Example truncation strength
                      truncation_cutoff=8 # Example: apply truncation to first 8 style vectors (4 blocks)
                     ).to(device)
print(f"Generator instantiated and moved to {device}.")

# Define dimensions (ensure these match with Generator setup)
img_channels = 3  # RGB video (3 channels)
img_resolution = 32 # Spatial resolution of video frames (e.g., 32x32)
max_res_log2 = 5  # log2(32) = 5 (i.e., resolutions 4, 8, 16, 32)
num_frames_disc = 8 # Number of frames in the input video for discriminator

# Re-instantiate the discriminator to ensure it uses the latest class definition
discriminator = Discriminator(img_channels=img_channels, img_resolution=img_resolution,
                              num_frames=num_frames_disc, # Added this argument
                              max_res_log2=max_res_log2,
                              temporal_kernel_size=(1,3,3),
                              base_channels=32 # Start with 32 channels for 4x4 resolution
                             ).to(device)
print(f"Discriminator instantiated and moved to {device}.")

# Initialize mean_w for the generator (it's already done in __init__, but explicitly showing here)
# For a fresh start, you might want to load a pre-computed mean_w or start with zeros
generator.mean_w = torch.zeros(w_dim, device=device)

# Simulate a few training iterations to update mean_w
num_simulated_steps = 100
batch_size = 4 # Example batch size

print(f"Initial generator.mean_w (first 5 elements): {generator.mean_w[:5].cpu().numpy()}")

for i in range(num_simulated_steps):
    # Generate random latent vectors z
    dummy_z = torch.randn(batch_size, z_dim, device=device)

    # Get the w_styles from the mapping network
    # We only need the w_styles, not the full video output, for mean_w computation
    with torch.no_grad(): # No gradients needed for mean_w computation
        w_styles = generator.mapping_network(dummy_z)

    # Average w_styles across batch and num_ws to get a single mean w vector for this batch
    current_batch_mean_w = w_styles.mean(dim=(0, 1))

    # Update the running average using EMA
    if i == 0:
        generator.mean_w.copy_(current_batch_mean_w) # For the first step, just assign
    else:
        generator.mean_w.mul_(ema_beta).add_(current_batch_mean_w * (1 - ema_beta))

    if (i + 1) % 20 == 0:
        print(f"Step {i+1}: Updated generator.mean_w (first 5 elements): {generator.mean_w[:5].cpu().numpy()}")

print(f"\nFinal generator.mean_w (first 5 elements) after {num_simulated_steps} steps: {generator.mean_w[:5].cpu().numpy()}")
print(f"Shape of generator.mean_w: {generator.mean_w.shape}")

# Now, when generator.forward() is called, it will use this updated mean_w for truncation.
print("Mean_w computation simulated successfully.")

# --- Optimizers ---

import torch.optim as optim

# Define learning rates and betas for Adam optimizer
# These are typical values for GAN training, but might need tuning
g_lr = 0.002
d_lr = 0.002
beta1 = 0.0
beta2 = 0.99 # Often used with StyleGAN

# Instantiate optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=g_lr, betas=(beta1, beta2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=d_lr, betas=(beta1, beta2))

print("Optimizers (Adam for Generator and Discriminator) instantiated.")
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from dataset import VideoDataset,video_files

# Define transformations for individual frames (C, H, W)
video_transform = transforms.Compose([
    transforms.ConvertImageDtype(torch.float32), # Convert to float and scale to [0,1]
    transforms.Resize((img_resolution, img_resolution)), # Resize spatial dimensions
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize to [-1, 1]
])

# Instantiate the dataset
# Ensure 'num_frames' matches what the generator expects
ucf101_dataset = VideoDataset(
    video_paths=video_files,
    num_frames=num_frames, # Using the num_frames defined earlier (e.g., 8)
    transform=video_transform,
    sample_strategy='random'
)

print(f"VideoDataset instantiated with {len(ucf101_dataset)} videos.")

# Create the DataLoader
batch_size = 4 # Example batch size
data_loader = DataLoader(
    ucf101_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2, # Use multiple workers for faster data loading
    pin_memory=True # Pin tensors to GPU memory for faster transfer if using CUDA
)

print(f"DataLoader instantiated with batch_size={batch_size}.")

# Test fetching a batch
for batch in data_loader:
    print(f"Shape of one batch from DataLoader: {batch.shape}")
    break

print("Data loading setup complete. Now ready for training loop.")

from train import train_loop
num_epochs = 10 # Define the number of epochs for training

# Call the training loop function
g_losses, d_losses, r1_penalties = train_loop(generator, discriminator, optimizer_G, optimizer_D, data_loader, device,
                                            num_epochs=num_epochs, r1_gamma=10.0, lazy_r1_interval=16, ema_beta_w=0.999)
z=torch.randn(1,z_dim,device=device)
generated_video=inference(generator,z,device)