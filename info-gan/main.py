from generator import Generator
from discriminator import Discriminator
import torch.optim as optim
import torch
from train import train
from dataset import dataloader
from losses import *
import matplotlib.pyplot as plt

# --- Hyperparameters (Example) ---
noise_dim = 100
latent_dim_categorical = 10 # e.g., for 10 classes in MNIST
latent_dim_continuous = 2   # e.g., for rotation and thickness
img_dim = 784               # e.g., for 28x28 MNIST images

# --- Instantiate Models ---
gen = Generator(noise_dim, latent_dim_categorical, latent_dim_continuous, img_dim)
disc = Discriminator(img_dim, latent_dim_categorical, latent_dim_continuous)

optimizer_g = optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = optim.Adam(disc.parameters(), lr=0.0002, betas=(0.5, 0.999))

epochs = 50
batch_size = 64

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move models to device
gen.to(device)
disc.to(device)

# Mutual information weighting factor
lambda_mi = 1.0 # This is a hyperparameter, often set to 1.0

train(epochs,dataloader,gen,disc,criterion_gan,criterion_categorical,criterion_continuous,optimizer_g,optimizer_d,lambda_mi,latent_dim_categorical,latent_dim_continuous,noise_dim,device)

import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import types # Import types for MethodType

# Define a corrected forward method for the Generator
def _corrected_generator_forward(self, x, c_cat, c_cont):
    input_vec = torch.cat([x, c_cat, c_cont], dim=1)
    return self.main(input_vec)

# Assign the corrected method to the generator instance
gen.forward = types.MethodType(_corrected_generator_forward, gen)

# Ensure generator is in evaluation mode
gen.eval()

# Number of images to generate per row/column for visualization
num_images = 10

# Fixed noise vector for consistency
fixed_noise = torch.randn(1, noise_dim, device=device)

# --- Visualize categorical disentanglement ---
# Generate images by varying categorical code, keeping continuous codes and noise fixed
print("Generating images by varying categorical codes...")
img_list_cat = []
for i in range(latent_dim_categorical): # Iterate through each categorical class
    # Repeat fixed noise for the batch
    z_batch = fixed_noise.repeat(num_images, 1)

    # Vary the categorical code
    c_cat_sample = torch.full((num_images,), i, dtype=torch.long, device=device)
    c_cat_one_hot = torch.nn.functional.one_hot(c_cat_sample, num_classes=latent_dim_categorical).float().to(device)

    # Fix continuous codes (e.g., to zeros or mid-range)
    c_cont_sample = torch.zeros(num_images, latent_dim_continuous, device=device)

    with torch.no_grad():
        fake_imgs = gen(z_batch, c_cat_one_hot, c_cont_sample).cpu()
    img_list_cat.append(fake_imgs)

# Reshape and display
fake_imgs_cat = torch.cat(img_list_cat)
grid_cat = make_grid(fake_imgs_cat, nrow=num_images, normalize=True, padding=2, pad_value=1)

plt.figure(figsize=(12, 12))
plt.imshow(np.transpose(grid_cat.numpy(), (1, 2, 0)))
plt.title("Generated Images: Varying Categorical Code (Digits)")
plt.axis("off")
plt.show()

# --- Visualize continuous disentanglement (e.g., rotation/thickness) ---
# Generate images by varying one continuous code, keeping others and categorical code fixed
print("Generating images by varying continuous codes...")
img_list_cont = []

# Pick a categorical code to fix (e.g., digit '1')
fixed_c_cat_idx = 1 # You can change this to any digit from 0-9
fixed_c_cat_one_hot = torch.nn.functional.one_hot(torch.tensor([fixed_c_cat_idx], device=device), num_classes=latent_dim_categorical).float()

# Create a range for one continuous latent code (e.g., from -1 to 1)
continuous_range = torch.linspace(-1, 1, num_images, device=device)

for j in range(latent_dim_continuous):
    # Repeat fixed noise and categorical code
    z_batch = fixed_noise.repeat(num_images, 1)
    c_cat_batch = fixed_c_cat_one_hot.repeat(num_images, 1)

    c_cont_batch = torch.zeros(num_images, latent_dim_continuous, device=device)
    c_cont_batch[:, j] = continuous_range # Vary only the j-th continuous code

    with torch.no_grad():
        fake_imgs = gen(z_batch, c_cat_batch, c_cont_batch).cpu()
    img_list_cont.append(fake_imgs)

# Reshape and display
fake_imgs_cont = torch.cat(img_list_cont)
grid_cont = make_grid(fake_imgs_cont, nrow=num_images, normalize=True, padding=2, pad_value=1)

plt.figure(figsize=(12, 12))
plt.imshow(np.transpose(grid_cont.numpy(), (1, 2, 0)))
plt.title(f"Generated Images: Varying Continuous Codes (Fixed Categorical: {fixed_c_cat_idx})")
plt.axis("off")
plt.show()

gen.train() # Set generator back to training mode if further training is planned