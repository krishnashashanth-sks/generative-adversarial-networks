import os
import torch
import torch.nn as nn
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

# Local imports from your separated files
from generator import Generator
from discriminator import Discriminator
from utils import weights_init,show_images
from dataset import train_loader  # Updated to use the loader function
from train import train                # Added missing train function import

def main():
    # Set runtime device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define missing hyperparameter constants
    nz = 100
    epochs = 20  # Added missing epochs definition
    batch_size = 64
    
    # Create required local folders
    os.makedirs("../data", exist_ok=True)
    os.makedirs("../outputs", exist_ok=True)

    # Initialize models (Corrected Generator parameter mapping: nz, ngf, nc)
    netG = Generator(nz=100, ngf=64, nc=1)
    netG.apply(weights_init)

    netD = Discriminator(nc=1, ndf=64)
    netD.apply(weights_init)

    # Move models to the active hardware accelerator
    netG.to(device)
    netD.to(device)

    # Setup optimization parameters
    criterion = nn.BCELoss()
    optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Static tensor tracking structural adjustments across timeline
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Execute the training procedure
    G_losses, D_losses, image_list = train(
        netG=netG,
        netD=netD,
        train_loader=train_loader,
        optimizerG=optimizerG,
        optimizerD=optimizerD,
        criterion=criterion,
        device=device,
        epocs=epochs,        # Matches parameter name inside train.py function signature
        nz=nz,
        fixed_noise=fixed_noise
    )

    # Plot the loss dynamics
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="Generator (G)")
    plt.plot(D_losses, label="Discriminator (D)")
    plt.xlabel("Logged Steps (Every 100 Batches)")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Visualize generated image updates saved across execution milestones
    print("\nVisualizing generated images at various iterations...")
    if len(image_list) > 0:
        interval = max(1, len(image_list) // 5)
        for j, img_batch in enumerate(image_list):
            if j % interval == 0 or j == len(image_list) - 1:
                print(f"-> Displaying structural milestone snapshot index #{j}:")
                show_images(img_batch)

    # Final generated image validation grid
    print("\nDisplaying final generated images matrix from fixed noise:")
    with torch.no_grad():
        final_fake_images = netG(fixed_noise).detach().cpu()
    
    # Convert separate image outputs cleanly into a neat grid layout before printing
    final_grid = vutils.make_grid(final_fake_images, padding=2, normalize=True)
    show_images(final_grid)

if __name__ == "__main__":
    main()
