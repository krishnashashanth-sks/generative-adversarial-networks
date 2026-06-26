import torch
import torchvision.transforms as transforms
import torch.optim as optim
from losses import WGAN_GP_Loss
from tensorflow.datasets import CIFAR10
from utils import get_current_resolution,save_samples
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch.nn as nn

def train_step(real_images, generator, discriminator, gen_optimizer, disc_optimizer, criterion, current_stage, alpha, latent_dim, device):
    batch_size = real_images.size(0)

    # ------------------
    # Train Discriminator
    # ------------------
    disc_optimizer.zero_grad()

    # Generate fake images
    z = torch.randn(batch_size, latent_dim).to(device)
    fake_images = generator(z)

    # Update Generator and Discriminator's alpha for progressive growing
    generator.update_stage(current_stage, alpha)
    discriminator.update_stage(current_stage, alpha)

    # Calculate discriminator loss
    d_loss = criterion.discriminator_loss(discriminator, real_images, fake_images, current_stage)
    d_loss.backward()
    disc_optimizer.step()

    # --------------
    # Train Generator
    # --------------
    gen_optimizer.zero_grad()

    # Generate new fake images (no detach for generator training)
    z = torch.randn(batch_size, latent_dim).to(device)
    fake_images = generator(z)

    # Update Generator and Discriminator's alpha for progressive growing
    generator.update_stage(current_stage, alpha)
    discriminator.update_stage(current_stage, alpha)

    # Calculate generator loss
    g_loss = criterion.generator_loss(discriminator, fake_images)
    g_loss.backward()
    gen_optimizer.step()

    return d_loss.item(), g_loss.item()

def train_gan(generator, discriminator, latent_dim, num_stages, config):
    device = config['device']
    generator.to(device)
    discriminator.to(device)

    # Optimizers
    gen_optimizer = optim.Adam(generator.parameters(), lr=config['lr_g'], betas=(0.0, 0.99))
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=config['lr_d'], betas=(0.0, 0.99))

    # Loss function
    criterion = WGAN_GP_Loss(lambda_gp=config['lambda_gp'], device=device)

    # Data loading
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)

    global_step = 0
    for current_stage in range(num_stages):
        current_resolution = get_current_resolution(current_stage)
        print(f"\n--- Training at resolution: {current_resolution}x{current_resolution} ---")

        # Adjust batch size based on resolution (smaller batch for higher resolution)
        batch_size = config['batch_sizes'].get(current_resolution, config['batch_sizes']['default'])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=config['num_workers'], pin_memory=True)

        # --- Fade-in phase ---
        if current_stage > 0:
            print(f"Starting fade-in for stage {current_stage}")
            # Generator and discriminator's update_stage is called inside train_step
            num_fade_in_steps_per_batch = config['fade_in_steps_per_stage'] // len(dataloader)
            if num_fade_in_steps_per_batch == 0: num_fade_in_steps_per_batch = 1 # Ensure at least one step

            for i, (real_images, _) in tqdm.tqdm(enumerate(dataloader), desc=f"Fade-in {current_resolution}x{current_resolution}"):
                alpha = min(1.0, (global_step - config['start_steps_per_stage'][current_stage]) / config['fade_in_steps_per_stage'])

                real_images = real_images.to(device)
                if real_images.shape[2] != current_resolution:
                    real_images = nn.functional.interpolate(real_images, size=(current_resolution, current_resolution), mode='bilinear', align_corners=False)

                d_loss, g_loss = train_step(real_images, generator, discriminator, gen_optimizer, disc_optimizer, criterion, current_stage, alpha, latent_dim, device)
                global_step += 1

                if (i + 1) % config['sample_interval'] == 0:
                    save_samples(generator, latent_dim, current_stage, device=device)

        # --- Stabilization phase ---
        print(f"Starting stabilization for stage {current_stage}")
        # Generator and discriminator's update_stage is called inside train_step
        num_stabilization_batches = config['stabilization_epochs_per_stage'] * len(dataloader)

        for epoch in tqdm.tqdm(range(config['stabilization_epochs_per_stage']), desc=f"Stabilization {current_resolution}x{current_resolution}"):
            for i, (real_images, _) in enumerate(dataloader):
                real_images = real_images.to(device)
                if real_images.shape[2] != current_resolution:
                    real_images = nn.functional.interpolate(real_images, size=(current_resolution, current_resolution), mode='bilinear', align_corners=False)

                d_loss, g_loss = train_step(real_images, generator, discriminator, gen_optimizer, disc_optimizer, criterion, current_stage, 1.0, latent_dim, device) # Alpha is 1.0 for stabilization
                global_step += 1

                if (global_step % config['sample_interval'] == 0):
                    save_samples(generator, latent_dim, current_stage, device=device)

        print(f"Stage {current_stage} ({current_resolution}x{current_resolution}) completed. D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")
        # Optional: Save checkpoint after each stage

    print("Training complete!")
