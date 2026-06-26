import torchvision.utils as vutils
import torch

def get_current_resolution(stage):
  return 4 * (2 ** stage)

def save_samples(generator, latent_dim, current_stage, num_samples=16, path='./generated_samples/', device='cpu'):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        # Ensure the generator is at the correct stage for saving samples
        # The alpha should be 1.0 to get a fully formed image at that resolution
        generator.update_stage(current_stage, 1.0)
        generated_images = generator(z).cpu()
    generator.train()

    # Denormalize images from [-1, 1] to [0, 1]
    generated_images = (generated_images + 1) / 2

    # Make sure the directory exists
    import os
    os.makedirs(path, exist_ok=True)

    # Save a grid of images
    vutils.save_image(generated_images, os.path.join(path, f'stage_{current_stage}_res_{get_current_resolution(current_stage)}x{get_current_resolution(current_stage)}.png'), nrow=4)
    print(f"Saved {num_samples} samples at stage {current_stage} ({get_current_resolution(current_stage)}x{get_current_resolution(current_stage)}) to {path}")
