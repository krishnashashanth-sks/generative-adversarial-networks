from generator import Generator
from discriminator import Discriminator
import torch
from train import train_gan
from inference import generate_image
import matplotlib.pyplot as plt
from utils import get_current_resolution

latent_dim = 100
generator = Generator(latent_dim=latent_dim)

discriminator = Discriminator()


config = {
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'latent_dim': 100,
    'lr_g': 0.001,
    'lr_d': 0.001,
    'lambda_gp': 10,
    'num_stages': 8, # Corresponds to resolutions up to 1024x1024
    'fade_in_steps_per_stage': 200, # Number of gradient steps for fading in new layers (reduced from 1000)
    'stabilization_epochs_per_stage': 2, # Number of epochs for stabilization (reduced from 10)
    'batch_sizes': {
        4: 128, 8: 128, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8, 512: 4, 1024: 2, 'default': 32
    },
    'num_workers': 2, # For DataLoader
    'sample_interval': 50, # How often to save samples during training
    'start_steps_per_stage': {0:0, 1:200, 2:400, 3:600, 4:800, 5:1000, 6:1200, 7:1400} # Approximate global steps when each stage begins
}

train_gan(
    generator=generator,
    discriminator=discriminator,
    latent_dim=config['latent_dim'],
    num_stages=config['num_stages'],
    config=config
)

# --- Inference Example after Training ---
print("\n--- Running Inference Example ---")
# Generate an image at the highest trained resolution (e.g., stage 7 for 1024x1024 if num_stages is 8)
target_stage = config['num_stages'] - 1
# It's crucial to set the generator's stage before inference to ensure it uses the correct layers.
generated_final_image = generate_image(generator, config['latent_dim'], target_stage, config['device'])

plt.figure(figsize=(4, 4))
plt.imshow(generated_final_image.squeeze(0).permute(1, 2, 0))
plt.axis('off')
plt.title(f"Generated Image at Resolution {get_current_resolution(target_stage)}x{get_current_resolution(target_stage)}")
plt.show()