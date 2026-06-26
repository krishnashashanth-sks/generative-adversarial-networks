import torch

def generate_image(generator, latent_dim, target_stage, device='cpu'):
    generator.eval() # Set generator to evaluation mode
    with torch.no_grad():
        z = torch.randn(1, latent_dim).to(device)
        # Set generator to the target stage with alpha=1.0 for a fully formed image
        generator.update_stage(target_stage, 1.0)
        generated_image = generator(z).cpu()
    generator.train() # Set generator back to training mode

    # Denormalize the image from [-1, 1] to [0, 1]
    generated_image = (generated_image + 1) / 2
    return generated_image