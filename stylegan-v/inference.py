import torch

def inference(generator, z=None, device='cuda'):
    """
    Generates a video using the trained generator model.

    Args:
        generator (nn.Module): The trained generator model.
        z (torch.Tensor, optional): The input latent vector. If None, a random vector is generated.
        device (str): The device to perform inference on (e.g., 'cuda' or 'cpu').

    Returns:
        torch.Tensor: The generated video tensor (batch_size, channels, frames, height, width).
    """
    generator.eval() # Set generator to evaluation mode

    with torch.no_grad(): # Disable gradient computation
        if z is None:
            # If z is not provided, generate a random one
            # Assuming batch_size=1 for a single inference, and z_dim from generator
            z = torch.randn(1, generator.z_dim, device=device)
        else:
            z = z.to(device)

        generated_video = generator(z)

    generator.train() # Set generator back to training mode (if it was in train mode before)
    return generated_video