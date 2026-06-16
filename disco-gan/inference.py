import torch

def translate_image(image_tensor, generator, device):
    """Translates a single image using the given generator model.

    Args:
        image_tensor (torch.Tensor): The input image tensor (CxHxW).
        generator (nn.Module): The generator model (e.g., gen_S2T or gen_T2S).
        device (torch.device): The device (cpu or cuda) to run the inference on.

    Returns:
        torch.Tensor: The translated image tensor (CxHxW).
    """
    generator.eval()
    with torch.no_grad():
        # Add a batch dimension, move to device, translate, remove batch dimension
        translated_image = generator(image_tensor.unsqueeze(0).to(device)).squeeze(0)
    generator.train() # Set back to train mode if training continues
    return translated_image.cpu()

# Denormalization utility (from previous visualization cell)
mean_vals = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std_vals = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def denormalize_image(img_tensor):
    """Denormalizes an image tensor for display.

    Args:
        img_tensor (torch.Tensor): A normalized image tensor.

    Returns:
        torch.Tensor: The denormalized and clamped image tensor (pixel values between 0 and 1).
    """
    img = img_tensor * std_vals + mean_vals
    return torch.clamp(img, 0, 1)

print("Inference function `translate_image` defined.")