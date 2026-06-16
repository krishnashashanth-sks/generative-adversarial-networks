import torch
# --- Helper function for mean_w update ---

def update_mean_w(generator, z, ema_beta=0.999, device='cpu'):
    """
    Updates the running average of the W latent space (mean_w) for the truncation trick.
    """
    with torch.no_grad():
        w_styles = generator.mapping_network(z)
        current_batch_mean_w = w_styles.mean(dim=(0, 1)) # Average across batch and num_ws

        if generator.mean_w.cpu().numpy().all() == 0: # Check if mean_w is still zeros (initial state)
            generator.mean_w.copy_(current_batch_mean_w) # For the very first update, just assign
        else:
            generator.mean_w.mul_(ema_beta).add_(current_batch_mean_w * (1 - ema_beta))