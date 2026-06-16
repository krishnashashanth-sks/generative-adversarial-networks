import torch
from losses import d_logistic_loss,d_r1_loss,g_logistic_loss
from utils import update_mean_w
def train_loop(generator, discriminator, optimizer_G, optimizer_D, data_loader, device,
               num_epochs, r1_gamma=10.0, lazy_r1_interval=16, ema_beta_w=0.999):
    print(f"Starting training on {device} for {num_epochs} epochs...")

    # Placeholder for tracking metrics
    g_losses = []
    d_losses = []
    r1_penalties = []

    for epoch in range(num_epochs):
        for idx, real_videos in enumerate(data_loader):
            real_videos = real_videos.to(device)
            batch_size = real_videos.size(0)

            # --- Update Discriminator ---
            optimizer_D.zero_grad()

            # 1. Train with real videos
            real_pred = discriminator(real_videos)

            # 2. Generate fake videos
            z = torch.randn(batch_size, generator.z_dim, device=device)
            fake_videos = generator(z).detach() # Detach to prevent G from updating

            # 3. Train with fake videos
            fake_pred = discriminator(fake_videos)

            # Calculate discriminator logistic loss
            d_loss = d_logistic_loss(real_pred, fake_pred)

            # 4. R1 Regularization (applied periodically)
            if idx % lazy_r1_interval == 0:
                real_videos.requires_grad = True # Enable grad for real_videos for R1
                real_pred_r1 = discriminator(real_videos)
                r1_penalty = d_r1_loss(real_pred_r1, real_videos) * r1_gamma
                d_loss = d_loss + r1_penalty
                r1_penalties.append(r1_penalty.item())
            else:
                r1_penalties.append(0.0)

            d_loss.backward()
            optimizer_D.step()
            d_losses.append(d_loss.item())

            # --- Update Generator ---
            optimizer_G.zero_grad()

            # Generate new fake videos for generator update
            z = torch.randn(batch_size, generator.z_dim, device=device)
            fake_videos = generator(z) # Do not detach

            # Get discriminator prediction for fake videos
            fake_pred = discriminator(fake_videos)

            # Calculate generator logistic loss
            g_loss = g_logistic_loss(fake_pred)

            g_loss.backward()
            optimizer_G.step()
            g_losses.append(g_loss.item())

            # --- Update mean_w for Truncation Trick ---
            update_mean_w(generator, z, ema_beta=ema_beta_w, device=device)

            if idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{idx}], "
                      f"D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")

        print(f"Epoch {epoch+1} finished.")

    print("Training complete!")
    return g_losses, d_losses, r1_penalties