import torch

def train_gan(
    gen, disc, dataloader, optimizer_G, optimizer_D, criterion, num_epochs, z_dim, device, batch_size
):
    print("Starting Training Loop...")

    for epoch in range(num_epochs):
        for i, (real_music, real_bar_latents) in enumerate(dataloader):
            real_music = real_music.to(device)
            real_bar_latents = real_bar_latents.to(device)

            # --------------------- #
            #  Train Discriminator  #
            # --------------------- #
            optimizer_D.zero_grad()

            # 1. Train D with real data
            real_labels = torch.ones(real_music.size(0), 1).to(device)
            output_real = disc(real_music, real_bar_latents)
            d_loss_real = criterion(output_real, real_labels)

            # 2. Train D with fake data
            z = torch.randn(real_music.size(0), z_dim).to(device)
            # Modify Generator to return bar_latents along with generated_music
            # For now, we'll keep generating bar_latents separately from gen.bar_gen(z)
            # but ideally, the Generator's forward pass would return both.
            # This requires a change in the Generator class if we want true consistency
            # For this example, we'll re-run bar_gen for fake_bar_latents.
            # Let's consider modifying the Generator's forward to return both generated_music and its bar_latents.
            # However, for now, to keep the Generator's interface simple, we'll re-compute them.
            generated_music_temp = gen(z) # This generates music
            fake_bar_latents = gen.bar_gen(z) # This generates bar latents from the same noise

            fake_labels = torch.zeros(real_music.size(0), 1).to(device)
            output_fake = disc(generated_music_temp.detach(), fake_bar_latents.detach())
            d_loss_fake = criterion(output_fake, fake_labels)

            # Combine losses and update Discriminator
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()

            # ----------------- #
            #  Train Generator  #
            # ----------------- #
            optimizer_G.zero_grad()

            # Generate fake music again for Generator training
            z = torch.randn(real_music.size(0), z_dim).to(device)
            generated_music = gen(z)
            fake_bar_latents = gen.bar_gen(z)

            # Generator wants Discriminator to classify fakes as real (label 1)
            g_labels = torch.ones(real_music.size(0), 1).to(device)
            output_g = disc(generated_music, fake_bar_latents)
            g_loss = criterion(output_g, g_labels)

            # Update Generator
            g_loss.backward()
            optimizer_G.step()

            # ----------------- #
            #  Print Progress   #
            # ----------------- #
            if (i + 1) % 10 == 0: # Print every 10 batches
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(dataloader)}]\t"
                    f"D_Loss: {d_loss.item():.4f}, G_Loss: {g_loss.item():.4f}"
                )

    print("Training complete.")
