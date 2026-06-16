from tqdm.auto import tqdm
import torch

def train(epochs,dataloader,gen,disc,criterion_gan,criterion_categorical,criterion_continuous,optimizer_g,optimizer_d,lambda_mi,latent_dim_categorical,latent_dim_continuous,noise_dim,device):
    print("Starting training loop...")
    for epoch in tqdm(range(epochs), desc="Epochs"):
        for i, (real_imgs, _) in tqdm(enumerate(dataloader), desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            real_imgs = real_imgs.to(device)

            # Get current batch size, as the last batch might be smaller
            current_batch_size = real_imgs.size(0)
            real_labels = torch.ones(current_batch_size, 1).to(device)
            fake_labels = torch.zeros(current_batch_size, 1).to(device)

            # --- Train Discriminator ---
            optimizer_d.zero_grad()

            # 1. Train with real images
            D_output_real, _, _ = disc(real_imgs)
            D_loss_real = criterion_gan(D_output_real, real_labels)

            # 2. Train with fake images
            # Generate noise and latent codes
            z = torch.randn(current_batch_size, noise_dim, device=device)
            c_cat_sample = torch.randint(0, latent_dim_categorical, (current_batch_size,), device=device)
            c_cat_one_hot = torch.nn.functional.one_hot(c_cat_sample, num_classes=latent_dim_categorical).float().to(device)
            c_cont_sample = torch.rand(current_batch_size, latent_dim_continuous, device=device) * 2 - 1 # Map to [-1, 1]

            fake_imgs = gen(z, c_cat_one_hot, c_cont_sample)

            # Detach fake images to prevent gradients from flowing back to the generator during D's update
            D_output_fake, pred_c_cat, pred_c_cont = disc(fake_imgs.detach())
            D_loss_fake = criterion_gan(D_output_fake, fake_labels)

            # 3. Auxiliary Loss for Discriminator (Q-network part on fake images)
            # Q-network aims to predict the latent codes from generated images.
            Q_loss_cat = criterion_categorical(pred_c_cat, c_cat_sample)
            Q_loss_cont = criterion_continuous(pred_c_cont, c_cont_sample)

            # Total Discriminator Loss
            # D tries to minimize its loss for real/fake, and Q tries to correctly predict c
            D_loss = D_loss_real + D_loss_fake + lambda_mi * (Q_loss_cat + Q_loss_cont)

            D_loss.backward()
            optimizer_d.step()

            # --- Train Generator ---
            optimizer_g.zero_grad()

            # Generate new fake images with new noise and latent codes
            z = torch.randn(current_batch_size, noise_dim, device=device)
            c_cat_sample = torch.randint(0, latent_dim_categorical, (current_batch_size,), device=device)
            c_cat_one_hot = torch.nn.functional.one_hot(c_cat_sample, num_classes=latent_dim_categorical).float().to(device)
            c_cont_sample = torch.rand(current_batch_size, latent_dim_continuous, device=device) * 2 - 1

            fake_imgs = gen(z, c_cat_one_hot, c_cont_sample)

            # Discriminator's output and Q's predictions for the newly generated fake images
            G_output, pred_c_cat_g, pred_c_cont_g = disc(fake_imgs)

            # 1. Adversarial Loss for Generator: G wants D to classify fakes as real
            G_loss_gan = criterion_gan(G_output, real_labels)

            # 2. Mutual Information Loss for Generator: G wants to make c predictable by Q
            G_loss_mi_cat = criterion_categorical(pred_c_cat_g, c_cat_sample)
            G_loss_mi_cont = criterion_continuous(pred_c_cont_g, c_cont_sample)

            # Total Generator Loss
            G_loss = G_loss_gan + lambda_mi * (G_loss_mi_cat + G_loss_mi_cont)

            G_loss.backward()
            optimizer_g.step()

            # --- Logging and printing (optional) ---
            if i % 100 == 0: # Print every 100 batches
                print(f"Epoch [{epoch}/{epochs}], Batch [{i}/{len(dataloader)}]\t"
                    f"D Loss: {D_loss.item():.4f}\tG Loss: {G_loss.item():.4f}\t"
                    f"Q Cat Loss: {Q_loss_cat.item():.4f}\tQ Cont Loss: {Q_loss_cont.item():.4f}")
