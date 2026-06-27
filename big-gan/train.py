import torch
from utils import orthogonal_regularization

def train(num_epochs,dataloader,generator,discriminator,optimizer_D,optimizer_G,discriminator_loss,generator_loss,latent_dim,batch_size,n_critic,num_classes,device):
        
    print("Starting Training Loop...")

    for epoch in range(num_epochs):
        for i, (real_images, labels) in enumerate(dataloader):
            real_images = real_images.to(device)
            labels = labels.to(device)
            current_batch_size = real_images.size(0)

            # Clear CUDA cache before D update
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            optimizer_D.zero_grad()

            z = torch.randn(current_batch_size, latent_dim, device=device)
            fake_labels = torch.randint(0, num_classes, (current_batch_size,), device=device)

            fake_images = generator(z, fake_labels).detach()

            d_real_output = discriminator(real_images, labels)
            d_fake_output = discriminator(fake_images, fake_labels)

            loss_d = discriminator_loss(d_real_output, d_fake_output)

            loss_d.backward()
            optimizer_D.step()

            if i % n_critic == 0:
                # Clear CUDA cache before G update
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                optimizer_G.zero_grad()

                z = torch.randn(current_batch_size, latent_dim, device=device)
                fake_labels = torch.randint(0, num_classes, (current_batch_size,), device=device)

                fake_images = generator(z, fake_labels)

                d_fake_output = discriminator(fake_images, fake_labels)

                loss_g = generator_loss(d_fake_output)

                ortho_reg_g = orthogonal_regularization(generator, strength=1e-4, num_classes=num_classes)
                loss_g += ortho_reg_g

                loss_g.backward()
                optimizer_G.step()

            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}] Batch [{i}/{len(dataloader)}] \
                        Loss D: {loss_d.item():.4f} Loss G: {loss_g.item():.4f}')
    print("Training loop finished.")