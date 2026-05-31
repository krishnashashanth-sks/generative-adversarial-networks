import torch
import torch.nn as nn
import torchvision.utils as vutils
from tqdm import tqdm

def train(netG, netD, train_loader, optimizerG, optimizerD, criterion, device, epochs, nz, fixed_noise):
    """
    Executes the training loop for a Simple GAN.
    
    Parameters:
        netG, netD: The Generator and Discriminator models.
        train_loader: PyTorch DataLoader for the dataset.
        optimizerG, optimizerD: Optimizers for both networks.
        criterion: The loss function (e.g., nn.BCELoss()).
        device: torch.device ('cuda' or 'cpu').
        epocs: Total number of training epochs.
        nz: Size of the latent z vector.
        fixed_noise: Static noise tensor used for tracking visual progress.
    
    Returns:
        G_losses, D_losses, image_list
    """
    G_losses = []
    D_losses = []
    image_list = []
    iters = 0
    
    print("Starting Training Loop...")
    
    for epoch in tqdm(range(epochs), desc="Epochs"):
        for i, data in enumerate(tqdm(train_loader, desc=f"Batch (Epoch {epoch})", leave=False), 0):
            
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            netD.zero_grad()
            real_cpu = data.to(device)
            b_size = real_cpu.size(0)
            label_real = torch.full((b_size,), 1.0, dtype=torch.float, device=device)
            
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label_real)
            errD_real.backward()
            D_x = output.mean().item()
            
            # Generate fake batch through G
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            label_fake = torch.full((b_size,), 0.0, dtype=torch.float, device=device)
            fake = netG(noise)
            
            # Forward pass fake batch through D (detached from G)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label_fake)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label_real.fill_(1.0)  # Fake labels are treated as real for generator cost
            output = netD(fake).view(-1)
            errG = criterion(output, label_real)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            
            # Log metrics every 100 batches
            if i % 100 == 0:
                print(f'\nEpoch [{epoch+1}/{epochs}] Batch [{i}/{len(train_loader)}]\t'
                      f'Loss_D: {errD.item():.4f}\tLoss_G: {errG.item():.4f}\t'
                      f'D(x): {D_x:.4f}\tD(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')
                G_losses.append(errG.item())
                D_losses.append(errD.item())

            # Periodically capture visual progress grids
            if (iters % 500 == 0) or ((epoch == epochs-1) and (i == len(train_loader)-1)):
                with torch.no_grad():
                    fake_images = netG(fixed_noise).detach().cpu()
                grid = vutils.make_grid(fake_images, padding=2, normalize=True)
                image_list.append(grid)
                
                # Saves the grid image directly to your outputs directory
                vutils.save_image(grid, f"../outputs/iter_{iters}_epoch_{epoch}.png")
                
            iters += 1
            
    print("Completed Training")
    return G_losses, D_losses, image_list
