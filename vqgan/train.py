from tqdm.auto import tqdm

def train(num_epochs,train_loader,vqgan_model,recon_loss_fn,perceptual_loss_fn,adv_loss_fn,optimizer_g,optimizer_d)
    g_losses=[]
    d_losses=[]
    print(f"Starting training for {num_epochs} epochs.")
    for epoch in tqdm(range(num_epochs)):
    for batch_idx,(data,_) in tqdm(enumerate(train_loader)):
        real_images=data.to(device)
        vqgan_model.discriminator.train(False)
        vqgan_model.train(True)
        optimizer_g.zero_grad()
        reconstructions,quant_loss,perplexity=vqgan_model(real_images)
        recon_loss=recon_loss_fn(reconstructions,real_images)*lambda_recon
        perceptual_loss=perceptual_loss_fn(reconstructions,real_images)*lambda_perceptual
        discriminator_output_fake=vqgan_model.discriminator(reconstructions)
        gan_loss=adv_loss_fn(discriminator_output_fake,torch.ones_like(discriminator_output_fake))*lambda_gan
        g_loss=recon_loss=perceptual_loss+quant_loss+gan_loss
        g_loss.backward()
        optimizer_g.step()
        g_losses.append(g_loss.item())
        vqgan_model.eval()
        vqgan_model.discriminator.train(True)
        optimizer_d.zero_grad()

        discriminator_output_real = vqgan_model.discriminator(real_images)
        d_loss_real = adv_loss_fn(discriminator_output_real, torch.ones_like(discriminator_output_real))

        discriminator_output_fake_detached = vqgan_model.discriminator(reconstructions.detach()) # Detach reconstructions
        d_loss_fake = adv_loss_fn(discriminator_output_fake_detached, torch.zeros_like(discriminator_output_fake_detached))

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_d.step()
        d_losses.append(d_loss.item())
        if batch_idx % 100 == 0: # Log every 100 batches
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], "
                    f"G Loss: {g_losses[-1]:.4f}, D Loss: {d_losses[-1]:.4f}")
    return g_losses,d_losses