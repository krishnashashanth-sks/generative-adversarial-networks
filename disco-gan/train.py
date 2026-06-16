from tqdm.auto import tqdm
import torch
from losses import adv_loss,identity_loss,cycle_loss
from utils import get_next_batch,visualize_generated_images

def train(num_epochs,source_dataloader,target_dataloader,gen_optimizer,gen_S2T,gen_T2S,disc_T,disc_S,disc_S_optimizer,disc_T_optimizer,lambda_cycle,lambda_identity,device):
    losses = {
        'G_adv': [], 'G_cycle': [], 'G_identity': [],
        'D_S': [], 'D_T': []
    }
    real_S_sample = next(iter(source_dataloader))[0]
    real_T_sample = next(iter(target_dataloader))[0]    
    for epoch in tqdm(range(num_epochs)):
        source_dataloader_iter = iter(source_dataloader)
        target_dataloader_iter = iter(target_dataloader)
        num_batches=min(len(source_dataloader),len(target_dataloader))
        for i in tqdm(range(num_batches)):
            real_S_batch,source_dataloader_iter=get_next_batch(source_dataloader_iter,source_dataloader)
            real_T_batch,target_dataloader_iter=get_next_batch(target_dataloader_iter,target_dataloader)
            real_S=real_S_batch[0].to(device)
            real_T=real_T_batch[0].to(device)
            gen_optimizer.zero_grad()
            # Fixed: loss_identity_T should use identity_T, not identity_S
            # Fixed: identity_S should be calculated before loss_identity_T
            identity_T = gen_S2T(real_S)
            identity_S = gen_T2S(real_T)
            loss_identity_T = identity_loss(identity_T, real_S)
            loss_identity_S = identity_loss(identity_S, real_T)
            loss_identiy=(lambda_identity*loss_identity_S)+(lambda_identity*loss_identity_T)
            fake_T=gen_S2T(real_S)
            pred_fake_T=disc_T(fake_T)
            loss_adv_G_S2T=adv_loss(pred_fake_T,torch.ones_like(pred_fake_T))
            fake_S=gen_T2S(real_T)
            pred_fake_S=disc_S(fake_S)
            loss_adv_G_T2S=adv_loss(pred_fake_S,torch.ones_like(pred_fake_S))
            loss_adv_G=loss_adv_G_S2T+loss_adv_G_T2S
            reconstructed_S=gen_T2S(fake_T)
            loss_cycle_S=cycle_loss(reconstructed_S,real_S)
            reconstructed_T=gen_S2T(fake_S)
            loss_cycle_T=cycle_loss(reconstructed_T,real_T)
            loss_cycle= (lambda_cycle*loss_cycle_S)+(lambda_cycle*loss_cycle_T)
            total_loss_G=loss_adv_G+loss_cycle+loss_identiy
            total_loss_G.backward()
            gen_optimizer.step()
            disc_S_optimizer.zero_grad()
            pred_real_S=disc_S(real_S)
            loss_D_real_S=adv_loss(pred_real_S,torch.ones_like(pred_real_S))
            # Fixed: Define loss_D_fake_S before using it
            pred_fake_S_detached = disc_S(fake_S.detach())
            loss_D_fake_S = adv_loss(pred_fake_S_detached, torch.zeros_like(pred_fake_S_detached))
            loss_D_S=(loss_D_real_S+loss_D_fake_S)*0.5
            loss_D_S.backward()
            disc_S_optimizer.step()
            disc_T_optimizer.zero_grad()
            pred_real_T=disc_T(real_T)
            loss_pred_real_T=adv_loss(pred_real_T,torch.ones_like(pred_real_T))
            disc_T_optimizer.zero_grad()

                # Real loss
            pred_real_T = disc_T(real_T)
            loss_D_real_T = adv_loss(pred_real_T, torch.ones_like(pred_real_T))

                #  Fake loss (detach fake_T to avoid backprop through G_S2T)
            pred_fake_T = disc_T(fake_T.detach())
            loss_D_fake_T = adv_loss(pred_fake_T, torch.zeros_like(pred_fake_T))

                # Total Discriminator T Loss
            loss_D_T = (loss_D_real_T + loss_D_fake_T) * 0.5
            loss_D_T.backward()
            disc_T_optimizer.step()

                # Store losses
            losses['G_adv'].append(loss_adv_G.item())
            losses['G_cycle'].append(loss_cycle.item())
            losses['G_identity'].append(loss_identiy.item()) # Fixed: Typo in variable name
            losses['D_S'].append(loss_D_S.item())
            losses['D_T'].append(loss_D_T.item())

            if i % 100 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{num_batches}]\n"
                        f"  G_adv Loss: {loss_adv_G.item():.4f}, G_cycle Loss: {loss_cycle.item():.4f}, G_identity Loss: {loss_identiy.item():.4f}\n" # Fixed: Typo in variable name
                        f"  D_S Loss: {loss_D_S.item():.4f}, D_T Loss: {loss_D_T.item():.4f}")
        visualize_generated_images(gen_S2T, gen_T2S, real_S_sample, real_T_sample, epoch, device)
    return losses