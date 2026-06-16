import itertools

def get_next_batch(dataloader_iter,dataloader):
  try:
    batch=next(dataloader_iter)
  except StopIteration:
    dataloader_iter=iter(dataloader)
    batch=next(dataloader_iter)
  return batch,dataloader_iter

import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch

def visualize_generated_images(gen_S2T,gen_T2S,real_S_sample,real_T_sample,epoch,device,num_images=5):
  gen_S2T.eval()
  gen_T2S.eval()
  with torch.no_grad():
    real_S_display=real_S_sample[:num_images].to(device)
    real_T_display=real_T_sample[:num_images].to(device)
    fake_T=gen_S2T(real_S_display)
    fake_S=gen_T2S(real_T_display)
    cycled_S=gen_T2S(fake_T)
    cycles_T=gen_S2T(fake_S)
    images_S=torch.cat([real_S_display,fake_T,cycled_S],dim=0)
    images_T=torch.cat([real_T_display,fake_S,cycles_T],dim=0)
    mean=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).to(device)
    std=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(device)
    images_S=images_S*std+mean
    images_T=images_T*std+mean
    images_S=torch.clamp(images_S,0,1)
    images_T=torch.clamp(images_T,0,1)
    grid_S=vutils.make_grid(images_S,nrow=num_images,padding=2,normalize=False)
    grid_T=vutils.make_grid(images_T,nrow=num_images,padding=2,normalize=False)
    fig,axes=plt.subplots(2,1,figsize=(15,6))
    axes[0].imshow(grid_S.cpu().permute(1,2,0))
    axes[0].set_title(f'Epoch {epoch} -Sources (MNIST) to Target Cycles')
    axes[0].axis('off')
    axes[1].imshow(grid_T.cpu().permute(1,2,0))
    axes[0].set_title(f'Epoch {epoch} -Target to Source Cycle')
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()
  gen_S2T.train()
  gen_T2S.train()
print("visualization function defined")