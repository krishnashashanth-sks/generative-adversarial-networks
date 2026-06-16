import torch.nn as nn
import torch

class Generator(nn.Module):
  def __init__(self,noise_dim,latent_dim_categorical,latent_dim_continuous,img_dim):
    super().__init__()
    self.main=nn.Sequential(
        nn.Linear(noise_dim+latent_dim_categorical+latent_dim_continuous,256),
        nn.ReLU(),
        nn.Linear(256,img_dim),
        nn.Tanh()
    )
  def forward(self,x,c_cat,c_cont):
    input_vec=torch.cat([x,c_cat,c_cont],dim=1)
    return self.main(input_vec)