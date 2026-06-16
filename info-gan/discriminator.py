import torch.nn as nn 

class Discriminator(nn.Module):
  def __init__(self,img_dim,latent_dim_categorical,latent_dim_continuous):
    super().__init__()
    self.features_extractor=nn.Sequential(
        nn.Linear(img_dim,256),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.3)
    )
    self.discriminator_head=nn.Sequential(
        nn.Linear(256,1),
        nn.Sigmoid() # Added Sigmoid activation
    )
    self.q_categorical_head=nn.Sequential(
        nn.Linear(256,latent_dim_categorical)
    )
    self.q_continuous_head=nn.Sequential(
        nn.Linear(256,latent_dim_continuous)
    )
  def forward(self,img):
    # Flatten the image tensor before passing it to the linear layers
    img_flat = img.view(img.size(0), -1)
    features=self.features_extractor(img_flat)
    validity=self.discriminator_head(features)
    pred_c_cat=self.q_categorical_head(features)
    pred_c_cont=self.q_continuous_head(features)
    return validity,pred_c_cat,pred_c_cont