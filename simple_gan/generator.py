import torch.nn as nn
class Generator(nn.Module):
  def __init__(self,latent_dim,num_channels,img_size):
    super(Generator, self).__init__()
    self.main=nn.Sequential(
        nn.ConvTranspose2d(latent_dim,img_size*8,4,1,0,bias=False),
        nn.BatchNorm2d(img_size*8),
        nn.ReLU(True),
        nn.ConvTranspose2d(img_size*8,img_size*4,4,2,1,bias=False),
        nn.BatchNorm2d(img_size*4),
        nn.ReLU(True),
        nn.ConvTranspose2d(img_size*4,img_size*2,4,2,1,bias=False),
        nn.BatchNorm2d(img_size*2),
        nn.ReLU(True),
        nn.ConvTranspose2d(img_size*2,img_size,4,2,1,bias=False),
        nn.ReLU(True),
        nn.ConvTranspose2d(img_size,num_channels,4,2,1,bias=False),
        nn.Tanh()
    )
  def forward(self,x):
    return self.main(x)
