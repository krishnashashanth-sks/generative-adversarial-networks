import torch.nn as nn
class Discriminator(nn.Module):
  def __init__(self,num_channels,img_size):
    super(Discriminator,self).__init__()
    self.main=nn.Sequential(
        # No BatchNorm for the first layer as per DCGAN guidelines
        nn.Conv2d(num_channels,img_size,4,2,1,bias=False),
        nn.LeakyReLU(0.2,inplace=True),
        nn.Conv2d(img_size,img_size*2,4,2,1,bias=False),
        nn.BatchNorm2d(img_size*2),
        nn.LeakyReLU(0.2,inplace=True),
        nn.Conv2d(img_size*2,img_size*4,4,2,1,bias=False),
        nn.BatchNorm2d(img_size*4),
        nn.LeakyReLU(0.2,inplace=True),
        nn.Conv2d(img_size*4,img_size*8,4,2,1,bias=False),
        nn.BatchNorm2d(img_size*8),
        nn.LeakyReLU(0.2,inplace=True),
        nn.Conv2d(img_size*8,1,4,1,0,bias=False),
        nn.Sigmoid()
    )
  def forward(self,x):
    out=self.main(x)
    return out.view(-1,1).squeeze(1)
