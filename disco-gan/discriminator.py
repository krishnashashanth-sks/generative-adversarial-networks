import torch.nn as nn

class Discriminator(nn.Module):
  def __init__(self,input_nc,ndf=64):
    super(Discriminator,self).__init__()
    self.main=nn.Sequential(
        nn.Conv2d(input_nc,ndf,kernel_size=4,stride=2,padding=1,bias=False),
        nn.LeakyReLU(0.2,inplace=True),
        self._discriminator_block(ndf,ndf*2),
        self._discriminator_block(ndf*2,ndf*4),
        self._discriminator_block(ndf*4,ndf*8),
        nn.Conv2d(ndf*8,1,kernel_size=4,stride=2,padding=1,bias=False)
    )

  def _discriminator_block(self,in_channels,out_channels,stride=2,use_norm=True):
    layers=[
        nn.Conv2d(in_channels,out_channels,kernel_size=4,stride=2,padding=1,bias=False)
    ]
    if use_norm:
      layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2,True))
    return nn.Sequential(*layers)
    
  def forward(self,input):
    return self.main(input)

print("Discriminator model class defined.")