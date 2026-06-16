import torch.nn as nn
import torch
class Generator(nn.Module):
  def __init__(self,input_nc,output_nc,ngf=64):
    super(Generator,self).__init__()
    # Encoder (7 downsampling blocks)
    self.down1=nn.Sequential(
        nn.Conv2d(input_nc,ngf,kernel_size=4,stride=2,padding=1,bias=False),
        nn.LeakyReLU(0.2,True)
    ) # 224 -> 112, out_channels: ngf
    self.down2=self._downsample_block(ngf,ngf*2) # 112 -> 56, out_channels: ngf*2
    self.down3=self._downsample_block(ngf*2,ngf*4) # 56 -> 28, out_channels: ngf*4
    self.down4=self._downsample_block(ngf*4,ngf*8) # 28 -> 14, out_channels: ngf*8
    self.down5=self._downsample_block(ngf*8,ngf*8) # 14 -> 7, out_channels: ngf*8
    self.down6=self._downsample_block(ngf*8,ngf*8) # 7 -> 3, out_channels: ngf*8
    self.down7=self._downsample_block(ngf*8,ngf*8) # 3 -> 1, out_channels: ngf*8 (bottleneck)

    # Decoder (7 upsampling blocks + final output)
    # up1 processes the bottleneck d7 (1x1)
    self.up1=self._upsampling_block(ngf*8,ngf*8, output_padding=1) # Input from d7 (ngf*8), upsamples 1x1 to 3x3, output ngf*8
    self.up2=self._upsampling_block(ngf*8*2,ngf*8, output_padding=1) # Input from (u1_out + d6) = (ngf*8 + ngf*8) = ngf*16. Upsamples 3x3 to 7x7, output ngf*8
    self.up3=self._upsampling_block(ngf*8*2,ngf*8) # Input from (u2_out + d5) = (ngf*8 + ngf*8) = ngf*16. Upsamples 7x7 to 14x14, output ngf*8
    self.up4=self._upsampling_block(ngf*8*2,ngf*4) # Input from (u3_out + d4) = (ngf*8 + ngf*8) = ngf*16. Upsamples 14x14 to 28x28, output ngf*4
    self.up5=self._upsampling_block(ngf*4*2,ngf*2) # Input from (u4_out + d3) = (ngf*4 + ngf*4) = ngf*8. Upsamples 28x28 to 56x56, output ngf*2
    self.up6=self._upsampling_block(ngf*2*2,ngf,use_dropout=True) # Input from (u5_out + d2) = (ngf*2 + ngf*2) = ngf*4. Upsamples 56x56 to 112x112, output ngf
    # Final output layer (upsamples 112x112 to 224x224)
    self.final_output_layer = nn.Sequential(
        nn.ConvTranspose2d(ngf*2,output_nc,kernel_size=4,stride=2,padding=1), # Input from (u6_out + d1) = (ngf + ngf) = ngf*2
        nn.Tanh()
    )

  def _downsample_block(self,in_channels,out_channels,use_norm=True):
    layers=[
        nn.Conv2d(in_channels,out_channels,kernel_size=4,stride=2,padding=1,bias=False)
    ]
    if use_norm:
      layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2,True))
    return nn.Sequential(*layers)

  def _upsampling_block(self,in_channels,out_channels,use_dropout=False, output_padding=0):
    layers=[
        nn.ConvTranspose2d(in_channels,out_channels,kernel_size=4,stride=2,padding=1,output_padding=output_padding,bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
    ]
    if use_dropout:
      layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)

  def forward(self,x):
    # Encoder
    d1=self.down1(x)
    d2=self.down2(d1)
    d3=self.down3(d2)
    d4=self.down4(d3)
    d5=self.down5(d4)
    d6=self.down6(d5)
    d7=self.down7(d6) # Bottleneck

    # Decoder
    u1=self.up1(d7)
    u2=self.up2(torch.cat([u1,d6],1))
    u3=self.up3(torch.cat([u2,d5],1))
    u4=self.up4(torch.cat([u3,d4],1))
    u5=self.up5(torch.cat([u4,d3],1))
    u6=self.up6(torch.cat([u5,d2],1))

    return self.final_output_layer(torch.cat([u6,d1],1))

print("Generator model class defined.")