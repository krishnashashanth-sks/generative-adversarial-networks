import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, num_channels, ngf=64):
        """
        Generator Network for DCGAN.
        
        Parameters:
            latent_dim (int): Dimension of the latent vector z (e.g., 100)
            num_channels (int): Number of channels in the output image (e.g., 1 for MNIST)
            ngf (int): Base feature map size multiplier (e.g., 64)
        """
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input is latent_dim x 1 x 1 -> State: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            # State: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            # State: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            # State: (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.ReLU(True),
            
            # Final Layer -> Output State: (num_channels) x 64 x 64
            nn.ConvTranspose2d(ngf, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)
