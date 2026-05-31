import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, num_channels=1, ndf=64):
        """
        Discriminator Network for DCGAN.
        
        Parameters:
            num_channels (int): Number of channels in the input image (e.g., 1 for MNIST)
            ndf (int): Base feature map size multiplier (e.g., 64)
        """
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input State: (num_channels) x 64 x 64 -> State: (ndf) x 32 x 32
            # No BatchNorm for the first layer as per DCGAN guidelines
            nn.Conv2d(num_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State: (ndf*2) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State: (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State: (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Final Layer Output State: 1 x 1 x 1 (Probability scalar)
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Flatten the output shape from (batch_size, 1, 1, 1) directly to (batch_size,)
        return self.main(x).view(-1)
