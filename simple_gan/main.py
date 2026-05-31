from generator import Generator
from discriminator import Discriminator
from utils import weights_init
import torch.nn as nn
import torchvision.utils as vutils
from dataset import train_loader, test_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nz=100
os.makedirs("../data", exist_ok=True)
os.makedirs("../outputs", exist_ok=True)

netG=Generator(100,1,64)
netG.apply(weights_init)

netD=Discriminator(1,64)
netD.apply(weights_init)

criterion=nn.BCELoss()

optimizerG=torch.optim.Adam(netG.parameters(),lr=0.0002,betas=(0.5,0.999))
optimizerD=torch.optim.Adam(netD.parameters(),lr=0.0002,betas=(0.5,0.999))

# Move models to the GPU
netG.to(device)
netD.to(device)

fixed_noise=torch.randn(64, nz, 1, 1, device=device)

G_losses, D_losses, image_list = train(
        netG=netG,
        netD=netD,
        train_loader=train_loader,
        optimizerG=optimizerG,
        optimizerD=optimizerD,
        criterion=criterion,
        device=device,
        epochs=epochs,
        nz=nz,
        fixed_noise=fixed_noise
    )
