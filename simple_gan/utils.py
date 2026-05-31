import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.utils as vutils

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
def show_images(images):
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    # Unnormalize images from [-1, 1] to [0, 1] for plotting
    # make_grid normalizes by default if normalize=True, but we want control
    plt.imshow(vutils.make_grid(images.cpu(), padding=2, normalize=True).permute(1, 2, 0))
    plt.show()
