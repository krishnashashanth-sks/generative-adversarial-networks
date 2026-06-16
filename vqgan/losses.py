import torch.nn as nn
from torchvision import models

recon_loss_fn = nn.L1Loss()
adv_loss_fn = nn.BCEWithLogitsLoss()

print("Reconstruction loss (L1Loss) and Adversarial loss (BCEWithLogitsLoss) initialized.")


class VGGPerceptualLoss(nn.Module):
  def __init__(self, requires_grad=False):
    super(VGGPerceptualLoss, self).__init__()
    vgg = models.vgg16(pretrained=True).features
    self.slice1 = nn.Sequential()
    for i, layer in enumerate(vgg):
      self.slice1.add_module(str(i), layer)
      if i == 4:
        break
    if not requires_grad:
      for param in self.parameters():
        param.requires_grad = False
    self.criterion = nn.MSELoss()

    # ImageNet normalization parameters
    self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

  def forward(self, fake_images, real_images):
    # Normalize images from [-1, 1] to [0, 1]
    fake_images = (fake_images + 1) / 2
    real_images = (real_images + 1) / 2

    # Further normalize using ImageNet stats
    fake_images = self.normalize(fake_images)
    real_images = self.normalize(real_images)

    # Extract features
    fake_features = self.slice1(fake_images)
    real_features = self.slice1(real_images)

    # Compute MSE loss between features
    return self.criterion(fake_features, real_features)