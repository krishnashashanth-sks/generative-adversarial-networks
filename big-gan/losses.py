import torch
import torch.nn.functional as F

def discriminator_loss(d_real, d_fake):
    loss_real = torch.mean(F.relu(1. - d_real))
    loss_fake = torch.mean(F.relu(1. + d_fake))
    return loss_real + loss_fake

def generator_loss(d_fake):
    return -torch.mean(d_fake)