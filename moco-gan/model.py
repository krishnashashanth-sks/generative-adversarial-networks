from layers import *
import torch.nn as nn
from generator import MocoGAN_Generator

class MocoGAN(nn.Module):
  def __init__(self,content_dim=512,motion_dim=128,sequence_length=5,ndf=64,ngf=64):
    super(MocoGAN,self).__init__()
    self.generator=MocoGAN_Generator(
        content_dim=content_dim,
        motion_dim=motion_dim,
        sequence_length=sequence_length
    )
    self.video_discriminator=VideoDiscriminator(3,ndf,sequence_length)
    self.content_discriminator=ContentDiscriminator(content_dim)
    self.motion_discriminator=MotionDiscriminator(motion_dim)
  def forward(self,initial_frame,motion_sequence):
    return self.generator(initial_frame,motion_sequence)