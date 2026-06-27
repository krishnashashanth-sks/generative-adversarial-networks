import torch.nn as nn
from layers import *

class MocoGAN_Generator(nn.Module):
  def __init__(self,content_dim=512,motion_dim=128,sequence_length=5):
    super(MocoGAN_Generator,self).__init__()
    self.content_encoder=ContentEncoder(in_channels=3, content_dim=content_dim)
    self.motion_encoder=MotionEncoder(motion_dim=motion_dim, sequence_length=sequence_length)
    self.video_decoder=VideoDecoder(content_dim=content_dim, motion_dim=motion_dim, sequence_length=sequence_length)
  def forward(self,initial_frame,motion_sequence):
    content_vector=self.content_encoder(initial_frame).squeeze(-1).squeeze(-1)
    motion_vector=self.motion_encoder(motion_sequence)
    return self.video_decoder(content_vector,motion_vector)