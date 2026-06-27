import torch
import torch.nn as nn

# Helper function for convolutional block
def conv_block(in_channels, out_channels, kernel_size, stride, padding, normalize=True, activation=True):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
    if normalize:
        layers.append(nn.InstanceNorm2d(out_channels))
    if activation:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)

# Helper function for deconvolutional block
def deconv_block(in_channels, out_channels, kernel_size, stride, padding, output_padding=0, normalize=True, activation=True):
    layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=False)]
    if normalize:
        layers.append(nn.InstanceNorm2d(out_channels))
    if activation:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)

class ContentEncoder(nn.Module):
  def __init__(self,in_channels=3,ngf=64,content_dim=512):
    super(ContentEncoder,self).__init__()
    self.encoder=nn.Sequential(
        conv_block(in_channels,ngf,kernel_size=4,stride=2,padding=1,normalize=False),
        conv_block(ngf,ngf*2,kernel_size=4,stride=2,padding=1),
        conv_block(ngf*2,ngf*4,kernel_size=4,stride=2,padding=1),
        conv_block(ngf*4,ngf*8,kernel_size=4,stride=2,padding=1),
        conv_block(ngf*8,ngf*8,kernel_size=4,stride=2,padding=1),
        conv_block(ngf*8,content_dim,kernel_size=4,stride=1,padding=0,normalize=False,activation=False)
    )
  def forward(self,x):
    return self.encoder(x)
  
class MotionEncoder(nn.Module):
  def __init__(self,in_channels=2,ngf=64,motion_dim=128,sequence_length=5):
    super(MotionEncoder,self).__init__()
    self.sequence_length = sequence_length
    self.spatial_encoder=nn.Sequential(
        conv_block(in_channels, ngf, kernel_size=4, stride=2, padding=1, normalize=False), # Corrected stride=2 and padding=1 for initial downsampling
        conv_block(ngf, ngf*2, kernel_size=4, stride=2, padding=1), # Corrected in_channels from ngf*2 to ngf
        conv_block(ngf*2, ngf*4, kernel_size=4, stride=2, padding=1),
        conv_block(ngf*4, ngf*8, kernel_size=4, stride=2, padding=1),
        conv_block(ngf*8, ngf*8, kernel_size=4, stride=2, padding=1),
        conv_block(ngf*8, motion_dim, kernel_size=4, stride=1, padding=0, normalize=False, activation=False),
    )
    self.gru=nn.GRU(input_size=motion_dim,hidden_size=motion_dim,batch_first=False)
  def forward(self,x):
    batch_size,seq_len,_,H,W=x.shape
    spatial_features=[]
    for t in range(seq_len):
      feature=self.spatial_encoder(x[:,t,:,:,:])
      spatial_features.append(feature.squeeze(-1).squeeze(-1))
    spatial_features=torch.stack(spatial_features,dim=0)
    _,motion_vector=self.gru(spatial_features)
    return motion_vector.squeeze(0)

class VideoDecoder(nn.Module):
  def __init__(self,content_dim=512,motion_dim=128,ngf=64,out_channels=3,sequence_length=5):
    super(VideoDecoder,self).__init__()
    self.sequence_length=sequence_length
    self.content_dim=content_dim
    self.motion_dim=motion_dim
    self.init_size=4
    self.init_channels=ngf*8
    self.rnn_input_dim=motion_dim
    self.rnn_hidden_dim=content_dim
    self.gru_cell=nn.GRUCell(self.rnn_input_dim,self.rnn_hidden_dim) # Corrected from nn.GRU to nn.GRUCell
    self.content_project=nn.Linear(content_dim,self.rnn_hidden_dim)
    self.decoder=nn.Sequential(
        deconv_block(self.rnn_hidden_dim,ngf*8,kernel_size=4,stride=1,padding=0),
        deconv_block(ngf*8,ngf*8,kernel_size=4,stride=2,padding=1),
        deconv_block(ngf*8,ngf*4,kernel_size=4,stride=2,padding=1),
        deconv_block(ngf*4,ngf*2,kernel_size=4,stride=2,padding=1),
        deconv_block(ngf*2,ngf,kernel_size=4,stride=2,padding=1),
        deconv_block(ngf,out_channels,kernel_size=4,stride=2,padding=1,normalize=False,activation=False),
        nn.Tanh()
    )
  def forward(self,content_vector,motion_vector):
    batch_size=content_vector.size(0)
    h_t=self.content_project(content_vector) # Initial hidden state for the GRUCell
    generated_frames=[]
    for _ in range(self.sequence_length):
      # motion_vector needs to be (batch, input_size) for GRUCell
      # h_t needs to be (batch, hidden_size) for GRUCell
      h_t=self.gru_cell(motion_vector,h_t)
      frame_features=h_t.view(batch_size,self.rnn_hidden_dim,1,1)
      frame=self.decoder(frame_features)
      generated_frames.append(frame)
    return torch.stack(generated_frames,dim=1)
  
def disc_conv_block(in_channels,out_channels,kernel_size,stride,padding,normalize=True,activation=True):
  layers=[nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False)]
  if normalize:
    layers.append(nn.InstanceNorm2d(out_channels))
  if activation:
    layers.append(nn.LeakyReLU(0.2,inplace=True))
  return nn.Sequential(*layers)

class VideoDiscriminator(nn.Module):
  def __init__(self,in_channels=3,ndf=64,sequence_length=5):
    super(VideoDiscriminator,self).__init__()
    self.sequence_length=sequence_length
    self.main=nn.Sequential(
        nn.Conv3d(in_channels,ndf,kernel_size=(4,4,4),stride=(1,2,2),padding=(1,1,1),bias=False),
        nn.LeakyReLU(0.2,inplace=True),
        nn.Conv3d(ndf,ndf*2,kernel_size=(4,4,4),stride=(1,2,2),padding=(1,1,1),bias=False),
        nn.BatchNorm3d(ndf*2),
        nn.LeakyReLU(0.2,inplace=True),
        nn.Conv3d(ndf*2,ndf*4,kernel_size=(4,4,4),stride=(1,2,2),padding=(1,1,1),bias=False),
        nn.BatchNorm3d(ndf*4),
        nn.LeakyReLU(0.2,inplace=True),
        nn.Conv3d(ndf*4,ndf*8,kernel_size=(4,4,4),stride=(1,2,2),padding=(1,1,1),bias=False),
        nn.BatchNorm3d(ndf*8),
        nn.LeakyReLU(0.2,inplace=True),
        nn.Conv3d(ndf*8,1,kernel_size=(1,8,8),stride=1,padding=0,bias=False)
    )
  def forward(self,x):
    return self.main(x).view(-1,1).squeeze(1)

class ContentDiscriminator(nn.Module):
  def __init__(self,content_dim=512):
    super(ContentDiscriminator,self).__init__()
    self.main=nn.Sequential(
        nn.Linear(content_dim,1024),
        nn.LeakyReLU(0.2,inplace=True),
        nn.Linear(1024,512),
        nn.LeakyReLU(0.2,inplace=True),
        nn.Linear(512,1)
    )
  def forward(self,x):
    return self.main(x)
  
class MotionDiscriminator(nn.Module):
  def __init__(self,motion_dim=128):
    super(MotionDiscriminator,self).__init__()
    self.main=nn.Sequential(
        nn.Linear(motion_dim,256),
        nn.LeakyReLU(0.2,inplace=True),
        nn.Linear(256,128),
        nn.LeakyReLU(0.2,inplace=True),
        nn.Linear(128,1)
    )
  def forward(self,x):
    return self.main(x)