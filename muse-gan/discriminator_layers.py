import torch.nn as nn

class BarDiscriminator(nn.Module):
  def __init__(self,n_tracks,n_pitches,n_steps_per_bar):
    super(BarDiscriminator,self).__init__()
    self.conv_layers=nn.Sequential(
        nn.Conv2d(n_tracks,64,kernel_size=(1,12),stride=(1,1),padding=0),
        nn.LeakyReLU(0.1,inplace=True),
        nn.Dropout(0.25),
        nn.Conv2d(64,128,kernel_size=(1,12),stride=(1,1),padding=0),
        nn.LeakyReLU(0.1,inplace=True),
        nn.Dropout(0.25),
        nn.Conv2d(128,256,kernel_size=(1,12),stride=(1,1),padding=0),
        nn.LeakyReLU(0.1,inplace=True),
        nn.Dropout(0.25)
    )
    self.adaptive_pool=nn.AdaptiveAvgPool2d((1,1))
    self.fc=nn.Sequential(
        nn.Linear(256,1)
    )
  def forward(self,bar_music):
    features=self.adaptive_pool(self.conv_layers(bar_music))
    features=features.view(features.size(0),-1)
    return self.fc(features)

class PhraseDiscriminator(nn.Module):
  def __init__(self,bar_latent_dim,n_bars):
    super(PhraseDiscriminator,self).__init__()
    self.lstm=nn.LSTM(
        input_size=bar_latent_dim,
        hidden_size=128,
        num_layers=1,
        batch_first=True
    )
    self.fc=nn.Sequential(
      nn.Linear(128,64),
      nn.LeakyReLU(0.2,inplace=True),
      nn.Linear(64,1)
    )
  def forward(self,bar_latents):
    _, (h_n, _) = self.lstm(bar_latents)
    return self.fc(h_n.squeeze(0)) # Squeeze the layer dimension (dim=0) to get (batch_size, hidden_size)