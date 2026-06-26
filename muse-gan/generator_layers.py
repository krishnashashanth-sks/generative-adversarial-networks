import torch
import torch.nn as nn

class BarGenerator(nn.Module):
  def __init__(self,z_dim,bar_latent_dim,n_bars):
    super(BarGenerator,self).__init__()
    self.n_bars=n_bars
    self.lstm=nn.LSTM(
        input_size=z_dim,
        hidden_size=bar_latent_dim,
        num_layers=1,
        batch_first=True
    )
  def forward(self,z):
    lstm_input=z.unsqueeze(1).repeat(1,self.n_bars,1)
    bar_latents,_=self.lstm(lstm_input)
    return bar_latents

class ChordGenerator(nn.Module):
  def __init__(self,bar_latent_dim,chord_latent_dim):
    super(ChordGenerator,self).__init__()
    self.fc=nn.Linear(bar_latent_dim,chord_latent_dim)
  def forward(self,bar_latents):
    return self.fc(bar_latents)

class NoteGenerator(nn.Module):
  def __init__(self,bar_latent_dim,chord_latent_dim,note_embed_dim,n_pitches,n_steps_per_bar):
    super(NoteGenerator,self).__init__()
    self.n_pitches=n_pitches
    self.n_steps_per_bar=n_steps_per_bar
    self.input_dim=bar_latent_dim+chord_latent_dim
    self.lstm=nn.LSTM(
        input_size=self.input_dim,
        hidden_size=note_embed_dim,
        num_layers=1,
        batch_first=True
    )
    self.output_layer=nn.Linear(note_embed_dim,n_pitches)
  def forward(self,bar_latent,chord_latent):
    combined_latent=torch.cat([bar_latent,chord_latent],dim=1)
    lstm_input=combined_latent.unsqueeze(1).repeat(1,self.n_steps_per_bar,1)
    note_embeddings,_=self.lstm(lstm_input)
    output_pitches=torch.sigmoid(self.output_layer(note_embeddings))
    return output_pitches