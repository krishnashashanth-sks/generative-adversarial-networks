import torch.nn as nn
from discriminator_layers import *
import torch

class Discriminator(nn.Module):
  def __init__(self,bar_latent_dim,n_bars,n_tracks,n_pitches,n_steps_per_bar):
    super(Discriminator,self).__init__()
    self.bar_discriminator=BarDiscriminator(n_tracks,n_pitches,n_steps_per_bar)
    self.phrase_discriminator=PhraseDiscriminator(bar_latent_dim,n_bars)
    self.final_classifier=nn.Sequential(
        nn.Linear(2,1)
    )
    self.n_bars=n_bars
  def forward(self,generated_music,bar_latents):
    batch_size=generated_music.size(0)
    bar_scores=[]
    for bar_idx in range(self.n_bars):
      current_bar_music=generated_music[:,:,bar_idx,:,:]
      bar_score=self.bar_discriminator(current_bar_music)
      bar_scores.append(bar_score)
    avg_bar_score=torch.mean(torch.stack(bar_scores,dim=1),dim=1)
    phrase_score=self.phrase_discriminator(bar_latents)

    if phrase_score.dim() == 3 and phrase_score.size(2) == 1:
        phrase_score = torch.mean(phrase_score, dim=1) # Average across bars

    combined_scores=torch.cat([avg_bar_score,phrase_score],dim=1)
    return self.final_classifier(combined_scores)