import torch.nn as nn
from generator_layers import *
import torch

class Generator(nn.Module):
  def __init__(self,z_dim,bar_latent_dim,chord_latent_dim,note_embed_dim,n_bars,n_tracks,n_pitches,n_steps_per_bar): # Fixed: n_step_per_bar -> n_steps_per_bar
    super(Generator,self).__init__()
    self.n_bars=n_bars
    self.n_tracks=n_tracks
    self.bar_gen=BarGenerator(z_dim,bar_latent_dim,n_bars)
    self.chord_gen=ChordGenerator(bar_latent_dim,chord_latent_dim)
    self.note_generators=nn.ModuleList([
        NoteGenerator(bar_latent_dim,chord_latent_dim,note_embed_dim,n_pitches,n_steps_per_bar)
        for _ in range(self.n_tracks)
    ])
  def forward(self,z):
    bar_latents=self.bar_gen(z)
    chord_latents=self.chord_gen(bar_latents) # Fixed: chod_gen -> chord_gen
    all_tracks_music=[]
    for track_idx in range(self.n_tracks):
      track_bars_music=[]
      for bar_idx in range(self.n_bars):
        current_bar_latent=bar_latents[:,bar_idx,:]
        current_chord_latent=chord_latents[:,bar_idx,:] # Fixed: chod_latent -> chord_latent
        notes_for_bar=self.note_generators[track_idx]( # Fixed: note_generator -> note_generators
            current_bar_latent,current_chord_latent
        )
        track_bars_music.append(notes_for_bar)
      all_tracks_music.append(torch.stack(track_bars_music,dim=1)) # Fixed: trac_bars_music -> track_bars_music
    generated_music=torch.stack(all_tracks_music,dim=1)
    return generated_music