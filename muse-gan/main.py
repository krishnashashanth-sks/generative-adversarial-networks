import matplotlib.pyplot as plt
import numpy as np
import torch
from generator import Generator
from discriminator import Discriminator
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import DummyMusicDataset
from train import train_gan

Z_DIM = 32  # Dimension of the latent noise vector
BAR_LATENT_DIM = 128  # Latent dimension for a single bar
CHORD_LATENT_DIM = 64 # Latent dimension for a chord progression
NOTE_EMBED_DIM = 32 # Embedding dimension for notes (pitch, velocity, duration)

N_BARS = 4  # Number of bars in a generated musical phrase
N_TRACKS = 4  # Number of instrument tracks (e.g., drum, piano, bass, guitar)
N_STEPS_PER_BAR = 16 # Number of time steps per bar (e.g., 16th notes)
N_PITCHES = 84 # Number of possible MIDI pitches (e.g., from C1 to B7)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen = Generator(Z_DIM, BAR_LATENT_DIM, CHORD_LATENT_DIM, NOTE_EMBED_DIM, N_BARS, N_TRACKS, N_PITCHES, N_STEPS_PER_BAR).to(device)

disc = Discriminator(BAR_LATENT_DIM, N_BARS, N_TRACKS, N_PITCHES, N_STEPS_PER_BAR).to(device)

# Training Parameters
BATCH_SIZE = 64
LR_G = 0.0002 # Learning rate for Generator
LR_D = 0.0002 # Learning rate for Discriminator
NUM_EPOCHS = 10

# Optimizers
optimizer_G = torch.optim.Adam(gen.parameters(), lr=LR_G, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(disc.parameters(), lr=LR_D, betas=(0.5, 0.999))

# Loss Function
# For simplicity, we'll use Binary Cross-Entropy Loss for this example.
# In a more advanced setup, WGAN-GP is often preferred for stability.
criterion = nn.BCEWithLogitsLoss()

# Instantiate the dummy dataset and dataloader
dummy_dataset = DummyMusicDataset(
    num_samples=BATCH_SIZE * 10, # For a few batches
    n_tracks=N_TRACKS,
    n_bars=N_BARS,
    n_steps_per_bar=N_STEPS_PER_BAR,
    n_pitches=N_PITCHES,
    bar_latent_dim=BAR_LATENT_DIM
)
dummy_dataloader = DataLoader(dummy_dataset, batch_size=BATCH_SIZE, shuffle=True)

train_gan(
    gen, disc, dummy_dataloader, optimizer_G, optimizer_D, criterion, NUM_EPOCHS, Z_DIM, device, BATCH_SIZE
)

# Set the generator to evaluation mode
gen.eval()

# Generate a single music sample
with torch.no_grad():
    z_sample = torch.randn(1, Z_DIM).to(device)
    generated_sample = gen(z_sample)

# Move to CPU and convert to numpy
generated_sample_np = generated_sample.cpu().squeeze(0).numpy() # Remove batch dimension

# The shape is (N_TRACKS, N_BARS, N_STEPS_PER_BAR, N_PITCHES)
# We want to visualize a single piano roll of (total_time_steps, N_PITCHES)
# First, combine N_BARS and N_STEPS_PER_BAR into a single time dimension
# Reshape each track to (N_BARS * N_STEPS_PER_BAR, N_PITCHES)

total_time_steps = N_BARS * N_STEPS_PER_BAR
combined_piano_roll = np.zeros((total_time_steps, N_PITCHES))

for track_idx in range(N_TRACKS):
    track_data = generated_sample_np[track_idx, :, :, :]
    # Reshape each track's bars into a continuous sequence
    track_piano_roll = track_data.reshape(total_time_steps, N_PITCHES)
    # Sum up the active notes (assuming binary activation, >0.5)
    combined_piano_roll += (track_piano_roll > 0.5)

# Clip values to 1 if multiple tracks play the same note at the same time
combined_piano_roll = np.clip(combined_piano_roll, 0, 1);

plt.figure(figsize=(15, 6))
plt.imshow(combined_piano_roll.T, aspect='auto', origin='lower', cmap='binary') # Transpose to have pitches on y-axis
plt.title('Generated Music Sample Piano-Roll (All Tracks Combined)')
plt.xlabel('Time Steps (16th notes)')
plt.ylabel('MIDI Pitch')
plt.colorbar(label='Note Activation')
plt.show()