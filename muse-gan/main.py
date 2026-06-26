import torch
from generator import Generator
from discriminator import Discriminator
import torch.nn as nn

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