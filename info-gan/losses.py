import torch.nn as nn

# --- Loss Functions  ---
criterion_gan = nn.BCELoss() # Binary Cross-Entropy for real/fake
criterion_categorical = nn.CrossEntropyLoss() # For categorical latent codes
# For continuous codes, often MSE or a log-likelihood loss for Gaussian distributed codes
criterion_continuous = nn.MSELoss()