import torch.nn as nn

# Loss functions
adv_loss = nn.BCEWithLogitsLoss()
cycle_loss = nn.L1Loss()
identity_loss = nn.L1Loss()