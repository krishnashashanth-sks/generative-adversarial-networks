from torch.utils.data import Dataset
import torch

# --- Dummy Dataset Class ---
class DummyVideoDataset(Dataset):
    def __init__(self, num_samples, sequence_length, initial_frame_shape, motion_sequence_shape):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.initial_frame_shape = initial_frame_shape # (C, H, W)
        self.motion_sequence_shape = motion_sequence_shape # (T, C_motion, H, W)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a dummy initial frame (C, H, W)
        initial_frame = torch.randn(self.initial_frame_shape)
        # Generate a dummy motion sequence (T, C_motion, H, W)
        motion_sequence = torch.randn(self.motion_sequence_shape)
        return initial_frame, motion_sequence