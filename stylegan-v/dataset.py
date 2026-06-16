import glob
import torch
import os
# Install PyAV, a dependency for torchvision.io.read_video
#!pip install av

import av # Explicitly import av to ensure torchvision.io finds it

import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# # Direct download link for UCF101.rar
# # This link is from the official UCF server, typically more reliable.
# ucf101_url = 'http://crcv.ucf.edu/data/UCF101/UCF101.rar'
# output_filename = 'UCF101.rar'

# # Download the file using wget, ignoring certificate errors
# wget {ucf101_url} -O {output_filename} --no-check-certificate

# print(f"Downloaded {output_filename}")

# # Install unrar if not already installed
# !apt-get install unrar

# # Extract the dataset
# !unrar x {output_filename}

# print("UCF101 dataset extracted.")

# Assuming the extracted data is in a folder named 'UCF-101'
data_root = './UCF-101'

# List all .avi files recursively within the data_root
video_files = glob.glob(os.path.join(data_root, '**', '*.avi'), recursive=True)

print(f"Found {len(video_files)} video files.")
if len(video_files) > 5:
    print("First 5 video files:")
    for i in range(5):
        print(video_files[i])
else:
    for video_file in video_files:
        print(video_file)

class VideoDataset(Dataset):
    def __init__(self, video_paths, num_frames=8, transform=None, sample_strategy='random'):
        self.video_paths = video_paths
        self.num_frames = num_frames
        self.transform = transform
        self.sample_strategy = sample_strategy # 'random' or 'uniform'

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        try:
            # Read video frames using torchvision.io.read_video
            # This returns (T, H, W, C)
            video, _, _ = torchvision.io.read_video(video_path, pts_unit='sec')

            if not isinstance(video, torch.Tensor):
                raise ValueError("Video reader did not return a torch.Tensor")

            total_frames = video.shape[0]
            if total_frames < self.num_frames:
                print(f"Warning: Video {video_path} is too short ({total_frames} frames). Trying next video.")
                return self.__getitem__((idx + 1) % len(self))

            if self.sample_strategy == 'random':
                start_frame = torch.randint(0, total_frames - self.num_frames + 1, (1,)).item() if total_frames > self.num_frames else 0
                indices = list(range(start_frame, start_frame + self.num_frames))
            elif self.sample_strategy == 'uniform':
                step = total_frames // self.num_frames
                if step == 0:
                     print(f"Warning: Video {video_path} is too short for uniform sampling ({total_frames} frames). Trying next video.")
                     return self.__getitem__((idx + 1) % len(self))
                indices = [i * step for i in range(self.num_frames)]
            else:
                raise ValueError(f"Unknown sample_strategy: {self.sample_strategy}")

            # Select the frames
            video = video[indices] # Now (num_frames, H, W, C)

            # Apply transformations frame by frame
            transformed_frames = []
            for i in range(self.num_frames):
                frame = video[i] # (H, W, C)
                frame = frame.permute(2, 0, 1) # (C, H, W) for transforms
                if self.transform:
                    frame = self.transform(frame)
                transformed_frames.append(frame)

            # Stack frames back into (C, T, H, W)
            video = torch.stack(transformed_frames, dim=1) # (C, num_frames, H, W)

            return video
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            # Fallback: try to load another video if one fails
            # This can lead to infinite recursion if all videos are problematic
            return self.__getitem__((idx + 1) % len(self))