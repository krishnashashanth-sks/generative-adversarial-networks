import torch.optim as optim
import torch.nn as nn
import torch
from model import MocoGAN
from dataset import DummyVideoDataset
from torch.utils.data import DataLoader
from train import train
from inference import generate_video_for_inference
from visualize import *
mocogan_model = MocoGAN()


# Hyperparameters (example values, these will need tuning)
LEARNING_RATE_G = 0.0002
LEARNING_RATE_D = 0.0002
BETA1 = 0.5
BETA2 = 0.999

# Initialize optimizers
optimizer_G = optim.Adam(
    mocogan_model.generator.parameters(),
    lr=LEARNING_RATE_G,
    betas=(BETA1, BETA2)
)
optimizer_D_video = optim.Adam(
    mocogan_model.video_discriminator.parameters(),
    lr=LEARNING_RATE_D,
    betas=(BETA1, BETA2)
)
optimizer_D_content = optim.Adam(
    mocogan_model.content_discriminator.parameters(),
    lr=LEARNING_RATE_D,
    betas=(BETA1, BETA2)
)
optimizer_D_motion = optim.Adam(
    mocogan_model.motion_discriminator.parameters(),
    lr=LEARNING_RATE_D,
    betas=(BETA1, BETA2)
)

# Loss Functions
criterion_GAN = nn.BCEWithLogitsLoss() # For adversarial losses
criterion_reconstruction = nn.L1Loss() # For reconstruction (e.g., initial frame reconstruction)

NUM_EPOCHS = 5 # Small number for demonstration
BATCH_SIZE = 1 # Small batch size for demonstration
SEQUENCE_LENGTH = 5

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mocogan_model.to(device)

# --- Instantiate Dummy Dataset and DataLoader ---
num_samples = 100 # Total number of dummy samples
initial_frame_shape = (3, 128, 128)
motion_sequence_shape = (SEQUENCE_LENGTH, 2, 128, 128)

dummy_dataset = DummyVideoDataset(
    num_samples=num_samples,
    sequence_length=SEQUENCE_LENGTH,
    initial_frame_shape=initial_frame_shape,
    motion_sequence_shape=motion_sequence_shape
)
dummy_dataloader = DataLoader(dummy_dataset, batch_size=BATCH_SIZE, shuffle=True)

train(NUM_EPOCHS,dummy_dataloader,BATCH_SIZE,SEQUENCE_LENGTH,mocogan_model,criterion_GAN,criterion_reconstruction,optimizer_G,optimizer_D_video,optimizer_D_content,optimizer_D_motion,device)


# Example usage for inference with DataLoader:
all_inferred_videos = []

print("Starting inference on dummy dataloader...")
# Iterate over the dummy DataLoader
for i, (initial_frame_batch, motion_sequence_batch) in enumerate(dummy_dataloader):
    initial_frame_batch = initial_frame_batch.to(device)
    motion_sequence_batch = motion_sequence_batch.to(device)

    # Generate video for the current batch
    inferred_video_batch = generate_video_for_inference(mocogan_model.generator, initial_frame_batch, motion_sequence_batch)
    all_inferred_videos.append(inferred_video_batch)

    if (i + 1) % 10 == 0:
        print(f"  Processed batch {i+1}/{len(dummy_dataloader)}")

# Concatenate all generated videos (if needed, or process them individually)
# For visualization, we might just take one example or a few.
# Let's take the first generated video from the first batch for demonstration.
inferred_video_example = all_inferred_videos[0] if all_inferred_videos else None



# Call the visualization function with a few inferred videos
num_videos_to_visualize = min(3, len(all_inferred_videos)) # Visualize up to 3 videos

for i in range(num_videos_to_visualize):
    video_to_visualize = all_inferred_videos[i]
    visualize_video_frames(video_to_visualize, title=f"MocoGAN Generated Video (Batch {i+1})")


# Call the animation function with a few inferred videos
num_videos_to_animate = min(3, len(all_inferred_videos)) # Animate up to 3 videos

for i in range(num_videos_to_animate):
    video_to_animate = all_inferred_videos[i]
    animate_video_frames(video_to_animate, filename=f"mocogan_generated_video_batch_{i+1}.gif", fps=2)