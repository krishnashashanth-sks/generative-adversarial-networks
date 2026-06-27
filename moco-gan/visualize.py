import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

def visualize_video_frames(video_tensor, title="Generated Video Frames"):
    # Ensure the tensor is on CPU and converted to numpy
    # video_tensor shape: (B, T, C, H, W)
    # The input video_tensor is expected to be a single video (batch size 1 assumed for display)
    video_np = video_tensor.squeeze(0).numpy() # Remove batch dimension, convert to numpy

    num_frames = video_np.shape[0]

    fig, axes = plt.subplots(1, num_frames, figsize=(3 * num_frames, 3))
    # If there's only one frame, axes will not be an array, so make it one
    if num_frames == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=16)

    for i in range(num_frames):
        frame = video_np[i] # Current frame shape: (C, H, W)

        # Permute from (C, H, W) to (H, W, C) for matplotlib
        frame_display = np.transpose(frame, (1, 2, 0))

        # Rescale from [-1, 1] to [0, 1] for visualization
        frame_display = (frame_display + 1) / 2.0

        # Clip values to ensure they are within [0, 1] in case of minor floating point issues
        frame_display = np.clip(frame_display, 0, 1)

        ax = axes[i]
        ax.imshow(frame_display)
        ax.set_title(f"Frame {i+1}")
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent suptitle overlap
    plt.show()


def animate_video_frames(video_tensor, filename="generated_video.gif", fps=2):
    # Ensure the tensor is on CPU and converted to numpy
    # video_tensor shape: (B, T, C, H, W)
    video_np = video_tensor.squeeze(0).numpy() # Remove batch dimension, convert to numpy

    num_frames = video_np.shape[0]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.axis('off') # Hide axes

    # Initialize the first frame
    frame = video_np[0] # Current frame shape: (C, H, W)
    frame_display = np.transpose(frame, (1, 2, 0))
    frame_display = (frame_display + 1) / 2.0
    frame_display = np.clip(frame_display, 0, 1)

    im = ax.imshow(frame_display)

    def update(i):
        frame = video_np[i]
        frame_display = np.transpose(frame, (1, 2, 0))
        frame_display = (frame_display + 1) / 2.0
        frame_display = np.clip(frame_display, 0, 1)
        im.set_array(frame_display)
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=1000/fps, blit=True)

    # Save the animation as a GIF
    print(f"Saving animation to {filename}...")
    ani.save(filename, writer='pillow', fps=fps)
    print("Animation saved!")

    plt.close(fig) # Close the plot to prevent it from displaying twice