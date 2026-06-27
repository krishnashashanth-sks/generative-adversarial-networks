
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def visualize_voxel_grid(voxel_grid, title="Generated Voxel Grid"):
    """
    Visualizes a single 3D binary voxel grid using matplotlib.
    Assumes voxel_grid has shape (D, H, W, 1) and values are in [-1, 1].
    """
    # Convert from [-1, 1] to [0, 1] if necessary for visualization
    if np.min(voxel_grid) < 0:
        voxel_grid = (voxel_grid + 1) / 2.0

    # Remove the channel dimension if it exists and is 1
    if voxel_grid.shape[-1] == 1:
        voxel_grid = voxel_grid[..., 0]

    # Ensure it's a binary grid for plotting
    binary_grid = (voxel_grid > 0.5).numpy() # Convert to numpy and apply threshold

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Get the coordinates of occupied voxels
    occupied_voxels = np.argwhere(binary_grid == True)

    if occupied_voxels.size > 0:
        # Plot each voxel as a point. Adjust marker size (s) for better visibility.
        ax.scatter(occupied_voxels[:, 0], occupied_voxels[:, 1], occupied_voxels[:, 2],
                   zdir='z', c='red', marker='o', s=20) # s is marker size, adjust as needed

    ax.set_title(title)
    ax.set_xlabel('Depth')
    ax.set_ylabel('Height')
    ax.set_zlabel('Width')
    ax.set_aspect('auto')
    plt.show()
