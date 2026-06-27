import os
import requests
import zipfile
import pickle # Keep pickle if needed for future advanced processing, but for now np.save is used
from tqdm.auto import tqdm
import numpy as np # Ensure numpy is imported for array operations
import tensorflow as tf # Import tensorflow for tf.data.Dataset

# Explicit definitions for standalone execution (uncommented for fix):
OUTPUT_VOXEL_SHAPE = (64, 64, 64, 1) # From cell 3f89e693

# Placeholder for the mesh_to_voxel_grid function
# In a real scenario, this function would convert an .off mesh file
# into a binary voxel grid of the specified shape.
# This typically requires libraries like 'trimesh', 'pyntcloud', or custom voxelization code.
# For demonstration purposes, we return a dummy voxel grid of the correct spatial dimensions.
def mesh_to_voxel_grid(off_filepath, target_spatial_shape=(64, 64, 64)):
    """
    Placeholder function to convert an .off file to a voxel grid.
    Returns a dummy binary voxel grid for demonstration.
    A real implementation would use libraries like `trimesh` or `voxelpy`.
    """
    # Create an empty voxel grid
    voxel_grid = np.zeros(target_spatial_shape, dtype=np.float32)
    # Simulate a sparse object by setting some random voxels to 1
    num_active_voxels = np.random.randint(500, 2000) # Arbitrary number of active active voxels
    if num_active_voxels > 0:
        coords = [np.random.randint(0, dim, size=num_active_voxels) for dim in target_spatial_shape]
        voxel_grid[tuple(coords)] = 1.0

    return voxel_grid

# Define paths and URLs
base_dir = './data'
download_url = 'http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip'
zip_file_path = os.path.join(base_dir, 'ModelNet10.zip')
extraction_path = os.path.join(base_dir, 'ModelNet10')
modelnet_path = os.path.join(extraction_path, 'ModelNet10') # Corrected path after extraction

processed_data_path = './data/processed_modelnet10_voxels.npy' # Path to save processed data (Numpy array)

# Create the base directory if it doesn't exist
os.makedirs(base_dir, exist_ok=True)

# --- Download and Extract ModelNet10 ---
# Check if the ModelNet10 directory is present and not empty
if not os.path.exists(modelnet_path) or not os.listdir(modelnet_path):
    print(f"Downloading ModelNet10 from {download_url}...")
    try:
        response = requests.get(download_url, stream=True)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        total_size = int(response.headers.get('content-length', 0))
        # Use tqdm for a download progress bar
        with open(zip_file_path, 'wb') as f:
            for chunk in tqdm(response.iter_content(chunk_size=8192),
                             total=total_size // 8192, unit="KB", desc="Downloading ModelNet10"):
                f.write(chunk)
        print(f"Download complete. File saved to {zip_file_path}")

        print(f"Extracting {zip_file_path} to {extraction_path}...")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extraction_path)
        print("Extraction complete.")

        # Verify extraction by listing contents
        print(f"Contents of extracted directory: {modelnet_path}")
        if os.path.exists(modelnet_path):
            for item in os.listdir(modelnet_path):
                print(f"- {item}")
        else:
            print(f"Error: Expected ModelNet10 directory not found at {modelnet_path} after extraction.")

    except requests.exceptions.RequestException as e:
        print(f"Failed to download ModelNet10: {e}")
    except zipfile.BadZipFile:
        print(f"Error: Corrupted zip file: {zip_file_path}")
    except Exception as e:
        print(f"An unexpected error occurred during download/extraction: {e}")
else:
    print(f"ModelNet10 data already present at {modelnet_path}. Skipping download and extraction.")


# --- Voxelize .off files or load pre-processed data ---
all_voxel_grids = []
target_spatial_shape = OUTPUT_VOXEL_SHAPE[:-1] # e.g., (64, 64, 64)

if os.path.exists(processed_data_path):
    print(f"Loading previously processed voxel data from {processed_data_path}...")
    try:
        all_voxel_grids = np.load(processed_data_path)
        print(f"Loaded {len(all_voxel_grids)} voxel grids.")
    except Exception as e:
        print(f"Error loading processed data: {e}. Re-processing .off files.")
        all_voxel_grids = [] # Reset to re-process

if len(all_voxel_grids) == 0:
    print(f"Processing .off files in {modelnet_path} and voxelizing...")

    if not os.path.exists(modelnet_path) or not os.listdir(modelnet_path):
        print(f"ModelNet10 directory not found at {modelnet_path} or is empty. Cannot process files.")
        # Fallback to dummy data if ModelNet10 is not available
        # This will be handled by the 'Prepare data for TensorFlow Dataset' block
    else:
        # Collect all OFF file paths
        off_files_to_process = []
        for class_name in os.listdir(modelnet_path):
            class_path = os.path.join(modelnet_path, class_name)
            if os.path.isdir(class_path):
                for split_type in ['train', 'test']:
                    split_path = os.path.join(class_path, split_type)
                    if os.path.isdir(split_path):
                        for filename in os.listdir(split_path):
                            if filename.endswith('.off'):
                                off_files_to_process.append(os.path.join(split_path, filename))

        # Process files with a progress bar
        voxel_grids_list = []
        for off_filepath in tqdm(off_files_to_process, desc="Voxelizing Models"):
            voxel_grid = mesh_to_voxel_grid(off_filepath, target_spatial_shape)
            if voxel_grid is not None:
                voxel_grids_list.append(voxel_grid)

        all_voxel_grids = np.array(voxel_grids_list, dtype=np.float32)

        # Save processed data if successful
        if len(all_voxel_grids) > 0:
            np.save(processed_data_path, all_voxel_grids)
            print(f"Processed {len(all_voxel_grids)} voxel grids and saved to {processed_data_path}.")
        else:
            print("No .off files were successfully voxelized. Will use a dummy dataset.")


# --- Prepare data for TensorFlow Dataset ---
if len(all_voxel_grids) == 0:
    print("No real voxel data available. Creating a dummy dataset for training.")
    num_placeholder_samples = 1000
    voxel_data_for_tf = tf.random.uniform(shape=(num_placeholder_samples,) + OUTPUT_VOXEL_SHAPE, minval=-1.0, maxval=1.0)
else:
    print("Preparing voxel data for TensorFlow...")

    # Ensure voxel grids have the channel dimension if OUTPUT_VOXEL_SHAPE expects it
    # all_voxel_grids shape: (N, D, H, W)
    # OUTPUT_VOXEL_SHAPE: (D, H, W, C) -- C is typically 1 for binary voxels
    if all_voxel_grids.shape[1:] == OUTPUT_VOXEL_SHAPE[:-1] and OUTPUT_VOXEL_SHAPE[-1] == 1:
        voxel_data_for_tf = np.expand_dims(all_voxel_grids, axis=-1)
        print(f"Expanded dimensions to add channel. New shape: {voxel_data_for_tf.shape}")
    elif all_voxel_grids.shape[1:] == OUTPUT_VOXEL_SHAPE[1:]: # Already has correct shape including channel
        voxel_data_for_tf = all_voxel_grids
        print(f"Voxel data already has correct shape. Shape: {voxel_data_for_tf.shape}")
    else:
        # This case implies a mismatch in dimensions that cannot be simply fixed by adding a channel
        print(f"Warning: Loaded voxel data shape {all_voxel_grids.shape} incompatible with OUTPUT_VOXEL_SHAPE {OUTPUT_VOXEL_SHAPE}. Attempting basic assignment.")
        voxel_data_for_tf = all_voxel_grids

    # Normalize data from [0, 1] (typical for binary voxelization) to [-1, 1] for Tanh activation
    # Assuming `mesh_to_voxel_grid` or loaded data produces values in [0, 1]
    voxel_data_for_tf = (voxel_data_for_tf * 2.0) - 1.0
    print(f"Normalized voxel data to range [-1, 1]. Resulting shape: {voxel_data_for_tf.shape}")


# Create tf.data.Dataset
dataset = tf.data.Dataset.from_tensor_slices(voxel_data_for_tf)