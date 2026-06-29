import os
import requests
import zipfile
import tarfile
from model import *

DOWNLOAD_BASE_DIR = "data"
os.makedir(DOWNLOAD_BASE_DIR)

# DATA_DIR_BIRDS should be the parent directory where 'birds' folder is located
# Assuming CUB_200_2011/CUB_200_2011 contains images, bounding_boxes.txt etc.
# and birds/train contains pickle files.
CUB_DATA_PARENT_DIR = os.path.join(DOWNLOAD_BASE_DIR, "birds") # Where CUB_200_2011.tgz will be extracted

# Define the full path to the extracted CUB dataset root where 'images' folder and .txt files are
CUB_DATASET_ROOT_PATH = os.path.join(CUB_DATA_PARENT_DIR, "CUB_200_2011", "CUB_200_2011")

# Define the path to the 'birds' folder that contains 'train' and 'test' folders with pickle files
STACKGAN_DATA_DIR = os.path.join(CUB_DATA_PARENT_DIR, "birds")

# URLs for dataset download
CUB_TGZ_URL = "https://data.deepai.org/CUB_200_2011.tgz"
CUB_TGZ_PATH = os.path.join(CUB_DATA_PARENT_DIR, "CUB_200_2011.tgz")

BIRDS_ZIP_URL = "https://github.com/hanzhanggit/StackGAN/raw/master/data/cub/birds.zip"
BIRDS_ZIP_PATH = os.path.join(CUB_DATA_PARENT_DIR, "birds.zip")

# Ensure base directories exist
os.makedirs(CUB_DATA_PARENT_DIR, exist_ok=True)

print("Checking for CUB_200_2011 image data...")
if not os.path.exists(os.path.join(CUB_DATASET_ROOT_PATH, "images")):
    print("CUB-200-2011 image data not found. Attempting to download and extract...")

    try:
        # Download CUB_200_2011.tgz
        print(f"Downloading {CUB_TGZ_URL} to {CUB_TGZ_PATH}...")
        response = requests.get(CUB_TGZ_URL, stream=True)
        response.raise_for_status()
        with open(CUB_TGZ_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")

        # Extract CUB_200_2011.tgz
        print(f"Extracting {CUB_TGZ_PATH} to {CUB_DATA_PARENT_DIR}...")
        with tarfile.open(CUB_TGZ_PATH, 'r:gz') as tar_ref:
            tar_ref.extractall(path=CUB_DATA_PARENT_DIR)
        print("CUB_200_2011.tgz extraction complete.")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading or extracting CUB_200_2011.tgz: {e}")
        print("Please ensure you have access to the internet and enough storage space.")

print("Checking for StackGAN pickle files (birds.zip)...")
if not os.path.exists(os.path.join(STACKGAN_DATA_DIR, "train", "filenames.pickle")):
    print("StackGAN pickle files not found. Attempting to download and extract birds.zip...")

    try:
        # Download birds.zip
        print(f"Downloading {BIRDS_ZIP_URL} to {BIRDS_ZIP_PATH}...")
        response = requests.get(BIRDS_ZIP_URL, stream=True)
        response.raise_for_status()
        with open(BIRDS_ZIP_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")

        # Extract birds.zip
        print(f"Extracting {BIRDS_ZIP_PATH} to {CUB_DATA_PARENT_DIR}...")
        with zipfile.ZipFile(BIRDS_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(CUB_DATA_PARENT_DIR)
        print("birds.zip extraction complete.")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading or extracting birds.zip: {e}")
        print("Please ensure you have access to the internet and enough storage space.")

# --- Instantiate and Train StackGAN Stage-I ---
print("\n--- Starting Stage-I GAN Training ---")
stage1_gan = StackGANStage1(
    epochs=1, # Reduced epochs for demonstration, increase for actual training
    batch_size=64,
    stage1_generator_lr=0.0002,
    stage1_discriminator_lr=0.0002,
    data_dir=STACKGAN_DATA_DIR
)
stage1_gan.train_stage1()

# --- Instantiate and Train StackGAN Stage-II ---
print("\n--- Starting Stage-II GAN Training ---")
stage2_gan = StackGANStage2(
    epochs=1, # Reduced epochs for demonstration, increase for actual training
    batch_size=32, # Batch size for Stage-II
    stage2_generator_lr=0.0002,
    stage2_discriminator_lr=0.0002,
    data_dir=STACKGAN_DATA_DIR
)
stage2_gan.train_stage2()

print("\n--- StackGAN Training Complete ---")