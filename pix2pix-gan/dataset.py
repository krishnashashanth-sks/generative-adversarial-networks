# !wget http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz
# !tar -xzf facades.tar.gz

import tensorflow as tf
import os
from utils import *

# Define image parameters
IMG_WIDTH = 256
IMG_HEIGHT = 256

# Define paths to the dataset
PATH = os.path.join('facades')
TRAIN_DATA_DIR = os.path.join(PATH, 'train')
TEST_DATA_DIR = os.path.join(PATH, 'test')
VAL_DATA_DIR = os.path.join(PATH, 'val')

print(f"Image width: {IMG_WIDTH}, Image height: {IMG_HEIGHT}")
print(f"Training data directory: {TRAIN_DATA_DIR}")
print(f"Test data directory: {TEST_DATA_DIR}")
print(f"Validation data directory: {VAL_DATA_DIR}")

BUFFER_SIZE = 400
BATCH_SIZE = 1

# List all image file paths in the training directory
train_dataset = tf.data.Dataset.list_files(TRAIN_DATA_DIR + '/*.jpg')

# Map the preprocessing function and configure for performance
train_dataset = train_dataset.map(preprocess_image_train,
                                    num_parallel_calls=tf.data.AUTOTUNE)

# Shuffle, batch, and prefetch the dataset
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

print(f"Training dataset created with BUFFER_SIZE={BUFFER_SIZE} and BATCH_SIZE={BATCH_SIZE}")
print("Dataset configured for shuffling, batching, and prefetching.")

print("Creating test dataset...")
test_dataset = tf.data.Dataset.list_files(TEST_DATA_DIR + '/*.jpg')
test_dataset = test_dataset.map(preprocess_image_test,
                                  num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

print("Creating validation dataset...")
val_dataset = tf.data.Dataset.list_files(VAL_DATA_DIR + '/*.jpg')
val_dataset = val_dataset.map(preprocess_image_test,
                                 num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

print("Test and validation datasets created and configured.")

print("Displaying a few examples from the training dataset...")
for example_input, example_target in train_dataset.take(1):
    generate_images_and_display(None, example_input, example_target)

print("Dataset sample visualization complete.")