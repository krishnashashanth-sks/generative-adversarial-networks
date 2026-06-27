import tensorflow as tf
import numpy as np

IMG_WIDTH = 32 # CIFAR10 image width
IMG_HEIGHT = 32 # CIFAR10 image height
BATCH_SIZE = 16 # Reusing the previously defined BATCH_SIZE

# Load the CIFAR10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Combine train and test for a single dataset source, or just use train
images = np.concatenate((x_train, x_test), axis=0)
labels = np.concatenate((y_train, y_test), axis=0)

# Get class names to determine NUM_DOMAINS
# CIFAR10 has 10 classes
class_names = [f'class_{i}' for i in range(10)] # Dummy class names for CIFAR10
NUM_DOMAINS = len(class_names)
print(f"Detected {NUM_DOMAINS} domains (classes) for CIFAR10.")

def preprocess_image_dataset(image, label):
    image = tf.cast(image, tf.float32) # Cast to float32
    image = (image / 127.5) - 1.0       # Normalize to [-1, 1]
    # Convert integer label to one-hot encoding for domain labels
    one_hot_label = tf.one_hot(tf.squeeze(label), depth=NUM_DOMAINS) # tf.squeeze for CIFAR10 labels
    return image, one_hot_label

# Create a tf.data.Dataset from the loaded numpy arrays
dataset_raw = tf.data.Dataset.from_tensor_slices((images, labels))
dataset_raw = dataset_raw.shuffle(buffer_size=len(images))
dataset_raw = dataset_raw.batch(BATCH_SIZE)

# Apply the preprocessing to the dataset
dataset = dataset_raw.map(preprocess_image_dataset, num_parallel_calls=tf.data.AUTOTUNE)

# Prefetch for better performance
dataset = dataset.prefetch(tf.data.AUTOTUNE)