import tensorflow as tf

# 1. Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# 2. Concatenate the training and testing images
all_images = np.concatenate([train_images, test_images])

# 3. Reshape the image data to include a channel dimension
# MNIST images are 28x28 grayscale, so add a channel dimension of 1
all_images = all_images.reshape(all_images.shape[0], 28, 28, 1)

# 4. Cast the image data to tf.float32
all_images = tf.cast(all_images, tf.float32)

# 5. Normalize the image pixel values to the range [-1, 1]
all_images = (all_images - 127.5) / 127.5

# 6. Create a tf.data.Dataset from the preprocessed images
dataset = tf.data.Dataset.from_tensor_slices(all_images)