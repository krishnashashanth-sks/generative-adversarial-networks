import tensorflow as tf

# Load the MNIST dataset
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

# Reshape and normalize images
# MNIST images are 28x28, our GAN was built for 32x32. We need to resize them.
# Also, add a channel dimension if not present (for grayscale images)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')

x_train = tf.image.resize(x_train, (32, 32)).numpy()

# Normalize pixel values to [-1, 1]
x_train = (x_train - 127.5) / 127.5

BATCH_SIZE=128

# Create a tf.data.Dataset
dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(x_train.shape[0]).batch(BATCH_SIZE, drop_remainder=True)