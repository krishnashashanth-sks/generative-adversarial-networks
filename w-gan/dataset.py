import tensorflow as tf

# Load and preprocess the MNIST dataset (same as for vanilla GAN)
(x_train_wgan, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_train_wgan = x_train_wgan.reshape(x_train_wgan.shape[0], 28, 28, 1).astype('float32')
x_train_wgan = tf.image.resize(x_train_wgan, (32, 32)).numpy()
x_train_wgan = (x_train_wgan - 127.5) / 127.5
BATCH_SIZE=128
dataset_wgan = tf.data.Dataset.from_tensor_slices(x_train_wgan).shuffle(x_train_wgan.shape[0]).batch(BATCH_SIZE, drop_remainder=True)