from tensorflow import keras
import numpy as np
import tensorflow as tf

BATCH_SIZE=54

(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
mnist_digits = np.concatenate([x_train, x_test], axis=0)
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

train_dataset = tf.data.Dataset.from_tensor_slices(mnist_digits)
train_dataset = train_dataset.shuffle(1024).batch(BATCH_SIZE)
