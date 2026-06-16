import tensorflow as tf
import tensorflow_datasets as tfds
from utils import preprocess_image_tfds
import matplotlib.pyplot as plt

# 'cycle_gan/monet2photo' is a well-known and valid configuration
dataset, info = tfds.load('cycle_gan/monet2photo', with_info=True, as_supervised=True)

train_real_raw = dataset['trainA'] # e.g., photos for 'monet2photo'
train_anime_raw = dataset['trainB'] # e.g., monet paintings for 'monet2photo'

# 2. RAM-Efficient Preprocessing
BUFFER_SIZE = 1000
BATCH_SIZE = 4 # Keep small to avoid 'System Crashed' OOM errors

train_real = train_real_raw.map(preprocess_image_tfds).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
train_anime = train_anime_raw.map(preprocess_image_tfds).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Zip them together for your AnimeGAN2 training loop
train_dataset = tf.data.Dataset.zip((train_real, train_anime))

for real_img, anime_img in train_dataset.take(1):
    print(f"Example real image batch shape: {real_img.shape}")
    print(f"Example anime image batch shape: {anime_img.shape}")
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow((real_img[0] * 0.5 + 0.5).numpy())
    plt.title("Example Real Image (Monet2Photo)")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow((anime_img[0] * 0.5 + 0.5).numpy())
    plt.title("Example Monet Painting (Monet2Photo)")
    plt.axis('off')
    plt.show()