import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from layers import *
from model import VAEGAN
from dataset import train_dataset
import matplotlib.pyplot as plt

IMAGE_SIZE=(28,28,1) # Changed for MNIST
LATENT_DIM=128
EPOCHS=10
LR_VAE=1e-4
LR_DISCRIMINATOR=1e-4
LAMBDA_REC=10
LAMBDA_KL=0.1
LAMBDA_ADV=1

encoder = build_encoder(IMAGE_SIZE, LATENT_DIM)
encoder.summary()

decoder = build_decoder(LATENT_DIM, IMAGE_SIZE)
decoder.summary()

discriminator = build_discriminator(IMAGE_SIZE)
discriminator.summary()

vae_gan = VAEGAN(encoder, decoder, discriminator,LAMBDA_REC,LATENT_DIM,LAMBDA_KL,LAMBDA_ADV)

# Optimizers
vae_optimizer = keras.optimizers.Adam(learning_rate=LR_VAE, beta_1=0.5)
discriminator_optimizer = keras.optimizers.Adam(learning_rate=LR_DISCRIMINATOR, beta_1=0.5)

vae_gan.compile(vae_optimizer=vae_optimizer, discriminator_optimizer=discriminator_optimizer)

vae_gan.fit(train_dataset, epochs=EPOCHS)

z_sample = tf.random.normal(shape=(10, LATENT_DIM))
generated_images = vae_gan.decoder.predict(z_sample)

plt.figure(figsize=(10, 2))
for i in range(10):
    ax = plt.subplot(1, 10, i + 1)
    plt.imshow(generated_images[i].squeeze(), cmap='gray') # Removed .numpy()
    plt.axis('off')
plt.show()