import matplotlib.pyplot as plt
from generator import build_generator_wgan
from discriminator import build_critic_wgan
from dataset import dataset_wgan,BATCH_SIZE
from tensorflow.keras import optimizers
from model import WGAN_GP
import tensorflow as tf

latent_dim_wgan = 100
epochs_wgan = 100
batch_size_wgan = BATCH_SIZE
gp_weight = 10.0 # Gradient penalty weight, a common value for WGAN-GP

generator_wgan = build_generator_wgan(latent_dim_wgan)
critic_wgan = build_critic_wgan()

wgan = WGAN_GP(generator=generator_wgan, critic=critic_wgan, latent_dim=latent_dim_wgan, gp_weight=gp_weight)

# Compile the WGAN-GP model
wgan.compile(
    g_optimizer=optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9), # Common optimizer settings for WGAN
    c_optimizer=optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
)

print(f"\nStarting WGAN-GP training for {epochs_wgan} epochs...")
wgan.fit(dataset_wgan, epochs=epochs_wgan)

# Generate some new images after WGAN training

noise_wgan = tf.random.normal([16, latent_dim_wgan]) # Generate 16 new images
generated_images_wgan = generator_wgan(noise_wgan)
generated_images_wgan = (generated_images_wgan * 0.5 + 0.5) # Denormalize to [0, 1]

plt.figure(figsize=(4, 4))
for i in range(generated_images_wgan.shape[0]):
    plt.subplot(4, 4, i+1)
    plt.imshow(generated_images_wgan[i, :, :, 0], cmap='gray')
    plt.axis('off')
plt.suptitle('Generated Images from WGAN-GP')
plt.show()