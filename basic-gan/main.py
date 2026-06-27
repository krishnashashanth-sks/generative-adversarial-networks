from generator import build_generator
from discriminator import build_discriminator
from model import build_gan
from dataset import dataset,BATCH_SIZE
from train import train_gan
from inference import generate_images
import matplotlib.pyplot as plt

latent_dim = 100
epochs = 100
batch_size = BATCH_SIZE

# Build and compile the Discriminator
discriminator = build_discriminator()

# Build the Generator
generator = build_generator(latent_dim)

# Build and compile the GAN (Generator + Discriminator)
gan = build_gan(generator, discriminator)

train_gan(generator, discriminator, gan, dataset, latent_dim, 5, batch_size)

generated_samples = generate_images(generator, latent_dim)
# You can then display these images
plt.figure(figsize=(4, 4))
for i in range(generated_samples.shape[0]):
    plt.subplot(4, 4, i+1)
    plt.imshow(generated_samples[i, :, :, 0], cmap='gray')
    plt.axis('off')
plt.show()