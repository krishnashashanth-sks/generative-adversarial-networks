import tensorflow as tf
import numpy as np
from dataset import dataset
from generator import BaseGenerator
from discriminator import BaseDiscriminator
from upsample_generator import UpsamplingGenerator
from upsample_discriminator import UpsamplingDiscriminator
from train import train_lapgan
from losses import *
from optimizers import *
from inference import infer_lapgan_image
import matplotlib.pyplot as plt

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

BATCH_SIZE = 64
BUFFER_SIZE = 10000 # Used for shuffling the dataset
LATENT_DIM = 100 # Dimension of the latent space for the generator

#  Shuffle the dataset
dataset = dataset.shuffle(BUFFER_SIZE)

#  Batch the dataset
dataset = dataset.batch(BATCH_SIZE)

#  Use dataset.prefetch() to optimize data loading
dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


# 1. Instantiate the BaseGenerator model
base_generator = BaseGenerator(LATENT_DIM)

# 2. Instantiate the first UpsamplingGenerator (7x7 to 14x14)
upsampling_generator_1 = UpsamplingGenerator(output_channels=1, latent_dim=LATENT_DIM)

# 3. Instantiate the second UpsamplingGenerator (14x14 to 28x28)
upsampling_generator_2 = UpsamplingGenerator(output_channels=1, latent_dim=LATENT_DIM)

# 4. Instantiate the BaseDiscriminator model
base_discriminator = BaseDiscriminator()

# 5. Instantiate the first UpsamplingDiscriminator (for 14x14 scale)
upsampling_discriminator_1 = UpsamplingDiscriminator()

# 6. Instantiate the second UpsamplingDiscriminator (for 28x28 scale)
upsampling_discriminator_2 = UpsamplingDiscriminator()

epochs=5

train_lapgan(dataset, epochs,BATCH_SIZE,LATENT_DIM,base_generator,upsampling_generator_1,upsampling_generator_2,generator_loss,generator_optimizer,base_discriminator,upsampling_discriminator_1,upsampling_discriminator_2,discriminator_loss,discriminator_optimizer)

print("Model inference function `infer_lapgan_image` defined.")

# --- Usage Example ---
print("\n--- Demonstrating inference usage ---")

# 1. Generate a new random latent vector for a single image
new_latent_vector = tf.random.normal([1, LATENT_DIM]) # Generate one image

# 2. Perform inference to generate the image
generated_image_tensor = infer_lapgan_image(new_latent_vector)

# 3. Post-process the image for display:
#    Denormalize from [-1, 1] to [0, 1] and convert to NumPy array.
#    Assuming the output is grayscale (single channel), select the first image in the batch
#    and remove the channel dimension.
display_image = (generated_image_tensor[0, :, :, 0] * 0.5 + 0.5).numpy()

# 4. Display the generated image
plt.figure(figsize=(4, 4))
plt.imshow(display_image, cmap='gray')
plt.axis('off')
plt.title("Generated Image (Inference)")
plt.show()