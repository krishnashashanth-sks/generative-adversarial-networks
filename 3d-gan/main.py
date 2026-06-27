from generator import make_generator_model
from discriminator import make_discriminator_model
from dataset import dataset,OUTPUT_VOXEL_SHAPE
from losses import *
from optimizers import *
from train import train
from inference import generate_samples
from visualize import visualize_voxel_grid


BUFFER_SIZE = 1000 # From cell bfa63cbf
BATCH_SIZE = 4 # From cell bfa63cbf
LATENT_DIM = 100
NOISE_DIM=100
epochs=10

generator = make_generator_model(LATENT_DIM, OUTPUT_VOXEL_SHAPE)

discriminator = make_discriminator_model(OUTPUT_VOXEL_SHAPE)

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train(dataset, epochs,BATCH_SIZE,NOISE_DIM,generator,discriminator,generator_loss,discriminator_loss,generator_optimizer,discriminator_optimizer)

# Generate a few new samples
num_inference_samples = 4 # You can change this number
generated_samples = generate_samples(generator, num_inference_samples, LATENT_DIM)

print(f"Generated {num_inference_samples} samples with shape: {generated_samples.shape}")

# Visualize the generated samples
for i in range(num_inference_samples):
    print(f"Visualizing Generated Sample {i+1}:")
    visualize_voxel_grid(generated_samples[i], title=f"Generated Sample {i+1}")