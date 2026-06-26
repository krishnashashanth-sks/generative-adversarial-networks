from generator import make_generator_model
from discriminator import make_discriminator_model

LATENT_DIM = 100
OUTPUT_VOXEL_SHAPE = (64, 64, 64, 1) # 64x64x64 voxel grid, 1 channel (binary)
generator = make_generator_model(LATENT_DIM, OUTPUT_VOXEL_SHAPE)

discriminator = make_discriminator_model(OUTPUT_VOXEL_SHAPE)
