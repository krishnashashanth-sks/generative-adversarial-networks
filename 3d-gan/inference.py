import tensorflow as tf

def generate_samples(generator_model, num_samples, latent_dim):
    """
    Generates new 3D voxel grids using the generator model.
    """
    noise = tf.random.normal([num_samples, latent_dim])
    generated_voxel_grids = generator_model(noise, training=False)
    return generated_voxel_grids