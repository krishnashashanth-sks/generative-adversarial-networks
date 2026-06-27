import tensorflow as tf

def generate_images(generator, latent_dim, num_images=16):
    noise = tf.random.normal([num_images, latent_dim])
    generated_images = generator(noise, training=False)
    
    # Denormalize images from [-1, 1] to [0, 1] for visualization
    generated_images = (generated_images * 0.5 + 0.5)
    
    print(f"Generated {num_images} images.")
    return generated_images
