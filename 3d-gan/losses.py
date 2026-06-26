import tensorflow as tf 

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False) # set from_logits=True if the discriminator output has no activation

def discriminator_loss(real_output, fake_output):
    # Loss for real images: discriminator should classify them as 1
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    # Loss for fake images: discriminator should classify them as 0
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    # Generator tries to make fake images look real, so it wants the discriminator to output 1 for fake images
    return cross_entropy(tf.ones_like(fake_output), fake_output)