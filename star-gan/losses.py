import tensorflow as tf

def discriminator_real_loss(real_output):
    # Discriminator wants to classify real images as 1
    return tf.reduce_mean(tf.square(real_output - 1))

def discriminator_fake_loss(fake_output):
    # Discriminator wants to classify fake images as 0
    return tf.reduce_mean(tf.square(fake_output - 0))

def generator_adversarial_loss(fake_output):
    # Generator wants to classify fake images as 1 (to fool discriminator)
    return tf.reduce_mean(tf.square(fake_output - 1))

def classification_loss(domain_output, target_domain_labels):
    # For discriminator, it's about correctly classifying real images' domains
    # For generator, it's about fooling discriminator to classify fake images into target domain
    return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(target_domain_labels, domain_output))

def reconstruction_loss(original_image, reconstructed_image):
    # Cycle consistency loss (L1 loss)
    return tf.reduce_mean(tf.abs(original_image - reconstructed_image))
