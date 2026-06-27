import tensorflow as tf

def reflect_pad(x, padding_size):
  # Correctly pads the input tensor x with reflection padding
  return tf.pad(x, [[0, 0], [padding_size, padding_size], [padding_size, padding_size], [0, 0]], 'REFLECT')

# Helper function for gradient penalty (common in WGAN-GP to ensure Lipschitz constraint)
@tf.function
def gradient_penalty(discriminator, real_images, fake_images):
    alpha = tf.random.uniform(shape=[real_images.shape[0], 1, 1, 1], minval=0.0, maxval=1.0)
    interpolated_images = real_images * alpha + fake_images * (1 - alpha)

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated_images)
        # 1. Get the discriminator output for this interpolated image (only source output needed for GP).
        d_interpolated, _ = discriminator(interpolated_images, training=True)

    # 2. Calculate the gradients w.r.t to this interpolated image.
    gradients = gp_tape.gradient(d_interpolated, interpolated_images)

    # Handle cases where gradients might be None (e.g., if interpolated_images don't contribute to d_interpolated)
    if gradients is None:
        return 0.0 # Or raise an error, depending on desired behavior

    # 3. Calculate the norm of the gradients.
    norm = tf.norm(tf.reshape(gradients, [gradients.shape[0], -1]), axis=1)

    # 4. Calculate the gradient penalty.
    gp = tf.reduce_mean((norm - 1.0)**2)
    return gp

