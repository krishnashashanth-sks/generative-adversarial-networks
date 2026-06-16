import tensorflow as tf

@tf.function
def train_step(input_image, target, epoch,generator,discriminator,generator_loss,discriminator_loss,generator_optimizer,discriminator_optimizer):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))


def fit(dataset, epochs,generator,discriminator,generator_loss,discriminator_loss,generator_optimizer,discriminator_optimizer):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        # Train
        for n, (input_image, target) in enumerate(dataset):
            train_step(input_image, target, epoch,generator,discriminator,generator_loss,discriminator_loss,generator_optimizer,discriminator_optimizer)

        print(f"Epoch {epoch+1} completed.")

print("Training step and fit functions defined.")