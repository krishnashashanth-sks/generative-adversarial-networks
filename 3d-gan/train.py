import tensorflow as tf
import time

@tf.function
def train_step(images,batch_size,noise_dim,generator,discriminator,generator_loss,discriminator_loss,generator_optimizer,discriminator_optimizer):
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

def train(dataset, epochs,batch_size,noise_dim,generator,discriminator,generator_loss,discriminator_loss,generator_optimizer,discriminator_optimizer):
    for epoch in range(epochs):
        start = time.time()

        gen_losses = []
        disc_losses = []
        for image_batch in dataset:
            g_loss, d_loss = train_step(image_batch,batch_size,noise_dim,generator,discriminator,generator_loss,discriminator_loss,generator_optimizer,discriminator_optimizer)
            gen_losses.append(g_loss)
            disc_losses.append(d_loss)

        print (f'Epoch {epoch + 1},'\
               f' Gen Loss: {tf.reduce_mean(gen_losses)},'\
               f' Disc Loss: {tf.reduce_mean(disc_losses)},'\
               f' Time taken: {time.time()-start:.2f}s')