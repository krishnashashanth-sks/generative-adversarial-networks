import tensorflow as tf
import time
from inference import generate_and_save_images

@tf.function
def train_step_scale(real_images, batch_size, latent_dim, base_generator, base_discriminator, discriminator_loss, discriminator_optimizer, generator_loss, generator_optimizer):
    noise = tf.random.normal([batch_size, latent_dim])
    
    # --- Train Discriminator ---
    with tf.GradientTape() as disc_tape:
        fake_images = base_generator(noise, training=True)
        down_sampled_real_images = tf.image.resize(
            real_images, method=tf.image.ResizeMethod.BICUBIC
        )
        real_output = base_discriminator(down_sampled_real_images, training=True)
        fake_output = base_discriminator(fake_images, training=True)
        disc_loss = discriminator_loss(real_output, fake_output)
        
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, base_discriminator.trainable_variables
    )
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, base_discriminator.trainable_variables)
    )
    
    # --- Train Generator ---
    with tf.GradientTape() as gen_tape:
        fake_images = base_generator(noise, training=True)
        fake_output = base_discriminator(fake_images, training=True)
        gen_loss = generator_loss(fake_output)
        
    gradients_of_generator = gen_tape.gradient(
        gen_loss, base_generator.trainable_variables
    )
    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, base_generator.trainable_variables)
    )
    
    return disc_loss, gen_loss


@tf.function
def train_step_scale2(real_images, batch_size, latent_dim, base_generator, upsampling_generator_1, upsampling_discriminator_1, discriminator_loss, discriminator_optimizer, generator_loss, generator_optimizer):
    noise = tf.random.normal([batch_size, latent_dim])
    base_fake_images = base_generator(noise, training=False)
    prev_res_real_images = tf.image.resize(
        real_images, method=tf.image.ResizeMethod.BICUBIC
    )
    
    # --- Train Discriminator Scale 2 ---
    with tf.GradientTape() as disc_tape:
        fake_images_scale2 = upsampling_generator_1([base_fake_images, noise], training=True)
        real_images_scale2 = tf.image.resize(
            real_images, method=tf.image.ResizeMethod.BICUBIC
        )
        real_output_scale2 = upsampling_discriminator_1(
            [real_images_scale2, prev_res_real_images], training=True
        )
        fake_output_scale2 = upsampling_discriminator_1(
            [fake_images_scale2, base_fake_images], training=True
        )
        disc_loss_scale2 = discriminator_loss(real_output_scale2, fake_output_scale2)
        
    gradients_of_discriminator_scale2 = disc_tape.gradient(
        disc_loss_scale2, upsampling_discriminator_1.trainable_variables
    )
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator_scale2, upsampling_discriminator_1.trainable_variables)
    )
    
    # --- Train Generator Scale 2 ---
    with tf.GradientTape() as gen_tape:
        fake_images_scale2 = upsampling_generator_1([base_fake_images, noise], training=True)
        fake_output_scale2 = upsampling_discriminator_1([fake_images_scale2, base_fake_images], training=True)
        gen_loss_scale2 = generator_loss(fake_output_scale2)
        
    gradients_of_generator = gen_tape.gradient(
        gen_loss_scale2, upsampling_generator_1.trainable_variables
    )
    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, upsampling_generator_1.trainable_variables)
    )
    
    return disc_loss_scale2, gen_loss_scale2


@tf.function
def train_step_scale3(real_images, batch_size, latent_dim, base_generator, upsampling_generator_1, upsampling_generator_2, upsampling_discriminator_2, discriminator_loss, discriminator_optimizer, generator_loss, generator_optimizer):
    noise = tf.random.normal([batch_size, latent_dim])
    
    # Generate previous scale images without tracking gradients
    base_fake_images = base_generator(noise, training=False)
    fake_images_scale2 = upsampling_generator_1([base_fake_images, noise], training=False)
    
    prev_res_real_images_scale2 = tf.image.resize(
        real_images, method=tf.image.ResizeMethod.BICUBIC
    )
    
    # --- Train Discriminator Scale 3 ---
    with tf.GradientTape() as disc_tape:
        fake_images_scale3 = upsampling_generator_2([fake_images_scale2, noise], training=True)
        
        # Real output expects the full real image and the previous scale real image
        real_output_scale3 = upsampling_discriminator_2(
            [real_images, prev_res_real_images_scale2], training=True
        )
        fake_output_scale3 = upsampling_discriminator_2(
            [fake_images_scale3, fake_images_scale2], training=True
        )
        disc_loss_scale3 = discriminator_loss(real_output_scale3, fake_output_scale3)
        
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss_scale3, upsampling_discriminator_2.trainable_variables
    )
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, upsampling_discriminator_2.trainable_variables)
    )
    
    # --- Train Generator Scale 3 ---
    with tf.GradientTape() as gen_tape:
        fake_images_scale3 = upsampling_generator_2([fake_images_scale2, noise], training=True)
        fake_output_scale3 = upsampling_discriminator_2(
            [fake_images_scale3, fake_images_scale2], training=True
        )
        gen_loss_scale3 = generator_loss(fake_output_scale3)
        
    gradients_of_generator = gen_tape.gradient(
        gen_loss_scale3, upsampling_generator_2.trainable_variables
    )
    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, upsampling_generator_2.trainable_variables)
    )
    
    return disc_loss_scale3, gen_loss_scale3

def train_lapgan(dataset, epochs,batch_size,latent_dim,base_generator,upsampling_generator_1,upsampling_generator_2,generator_loss,generator_optimizer,base_discriminator,upsampling_discriminator_1,upsampling_discriminator_2,discriminator_loss,discriminator_optimizer):
  seed = tf.random.normal([16, latent_dim]) # Corrected 'seef' to 'seed'
  generator_model_chain = [base_generator, upsampling_generator_1, upsampling_generator_2] # Corrected 'base_generated' to 'base_generator'

  for epoch in range(epochs):
    start = time.time()
    print(f"Epoch {epoch+1}/{epochs}")

    # Initialize total losses for each scale
    disc_loss_total_scale1 = tf.constant(0.0)
    gen_loss_total_scale1 = tf.constant(0.0)
    disc_loss_total_scale2 = tf.constant(0.0)
    gen_loss_total_scale2 = tf.constant(0.0)
    disc_loss_total_scale3 = tf.constant(0.0)
    gen_loss_total_scale3 = tf.constant(0.0)
    batch_count = 0

    for image_batch in dataset:
      # Train Scale 1
      disc_loss1, gen_loss1 = train_step_scale(image_batch, batch_size, latent_dim, base_generator, base_discriminator, discriminator_loss, discriminator_optimizer, generator_loss, generator_optimizer) # Corrected 'traing_steo_scale1' to 'train_step_scale1' and 'image_path' to 'image_batch'
      disc_loss_total_scale1 += disc_loss1
      gen_loss_total_scale1 += gen_loss1

      # Train Scale 2
      disc_loss2, gen_loss2 = train_step_scale2(image_batch, batch_size, latent_dim, base_generator, upsampling_generator_1, upsampling_discriminator_1, discriminator_loss, discriminator_optimizer, generator_loss, generator_optimizer)
      disc_loss_total_scale2 += disc_loss2 
      gen_loss_total_scale2 += gen_loss2 

      # Train Scale 3
      disc_loss3, gen_loss3 = train_step_scale3(image_batch, batch_size, latent_dim, base_generator, upsampling_generator_1, upsampling_generator_2, upsampling_discriminator_2, discriminator_loss, discriminator_optimizer, generator_loss, generator_optimizer)
      disc_loss_total_scale3 += disc_loss3 # Corrected '++' to '+='
      gen_loss_total_scale3 += gen_loss3 # Corrected 'gen_loss_total_scale' to 'gen_loss_total_scale3'

      batch_count += 1

    if batch_count > 0:
      avg_disc_loss1 = disc_loss_total_scale1 / batch_count
      avg_gen_loss1 = gen_loss_total_scale1 / batch_count
      avg_disc_loss2 = disc_loss_total_scale2 / batch_count
      avg_gen_loss2 = gen_loss_total_scale2 / batch_count
      avg_disc_loss3 = disc_loss_total_scale3 / batch_count # Corrected to use 'disc_loss_total_scale3'
      avg_gen_loss3 = gen_loss_total_scale3 / batch_count # Added missing average for gen_loss3

      print(f'  Scale 1: Disc Loss: {avg_disc_loss1:.4f}, Gen Loss: {avg_gen_loss1:.4f}')
      print(f'  Scale 2: Disc Loss: {avg_disc_loss2:.4f}, Gen Loss: {avg_gen_loss2:.4f}')
      print(f'  Scale 3: Disc Loss: {avg_disc_loss3:.4f}, Gen Loss: {avg_gen_loss3:.4f}') # Added avg_gen_loss3

    # Generate and save images at the end of the epoch
    generate_and_save_images(generator_model_chain, seed, epoch + 1)
    print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
