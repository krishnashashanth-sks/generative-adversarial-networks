from utils import gradient_penalty

import tensorflow as tf

@tf.function
def train_step(real_images, real_labels, target_labels,
               generator, discriminator,
               generator_optimizer, discriminator_optimizer,
               discriminator_real_loss,discriminator_fake_loss,classification_loss,
               generator_adversarial_loss,reconstruction_loss,
               lambda_cls, lambda_rec, lambda_gp):

    # Step 1: Train Discriminator
    with tf.GradientTape() as disc_tape:
        # Generate fake images for target domain
        fake_images = generator([real_images, target_labels], training=True)

        # Discriminator predictions for real images and generated fake images
        real_src_output, real_cls_output = discriminator(real_images, training=True)
        fake_src_output, _ = discriminator(fake_images, training=True) # Classification for fake images not used for D_loss, but for G_loss later

        # Discriminator Source (real/fake) Loss (LSGAN)
        disc_real_src_loss = discriminator_real_loss(real_src_output)
        disc_fake_src_loss = discriminator_fake_loss(fake_src_output)
        disc_src_loss = disc_real_src_loss + disc_fake_src_loss

        # Discriminator Classification Loss (on real images to classify their true domain)
        disc_cls_loss = classification_loss(real_cls_output, real_labels)

        # Gradient Penalty
        gp_loss = gradient_penalty(discriminator, real_images, fake_images)

        # Total Discriminator Loss
        disc_total_loss = disc_src_loss + lambda_cls * disc_cls_loss + lambda_gp * gp_loss

    # Apply gradients to Discriminator
    disc_gradients = disc_tape.gradient(disc_total_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

    # Step 2: Train Generator
    with tf.GradientTape() as gen_tape:
        # Generate fake images for target domain
        fake_images = generator([real_images, target_labels], training=True)

        # Reconstruct original images from fake images using original domain labels
        reconstructed_images = generator([fake_images, real_labels], training=True)

        # Discriminator predictions for fake images (for generator's objective)
        fake_src_output, fake_cls_output = discriminator(fake_images, training=True)

        # Generator Adversarial Loss (Generator wants D to classify fake as real)
        gen_adv_loss = generator_adversarial_loss(fake_src_output)

        # Generator Classification Loss (Generator wants D to classify fake images into target_labels)
        gen_cls_loss = classification_loss(fake_cls_output, target_labels)

        # Reconstruction Loss (Cycle Consistency: original -> target -> original)
        gen_rec_loss = reconstruction_loss(real_images, reconstructed_images)

        # Total Generator Loss
        gen_total_loss = gen_adv_loss + lambda_cls * gen_cls_loss + lambda_rec * gen_rec_loss

    # Apply gradients to Generator
    gen_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))

    return {
        'd_src_loss': disc_src_loss,
        'd_cls_loss': disc_cls_loss,
        'd_gp_loss': gp_loss,
        'd_total_loss': disc_total_loss,
        'g_adv_loss': gen_adv_loss,
        'g_cls_loss': gen_cls_loss,
        'g_rec_loss': gen_rec_loss,
        'g_total_loss': gen_total_loss
    }
