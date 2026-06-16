import tensorflow as tf
from tqdm.auto import tqdm
from utils import generate_and_save_images

@tf.function
def train_step(real_photo,real_anime,generator,discriminator,gray_discriminator,generator_adversarial_loss,gray_adversarial_loss,vgg_feature_extractor,content_loss,style_loss,color_reconstrution_loss
               ,LAMBDA_ADVERSARIAL,LAMBDA_GRAY_ADVERSARIAL,LAMBDA_CONTENT,LAMBDA_STYLE,LAMBDA_COLOR,discriminator_loss,gray_discriminator_loss,generator_optimizer,discriminator_optimizer,gray_discriminator_optimizer):
  with tf.GradientTape(persistent=True) as tape:
    fake_anime=generator(real_photo,training=True)
    real_photo_gray=tf.image.rgb_to_grayscale(real_photo)
    fake_anime_gray=tf.image.rgb_to_grayscale(fake_anime)
    real_anime_gray=tf.image.rgb_to_grayscale(real_anime)
    disc_real_output=discriminator([real_photo,real_anime],training=True)
    disc_fake_output=discriminator([real_photo,fake_anime],training=True)
    gray_disc_real_output=gray_discriminator(real_anime_gray,training=True)
    gray_disc_fake_output=gray_discriminator(fake_anime_gray,training=True)
    gen_adv_loss=generator_adversarial_loss(disc_fake_output)
    gen_gray_adv_loss=gray_adversarial_loss(gray_disc_fake_output)
    real_photo_vgg_features=vgg_feature_extractor(real_photo)
    fake_anime_vgg_features=vgg_feature_extractor(fake_anime)
    content_features_real=real_photo_vgg_features[0]
    content_features_fake=fake_anime_vgg_features[0]
    style_features_real=real_photo_vgg_features[1:]
    style_features_fake=fake_anime_vgg_features[1:]
    perc_loss=content_loss(content_features_real,content_features_fake)
    style_l=style_loss(style_features_real,style_features_fake)
    color_rec_loss=color_reconstrution_loss(real_photo,fake_anime)
    total_generator_loss=(
        gen_adv_loss*LAMBDA_ADVERSARIAL+
        gen_gray_adv_loss*LAMBDA_GRAY_ADVERSARIAL+
        perc_loss*LAMBDA_CONTENT+
        style_l*LAMBDA_STYLE+
        color_rec_loss*LAMBDA_COLOR
    )
    disc_loss=discriminator_loss(disc_real_output,disc_fake_output)
    gray_disc_loss=gray_discriminator_loss(gray_disc_real_output,gray_disc_fake_output)
  generator_gradients=tape.gradient(total_generator_loss,generator.trainable_variables)
  discriminator_gradients=tape.gradient(disc_loss,discriminator.trainable_variables)
  gray_discriminator_gradients=tape.gradient(gray_disc_loss,gray_discriminator.trainable_variables)
  generator_optimizer.apply_gradients(zip(generator_gradients,generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,discriminator.trainable_variables))
  gray_discriminator_optimizer.apply_gradients(zip(gray_discriminator_gradients,gray_discriminator.trainable_variables))
  return total_generator_loss, disc_loss, gray_disc_loss, gen_adv_loss, gen_gray_adv_loss, perc_loss, style_l, color_rec_loss


def train(dataset, epochs,generator,discriminator,gray_discriminator,generator_adversarial_loss,gray_adversarial_loss,vgg_feature_extractor,content_loss,style_loss,color_reconstrution_loss
          ,LAMBDA_ADVERSARIAL,LAMBDA_GRAY_ADVERSARIAL,LAMBDA_CONTENT,LAMBDA_STYLE,LAMBDA_COLOR,discriminator_loss,gray_discriminator_loss,generator_optimizer,discriminator_optimizer,gray_discriminator_optimizer):
    # Take one sample for generating images during training to observe progress
    # We need to ensure fixed_test_input is a batch with the correct dimension
    for example_real_photo, example_anime in dataset.take(1):
        fixed_test_input = example_real_photo # Use the real photo as fixed test input
        break

    print("Starting training...")

    for epoch in tqdm(range(epochs)):
        start = tf.timestamp()

        total_gen_loss = 0
        total_disc_loss = 0
        total_gray_disc_loss = 0
        # Initialize variables for individual generator loss components
        total_gen_adv_loss = 0
        total_gen_gray_adv_loss = 0
        total_perc_loss = 0
        total_style_loss = 0
        total_color_rec_loss = 0

        num_batches = 0

        # Iterate over the dataset
        for real_photo, real_anime in tqdm(dataset):
            # Capture all returned loss components from train_step
            gen_loss, disc_l, gray_disc_l, gen_adv_l, gen_gray_adv_l, perc_l, style_l, color_rec_l = train_step(real_photo, real_anime,generator,discriminator,gray_discriminator,generator_adversarial_loss,gray_adversarial_loss,vgg_feature_extractor,content_loss,style_loss,color_reconstrution_loss
            ,LAMBDA_ADVERSARIAL,LAMBDA_GRAY_ADVERSARIAL,LAMBDA_CONTENT,LAMBDA_STYLE,LAMBDA_COLOR,discriminator_loss,gray_discriminator_loss,generator_optimizer,discriminator_optimizer,gray_discriminator_optimizer)

            total_gen_loss += gen_loss
            total_disc_loss += disc_l
            total_gray_disc_loss += gray_disc_l
            # Accumulate individual generator loss components
            total_gen_adv_loss += gen_adv_l
            total_gen_gray_adv_loss += gen_gray_adv_l
            total_perc_loss += perc_l
            total_style_loss += style_l
            total_color_rec_loss += color_rec_l

            num_batches += 1

            if num_batches % 100 == 0:
                print(f'. Batch {num_batches}, Gen Total Loss: {gen_loss:.4f}, Disc Loss: {disc_l:.4f}, Gray Disc Loss: {gray_disc_l:.4f}')
                print(f'  Gen Components - Adv: {gen_adv_l:.4f}, Gray Adv: {gen_gray_adv_l:.4f}, Perc: {perc_l:.4f}, Style: {style_l:.4f}, Color: {color_rec_l:.4f}')

        print(f'Epoch {epoch + 1} finished.')
        if num_batches > 0:
            print(f'Average Generator Total Loss: {total_gen_loss / num_batches:.4f}')
            print(f'Average Discriminator Loss: {total_disc_loss / num_batches:.4f}')
            print(f'Average Gray Discriminator Loss: {total_gray_disc_loss / num_batches:.4f}')
            # Print average individual generator loss components
            print(f'Average Gen Components - Adv: {total_gen_adv_loss / num_batches:.4f}')
            print(f'Average Gen Components - Gray Adv: {total_gen_gray_adv_loss / num_batches:.4f}')
            print(f'Average Gen Components - Perceptual: {total_perc_loss / num_batches:.4f}')
            print(f'Average Gen Components - Style: {total_style_loss / num_batches:.4f}')
            print(f'Average Gen Components - Color Reconstruction: {total_color_rec_loss / num_batches:.4f}')
        else:
            print("No batches processed in this epoch.")

        # Generate and save images every few epochs or at the end of each epoch
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            generate_and_save_images(generator, fixed_test_input, epoch + 1)

        print(f'Time taken for epoch {epoch + 1} is {tf.timestamp() - start:.2f} sec\n')
