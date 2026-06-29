from tensorflow.keras.layers import Input, Dense, LeakyReLU, Lambda,  Conv2D, BatchNormalization, ReLU, Activation, Reshape, concatenate,Flatten
from layers import *
from utils import *
from tensorflo.keras import Model

def build_stage1_generator():
  input_raw_text_embedding = Input(shape=(1024,), name='s1_gen_raw_text_input')
  input_noise = Input(shape=(100,), name='s1_gen_noise_input')

  c_embedding, mls_output = build_ca_network(input_raw_text_embedding) # c_embedding is 128-dim
  combined_input = concatenate([c_embedding, input_noise], axis=-1)

  x = Dense(4*4*256, use_bias=False)(combined_input)
  x = ReLU()(x)
  x = Reshape((4,4,256))(x)

  x = UpSamplingBlock(x, 128) # 8x8
  x = UpSamplingBlock(x, 64)  # 16x16
  x = UpSamplingBlock(x, 32)  # 32x32
  x = UpSamplingBlock(x, 16)  # 64x64
  x = Conv2D(3, kernel_size=(3,3), padding='same', activation='tanh')(x) # Output 64x64x3

  return Model(inputs=[input_raw_text_embedding, input_noise], outputs=[x, c_embedding, mls_output], name='stage1_generator')

def build_stage1_discriminator(image_size=64, ndf=64, conditioning_dim=128):
  input_image = Input(shape=(image_size, image_size, 3), name='s1_disc_input_image')
  input_text_embedding = Input(shape=(conditioning_dim,), name='s1_disc_input_text_embedding') # c_embedding is 128-dim

  # Image pathway
  x = Conv2D(ndf, kernel_size=(4,4), strides=2, padding='same', use_bias=False, kernel_initializer='he_uniform')(input_image) # 64x64 -> 32x32
  x = LeakyReLU(alpha=0.2)(x)

  x = ConvBlock(x, ndf * 2) # 32x32 -> 16x16
  x = ConvBlock(x, ndf * 4) # 16x16 -> 8x8
  image_features_spatial = ConvBlock(x, ndf * 8) # 8x8 -> 4x4. Output (batch_size, 4, 4, ndf*8)

  # Reshape and tile text embedding to match spatial dimensions of image_features (4x4)
  reshaped_text_embedding = Reshape((1, 1, conditioning_dim))(input_text_embedding)
  tiled_text_embedding = Lambda(lambda t: tf.tile(t, [1, image_features_spatial.shape[1], image_features_spatial.shape[2], 1]))(reshaped_text_embedding)

  # Concatenate image features with tiled text embedding
  concat = concatenate([image_features_spatial, tiled_text_embedding], axis=-1)

  # Final classification layers
  x_combined = ConvBlock(concat, ndf * 8, kernel_size=(4,4), strides=1, padding='valid') # 4x4 -> 1x1
  x_flat = Flatten()(x_combined)
  output_prob = Dense(1, activation='sigmoid')(x_flat)

  return Model(inputs=[input_image, input_text_embedding], outputs=[output_prob], name='stage1_discriminator')

def build_adversarial(generator_model, discriminator_model):
  input_raw_text_embedding = Input(shape=(1024,), name='s1_adv_raw_text_input')
  input_noise = Input(shape=(100,), name='s1_adv_noise_input')

  # Generator produces an image, a 128-dim c_embedding, and a 256-dim mls_output
  generated_image, c_embedding_for_discriminator, mls_output_for_kl_loss = generator_model([input_raw_text_embedding, input_noise])

  # Freeze the discriminator's weights when training the generator within this adversarial model
  discriminator_model.trainable = False

  # Discriminator evaluates the generated image conditioned on the text embedding
  probs = discriminator_model([generated_image, c_embedding_for_discriminator])

  # The adversarial model takes text and noise, and outputs the discriminator's probability
  # and the mls_output from CA for KL divergence loss
  adversarial_model = Model(
      inputs=[input_raw_text_embedding, input_noise],
      outputs=[probs, mls_output_for_kl_loss]
  )
  return adversarial_model

def build_stage2_generator():
  input_raw_text_embedding = Input(shape=(1024,), name='s2_gen_raw_text_input')
  c_embedding, mls_output = build_ca_network(input_raw_text_embedding)

  input_images_low_res = Input(shape=(64,64,3), name='s2_gen_low_res_image_input') # Stage-I output is 64x64x3

  # Process low-res image to extract features and integrate C embedding
  x_img = Conv2D(128, kernel_size=(3,3), padding='same', strides=1, use_bias=False, kernel_initializer='he_uniform')(input_images_low_res)
  x_img = BatchNormalization()(x_img)
  x_img = ReLU()(x_img)

  x_img = Conv2D(256, kernel_size=(3,3), padding='same', strides=1, use_bias=False, kernel_initializer='he_uniform')(x_img)
  x_img = BatchNormalization()(x_img)
  x_img = ReLU()(x_img) # This becomes 64x64x256

  # Tile c_embedding (128-dim) to match image spatial dimensions (64x64)
  c_tiled = Reshape((1, 1, 128))(c_embedding)
  c_tiled = Lambda(lambda t: tf.tile(t, [1, x_img.shape[1], x_img.shape[2], 1]))(c_tiled)

  # Concatenate processed image features with tiled conditional embedding
  x = concatenate([x_img, c_tiled], axis=-1)
  # Now x is 64x64x(256+128) = 64x64x384

  # Residual blocks (adjust input channels if needed, here it's implicitly handled)
  x = residual_block(x, num_kernels=384) # num_kernels must match input channels for add
  x = residual_block(x, num_kernels=384)
  x = residual_block(x, num_kernels=384)
  x = residual_block(x, num_kernels=384)

  # Upsampling blocks to go from 64x64 to 256x256
  x = UpSamplingBlock(x, 256) # 64x64 -> 128x128
  x = UpSamplingBlock(x, 128) # 128x128 -> 256x256

  # Final convolution to output 3 channels (for RGB image)
  x = Conv2D(3, kernel_size=(3,3), padding='same', use_bias=False, kernel_initializer="he_uniform")(x)
  x = Activation('tanh')(x) # Output range [-1, 1]

  return Model(inputs=[input_raw_text_embedding, input_images_low_res], outputs=[x, c_embedding, mls_output], name='stage2_generator')

def build_stage2_discriminator(image_size=256, ndf=64, conditioning_dim=128):
  input_image = Input(shape=(image_size, image_size, 3), name='s2_disc_input_image')
  input_text_embedding = Input(shape=(conditioning_dim,), name='s2_disc_input_text_embedding') # c_embedding is 128-dim

  # Image pathway
  x = ConvBlock(input_image, ndf, strides=2) # 256x256 -> 128x128
  x = ConvBlock(x, ndf * 2, strides=2) # 128x128 -> 64x64
  x = ConvBlock(x, ndf * 4, strides=2) # 64x64 -> 32x32
  x = ConvBlock(x, ndf * 8, strides=2) # 32x32 -> 16x16
  x = ConvBlock(x, ndf * 16, strides=2) # 16x16 -> 8x8
  image_features_spatial = ConvBlock(x, ndf * 32, strides=2) # 8x8 -> 4x4. Output (batch_size, 4, 4, ndf*32)

  # Reshape and tile text embedding to match spatial dimensions of image_features (4x4)
  reshaped_text_embedding = Reshape((1, 1, conditioning_dim))(input_text_embedding)
  tiled_text_embedding = Lambda(lambda t: tf.tile(t, [1, image_features_spatial.shape[1], image_features_spatial.shape[2], 1]))(reshaped_text_embedding)

  # Concatenate image features with tiled text embedding
  concat = concatenate([image_features_spatial, tiled_text_embedding], axis=-1)

  # Final classification layers
  x_combined = ConvBlock(concat, ndf * 32, kernel_size=(4,4), strides=1, padding='valid') # 4x4 -> 1x1
  x_flat = Flatten()(x_combined)
  output_prob = Dense(1, activation='sigmoid')(x_flat)

  return Model(inputs=[input_image, input_text_embedding], outputs=[output_prob], name='stage2_discriminator')

def build_stage2_adversarial(stage2_dis_model, stage2_gen_model, stage1_gen_model):
  input_raw_text_embedding = Input(shape=(1024,), name='s2_adv_raw_text_input')
  input_noise = Input(shape=(100,), name='s2_adv_noise_input')

  # Stage-I Generator: produces low-res image, c_embedding, mls
  low_res_image_from_s1, c_embedding_s1, mls_s1 = stage1_gen_model([input_raw_text_embedding, input_noise])

  # Stage-II Generator: produces high-res image, c_embedding, mls
  high_res_image_from_s2, c_embedding_s2, mls_s2 = stage2_gen_model([input_raw_text_embedding, low_res_image_from_s1])

  # Freeze discriminator's weights for adversarial training
  stage2_dis_model.trainable = False
  stage1_gen_model.trainable = False # Freeze S1 Gen as well, S2 Gen builds on its output

  # Discriminator evaluates the high-res image from S2 G, conditioned on c_embedding_s2
  probability = stage2_dis_model([high_res_image_from_s2, c_embedding_s2])

  # Adversarial model outputs D's probability and G2's mls for KL loss
  adversarial_model = Model(
      inputs=[input_raw_text_embedding, input_noise],
      outputs=[probability, mls_s2],
      name='stage2_adversarial_model'
  )
  return adversarial_model

