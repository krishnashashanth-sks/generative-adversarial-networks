import tensorflow as tf
from utils import gram_matrix

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real,generated):
  real_loss=loss_object(tf.ones_like(real),real)
  generated_loss=loss_object(tf.zeros_like(generated),generated)
  return real_loss+generated_loss*0.5

def generator_adversarial_loss(generated):
  return loss_object(tf.ones_like(generated),generated)

def content_loss(real_features,generated_features):
  return tf.reduce_mean(tf.abs(real_features - generated_features))

def style_loss(real_style_features,generated_style_features):
  s_loss=0
  for real_feature,generated_feature in zip(real_style_features,generated_style_features):
    real_gram=gram_matrix(real_feature)
    generated_gram=gram_matrix(generated_feature)
    s_loss+=tf.reduce_mean(tf.abs(real_gram-generated_gram))
  return s_loss

def color_reconstrution_loss(input_image,target_image):
  # Create a 3x3 averaging kernel for depthwise convolution
  # Shape: [filter_height, filter_width, in_channels, channel_multiplier]
  # For a 3-channel image, in_channels = 3. For averaging, channel_multiplier = 1.
  blur_kernel = tf.constant(1./9, shape=[3, 3, 3, 1], dtype=tf.float32)
  blurred_input=tf.nn.depthwise_conv2d(input_image,blur_kernel,strides=[1,1,1,1],padding='SAME')
  blurred_generated=tf.nn.depthwise_conv2d(target_image,blur_kernel,strides=[1,1,1,1],padding='SAME')
  return tf.reduce_mean(tf.abs(blurred_input - blurred_generated))

def gray_discriminator_loss(real_gray,generated_gray):
  real_loss=loss_object(tf.ones_like(real_gray),real_gray)
  generated_loss=loss_object(tf.zeros_like(generated_gray),generated_gray)
  return real_loss+generated_loss*0.5

def gray_adversarial_loss(generated_gray):
  return loss_object(tf.ones_like(generated_gray),generated_gray)