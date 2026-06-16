import tensorflow as tf
import os
import matplotlib.pyplot as plt

IMG_WIDTH = 256
IMG_HEIGHT = 256

def load_image(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image

def normalize_image(image):
    image = (image / 127.5) - 1  # Normalize to [-1, 1]
    return image

def preprocess_image(image_path):
    # This function is for direct file loading, which we are not using with TFDS
    # It's kept for reference if you decide to use local files later.
    image = load_image(image_path)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = normalize_image(image)
    return image

def preprocess_image_tfds(image, label):
    # This runs on-the-fly, keeping RAM usage flat
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    return (image / 127.5) - 1  # Normalize to [-1, 1] for GANs

def get_vgg_feature_extractor():
  vgg=tf.keras.applications.VGG19(include_top=False,weights='imagenet')
  vgg.trainable=False
  content_layers=[
      'block5_conv2'
  ]
  style_layers=[
      'block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1'
  ]
  outputs=[
      vgg.get_layer(name).output for name in content_layers+style_layers
  ]
  return tf.keras.Model([vgg.input],outputs)

def gram_matrix(input_tensor):
  result=tf.linalg.einsum('bijc,bijd->bcd',input_tensor,input_tensor)
  input_shape=tf.shape(input_tensor)
  num_locations=tf.cast(input_shape[1]*input_shape[2],tf.float32)
  return result/num_locations

def generate_and_save_images(model, test_input, epoch, output_dir='output_images'):
    # Make sure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Ensure test_input is a dataset and take one batch
    if isinstance(test_input, tf.data.Dataset):
        for batch in test_input.take(1):
            test_input = batch[0] # Assuming real_photo is the first element
            break

    # The training=False flag is important for batchnorm and dropout layers
    prediction = model(test_input, training=False)

    plt.figure(figsize=(15, 7))

    display_list = [test_input[0], prediction[0]]
    title = ['Input Photo', 'Generated Anime']

    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.title(title[i])
        # Getting the pixel values in the [-1, 1] range to plot.
        plt.imshow((display_list[i] * 0.5 + 0.5).numpy())
        plt.axis('off')

    # Save the figure
    image_path = os.path.join(output_dir, f'image_at_epoch_{epoch:04d}.png')
    plt.savefig(image_path)
    plt.show()
    print(f"Generated image saved to {image_path}")
