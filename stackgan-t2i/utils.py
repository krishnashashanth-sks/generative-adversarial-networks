from tensorflow.keras.layers import Input, Dense, LeakyReLU, Lambda,  Conv2D, BatchNormalization, ReLU, add, UpSampling2D
import tensorflow as tf
import pickle 
import numpy as np
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras import Model

def conditioning_augmentation(x):
  mean = x[:, :128]
  log_sigma = x[:, 128:]
  stddev = tf.math.exp(log_sigma)
  epsilon = tf.random.normal(shape=tf.shape(mean), dtype=tf.float32)
  c = mean + stddev * epsilon
  return c

def build_ca_network(input_tensor):
  # input_tensor is (batch_size, 1024)
  mls = Dense(256)(input_tensor) # Output (batch_size, 256) (mean+log_sigma)
  mls = LeakyReLU(alpha=0.2)(mls)
  c = Lambda(conditioning_augmentation)(mls) # Output (batch_size, 128)
  return c, mls # c is the sampled conditioning vector, mls for KL loss

def UpSamplingBlock(x, num_kernels):
  x = UpSampling2D(size=(2,2))(x)
  x = Conv2D(num_kernels, (3,3), padding='same', strides=1, use_bias=False, kernel_initializer='he_uniform')(x)
  x = BatchNormalization(gamma_initializer='ones', beta_initializer='zeros')(x)
  x = ReLU()(x)
  return x

def ConvBlock(x, num_kernels, kernel_size=(4,4), strides=2, padding='same', activation=True):
  x = Conv2D(num_kernels, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False, kernel_initializer='he_uniform')(x)
  x = BatchNormalization(gamma_initializer='ones', beta_initializer='zeros')(x)
  if activation:
    x = LeakyReLU(alpha=0.2)(x)
  return x

def residual_block(input_tensor, num_kernels=512):
  # Path 1
  x = Conv2D(num_kernels, kernel_size=(3,3), padding='same', use_bias=False, kernel_initializer='he_uniform')(input_tensor)
  x = BatchNormalization(gamma_initializer='ones', beta_initializer='zeros')(x)
  x = ReLU()(x)
  # Path 2
  x = Conv2D(num_kernels, kernel_size=(3,3), padding='same', use_bias=False, kernel_initializer="he_uniform")(x)
  x = BatchNormalization(gamma_initializer="ones", beta_initializer='zeros')(x)
  # Add shortcut connection
  x = add([x, input_tensor])
  x = ReLU()(x) # Final activation after addition
  return x

def build_embedding_compressor():
  input_layer1=Input(shape=(1024,))
  x=Dense(128)(input_layer1)
  x=ReLU()(x)
  model=Model(inputs=[input_layer1],outputs=[x])
  return model

def adversarial_loss(y_true, y_pred):
  # y_pred will be the mls_output (256-dim tensor from CA)
  # The loss is for KL divergence between N(mean, log_sigma) and N(0, I)
  mean = y_pred[:, :128]
  log_sigma = y_pred[:, 128:]
  loss = -log_sigma + 0.5 * (-1 + tf.math.exp(2.0 * log_sigma) + tf.math.square(mean))
  loss = tf.reduce_mean(loss)
  return loss

def load_class_ids_filenames(class_id_path, filename_path):
  with open(class_id_path, 'rb') as file:
    class_id = pickle.load(file, encoding='latin1')
  with open(filename_path, 'rb') as file:
    filename = pickle.load(file, encoding='latin1')
  return class_id, filename

def load_text_embedding(text_embeddings_path):
  with open(text_embeddings_path, 'rb') as file:
    embeds = pickle.load(file, encoding='latin1')
    embeds = np.array(embeds)
  return embeds

def load_bbox(data_path):
  bbox_file = os.path.join(data_path, "bounding_boxes.txt")
  images_file = os.path.join(data_path, "images.txt")

  if not os.path.exists(bbox_file):
    print(f"Warning: Bounding box file not found at {bbox_file}. Returning empty dictionary.")
    return {}
  if not os.path.exists(images_file):
    print(f"Warning: Images file not found at {images_file}. Returning empty dictionary.")
    return {}

  bbox_df_with_names = pd.read_csv(bbox_file, delim_whitespace=True, header=None, names=['filename', 'x', 'y', 'width', 'height'])
  filename_df = pd.read_csv(images_file, delim_whitespace=True, header=None)

  bbox_dict = {}
  for idx, row in filename_df.iterrows():
      img_id = row[0]
      filename = row[1]
      bbox_info = bbox_df_with_names[bbox_df_with_names['filename'] == filename.replace('.jpg', '')]
      if not bbox_info.empty:
          bbox_dict[filename.replace('.jpg', '')] = bbox_info[['x', 'y', 'width', 'height']].iloc[0].tolist()
      else:
          # Fallback: if filename with extension not found, try without it
          bbox_info = bbox_df_with_names[bbox_df_with_names['filename'] == filename]
          if not bbox_info.empty:
              bbox_dict[filename] = bbox_info[['x', 'y', 'width', 'height']].iloc[0].tolist()
          else:
              pass # Bounding box not found, it will be skipped by load_data

  return bbox_dict

def load_images(image_path, bounding_box, size):
  try:
    image = Image.open(image_path).convert('RGB')
  except FileNotFoundError:
    print(f"Image file not found: {image_path}. Skipping.")
    return None
  except Exception as e:
    print(f"Error opening image {image_path}: {e}. Skipping.")
    return None

  w,h = image.size
  if bounding_box is not None:
    x_min, y_min, box_width, box_height = bounding_box

    x1 = int(x_min)
    y1 = int(y_min)
    x2 = int(x_min + box_width)
    y2 = int(y_min + box_height)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 > x1 and y2 > y1:
      image = image.crop((x1, y1, x2, y2))
    else:
      print(f"Warning: Invalid bounding box for {image_path}. Skipping crop.")

  image = image.resize((size, size), Image.LANCZOS)
  image = np.array(image)
  return image

def load_data(class_id_path, filename_path, embeddings_path, size, dataset_root_path):
  class_id, filenames = load_class_ids_filenames(class_id_path, filename_path)
  embeddings = load_text_embedding(embeddings_path)
  bbox_dict = load_bbox(dataset_root_path)

  x, y, embeds = [], [], []

  for i, filename in enumerate(filenames):
    # Ensure filename is processed correctly for bbox lookup
    processed_filename = filename.replace('.jpg', '') # Match the key format in bbox_dict
    if processed_filename not in bbox_dict:
        # print(f'Warning: Bounding box for {filename} not found. Skipping image.')
        continue

    bbox = bbox_dict[processed_filename]

    try:
      image_full_path = f'{dataset_root_path}/images/{filename}.jpg'
      image = load_images(image_full_path, bbox, size)
      if image is None: # Skip if image loading failed
          continue

      # Assuming embeddings are indexed by file. If there are multiple captions per image,
      # a more complex mapping might be needed. For now, use the `i` index.
      e = embeddings[i, :, :]
      embed_index = np.random.randint(0, e.shape[0])
      embed = e[embed_index, :]
      x.append(np.array(image))
      y.append(class_id[i])
      embeds.append(embed)
    except Exception as e:
      print(f'Error processing {filename}: {e}')

  if len(x) == 0:
    print("No images loaded. Returning empty arrays.")
    return np.array([]), np.array([]), np.array([])

  x = np.array(x)
  y = np.array(y)
  embeds = np.array(embeds)
  return x, y, embeds

def save_image(file,save_path):
  plt.imsave(save_path, file)
