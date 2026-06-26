import time
import os
import matplotlib.pyplot as plt

def generate_and_save_images(model_chain,test_input,epoch,save_dir='./generated_images'):
  base_generated_images=model_chain[0](test_input,training=False)
  scale2_generated_images=model_chain[1]([base_generated_images,test_input],training=False)
  predictions=model_chain[2]([scale2_generated_images,test_input],training=False)
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  fig=plt.figure(figsize=(10,10))
  for i in range(predictions.shape[0]):
    plt.subplot(4,4,i+1)
    plt.imshow(predictions[i,:,:,0]*127.5+127.5,cmap='gray')
    plt.axis('off')
  plt.save_fig(os.path.join(save_dir,'image_at_epoch_{:04d}.png'.format(epoch)))

def infer_lapgan_image(latent_vector):
  """
  Performs inference with the LAPGAN generator chain.

  Args:
    latent_vector: A TensorFlow tensor representing the latent noise vector.
                   Shape should be (batch_size, LATENT_DIM).

  Returns:
    A TensorFlow tensor representing the generated image(s).
    The image pixel values are in the range [-1, 1].
  """
  # Ensure generators are globally accessible as they were instantiated in a previous cell
  # base_generator, upsampling_generator_1, upsampling_generator_2

  base_gen_output = base_generator(latent_vector, training=False)
  scale2_gen_output = upsampling_generator_1([base_gen_output, latent_vector], training=False)
  final_image = upsampling_generator_2([scale2_gen_output, latent_vector], training=False)
  return final_image