import tensorflow as tf
import matplotlib.pyplot as plt
from utils import normalize_image

def infer_and_display(model, input_image_path, IMG_WIDTH, IMG_HEIGHT, output_path=None):
    """
    Performs inference on a single image using the generator model
    and displays the input and generated images.
    Args:
        model: The trained generator model.
        input_image_path (str): Path to the input real-world photo.
        output_path (str, optional): Path to save the generated image. Defaults to None.
    """
    # Load and preprocess the input image
    input_image = tf.io.read_file(input_image_path)
    input_image = tf.image.decode_jpeg(input_image, channels=3)
    input_image = tf.image.convert_image_dtype(input_image, tf.float32)
    input_image = tf.image.resize(input_image, [IMG_HEIGHT, IMG_WIDTH],
                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    input_image = normalize_image(input_image) # Normalize to [-1, 1]

    # Add a batch dimension
    input_image = tf.expand_dims(input_image, 0)

    # Generate anime image
    generated_image = model(input_image, training=False)

    # Post-process for display (denormalize to [0, 1])
    display_input = (input_image[0] * 0.5 + 0.5).numpy()
    display_generated = (generated_image[0] * 0.5 + 0.5).numpy()

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Input Photo')
    plt.imshow(display_input)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Generated Anime')
    plt.imshow(display_generated)
    plt.axis('off')
    plt.show()

    if output_path:
        # Convert to uint8 and save
        generated_image_uint8 = tf.image.convert_image_dtype(display_generated, dtype=tf.uint8)
        tf.io.write_file(output_path, tf.image.encode_jpeg(generated_image_uint8))
        print(f"Generated image saved to {output_path}")
