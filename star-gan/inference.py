import matplotlib.pyplot as plt
import numpy as np

def generate_and_display_images(generator, test_input_image, test_real_label, target_domain_indices, epoch, num_domains=NUM_DOMAINS):
    # test_input_image: A single image tensor (batch size 1)
    # test_real_label: A single one-hot encoded label for the test_input_image
    # target_domain_indices: A list of integer indices for target domains to translate to

    fig, axes = plt.subplots(1, len(target_domain_indices) + 1, figsize=(20, 5))

    # Display original image
    original_image_display = (test_input_image[0].numpy() + 1) / 2.0 # Denormalize to [0, 1]
    axes[0].imshow(original_image_display)
    axes[0].set_title(f"Original (Domain {np.argmax(test_real_label[0])})")
    axes[0].axis('off')

    for i, target_idx in enumerate(target_domain_indices):
        # Create one-hot encoded target label
        target_label_one_hot = tf.one_hot([target_idx], depth=num_domains)

        # Generate fake image
        generated_image = generator([test_input_image, target_label_one_hot], training=False)
        generated_image_display = (generated_image[0].numpy() + 1) / 2.0 # Denormalize to [0, 1]

        axes[i+1].imshow(generated_image_display)
        axes[i+1].set_title(f"To Domain {target_idx}")
        axes[i+1].axis('off')

    plt.suptitle(f"Epoch {epoch} Sample Generations", fontsize=16)
    plt.show()
