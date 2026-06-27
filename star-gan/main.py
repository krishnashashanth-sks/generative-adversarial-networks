from generator import build_generator
from discriminator import build_discriminator
from inference import generate_and_display_images
from train import train_step
from losses import *
from optimizers import *
from dataset import dataset,BATCH_SIZE,NUM_DOMAINS

generator = build_generator(input_shape=(256, 256, 3), num_domains=5)

discriminator = build_discriminator(input_shape=(256, 256, 3), num_domains=5)

# Define loss weights for StarGAN
LAMBDA_CLS = 1.0   # Weight for domain classification loss
LAMBDA_REC = 10.0  # Weight for reconstruction (cycle-consistency) loss
LAMBDA_GP = 10.0   # Weight for gradient penalty (for WGAN-GP or stabilizing LSGAN)
EPOCHS = 10

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    total_g_loss = 0.0
    total_d_loss = 0.0
    num_batches = 0

    for batch_idx, (real_images, real_labels) in enumerate(dataset):
        # Generate random target domain labels for the batch
        target_labels_idx = tf.random.uniform([BATCH_SIZE], minval=0, maxval=NUM_DOMAINS, dtype=tf.int32)
        target_labels = tf.one_hot(target_labels_idx, depth=NUM_DOMAINS)

        # Perform one training step
        metrics = train_step(real_images, real_labels, target_labels,
                             generator, discriminator,
                             generator_optimizer, discriminator_optimizer,
                             discriminator_real_loss,discriminator_fake_loss,
                             classification_loss,
                            generator_adversarial_loss,reconstruction_loss,
                             LAMBDA_CLS, LAMBDA_REC, LAMBDA_GP)

        total_g_loss += metrics['g_total_loss']
        total_d_loss += metrics['d_total_loss']
        num_batches += 1

        if batch_idx % 50 == 0: # Print progress every 50 batches
            print(f"  Batch {batch_idx+1}: G_loss={metrics['g_total_loss']:.4f}, D_loss={metrics['d_total_loss']:.4f}, "
                  f"G_adv={metrics['g_adv_loss']:.4f}, G_cls={metrics['g_cls_loss']:.4f}, G_rec={metrics['g_rec_loss']:.4f}, "
                  f"D_src={metrics['d_src_loss']:.4f}, D_cls={metrics['d_cls_loss']:.4f}, D_gp={metrics['d_gp_loss']:.4f}")

    avg_g_loss = total_g_loss / num_batches
    avg_d_loss = total_d_loss / num_batches
    print(f"Epoch {epoch+1} Summary: Avg G_loss={avg_g_loss:.4f}, Avg D_loss={avg_d_loss:.4f}")

# Get a sample image and its real label from the dataset
# We'll use the first image from the first batch as our test input
for test_img_batch, test_label_batch in dataset.take(1):
    sample_test_image = tf.expand_dims(test_img_batch[0], axis=0) # Take first image and add batch dim
    sample_test_label = tf.expand_dims(test_label_batch[0], axis=0) # Take first label and add batch dim
    break

# Define some target domain indices for translation
# These should be valid indices within your NUM_DOMAINS
# For example, if NUM_DOMAINS is 29, you might pick 3-4 distinct indices.
# Ensure they are different from the original domain of `sample_test_label` for meaningful translation.

# Get the original domain index
original_domain_idx = np.argmax(sample_test_label[0].numpy())

target_domain_indices_to_try = []
# Try to select up to 4 different target domains, excluding the original
for i in range(NUM_DOMAINS):
    if i != original_domain_idx:
        target_domain_indices_to_try.append(i)
    if len(target_domain_indices_to_try) >= 4:
        break

print(f"Selected original image from domain: {original_domain_idx}")
print(f"Will attempt to translate to domains: {target_domain_indices_to_try}")

# --- Call the generation function after training ---
# You can uncomment the line below to also generate images at certain epochs during training
# For now, let's just call it once after the entire training is done.

print("\n--- Generating final sample images after training ---")
generate_and_display_images(
    generator,
    sample_test_image,
    sample_test_label,
    target_domain_indices_to_try,
    epoch='Final' # Label for the epoch argument
)
