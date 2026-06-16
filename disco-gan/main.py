import torch.optim as optim
import torch.nn as nn
from generator import Generator
from discriminator import Discriminator
import torch
from dataset import source_dataloader, target_dataloader, source_dataset, target_dataset
from train import train
from inference import translate_image,denormalize_image
import matplotlib.pyplot as plt

# Initialize models (assuming 3 input/output channels for color images)
# For MNIST, input_nc should be 1, but for SVHN, it's 3. Let's assume 3 for a general case.
# We will instantiate with 3 channels and later adapt if necessary.
gen_S2T = Generator(input_nc=3, output_nc=3)
gen_T2S = Generator(input_nc=3, output_nc=3)
disc_S = Discriminator(input_nc=3)
disc_T = Discriminator(input_nc=3)

# Optimizers
# Assuming default learning rates for now
lr = 0.0002
beta1 = 0.5
betal2 = 0.999

gen_optimizer = optim.Adam(list(gen_S2T.parameters()) + list(gen_T2S.parameters()), lr=lr, betas=(beta1, betal2))
disc_S_optimizer = optim.Adam(disc_S.parameters(), lr=lr, betas=(beta1, betal2))
disc_T_optimizer = optim.Adam(disc_T.parameters(), lr=lr, betas=(beta1, betal2))

print("Loss functions and optimizers defined and models initialized.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gen_S2T.to(device)
gen_T2S.to(device)
disc_S.to(device)
disc_T.to(device)

num_epochs = 1
lambda_cycle = 10.0
lambda_identity = 0.5

print(f"Using device: {device}")
print(f"Models moved to {device}.")
print(f"Training parameters: num_epochs={num_epochs}, lambda_cycle={lambda_cycle}, lambda_identity={lambda_identity}")

losses = train(num_epochs,source_dataloader,target_dataloader,gen_optimizer,gen_S2T,gen_T2S,disc_T,disc_S_optimizer,disc_T_optimizer,lambda_cycle,lambda_identity,device)


# Plotting the losses
plt.figure(figsize=(15, 5))

# Plot Generator Adversarial Loss
plt.subplot(1, 3, 1)
plt.plot(losses['G_adv'], label='Generator Adversarial Loss')
plt.title('Generator Adversarial Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()

# Plot Generator Cycle and Identity Losses
plt.subplot(1, 3, 2)
plt.plot(losses['G_cycle'], label='Generator Cycle Loss')
plt.plot(losses['G_identity'], label='Generator Identity Loss')
plt.title('Generator Cycle & Identity Losses')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()

# Plot Discriminator Losses
plt.subplot(1, 3, 3)
plt.plot(losses['D_S'], label='Discriminator Source Loss')
plt.plot(losses['D_T'], label='Discriminator Target Loss')
plt.title('Discriminator Losses')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

print("Loss visualization complete.")

# --- Usage Example ---
print("\n--- Demonstrating Usage of Inference Function ---")

# Take one sample image from each dataset for demonstration
# Ensure the dataset indices are valid
if len(source_dataset) > 0:
    sample_s_original, _ = source_dataset[0] # Get one image from source dataset
else:
    print("Source dataset is empty, cannot get sample for inference.")
    sample_s_original = None

if len(target_dataset) > 0:
    sample_t_original, _ = target_dataset[0] # Get one image from target dataset
else:
    print("Target dataset is empty, cannot get sample for inference.")
    sample_t_original = None

if sample_s_original is not None and sample_t_original is not None:
    # Perform S -> T -> S cycle
    fake_T_from_S = translate_image(sample_s_original, gen_S2T, device)
    cycled_S_from_T = translate_image(fake_T_from_S, gen_T2S, device)

    # Perform T -> S -> T cycle
    fake_S_from_T = translate_image(sample_t_original, gen_T2S, device)
    cycled_T_from_S = translate_image(fake_S_from_T, gen_S2T, device)

    # Prepare images for display
    # Source Cycle: Original S, Fake T, Cycled S
    source_cycle_images = [sample_s_original, fake_T_from_S, cycled_S_from_T]
    # Target Cycle: Original T, Fake S, Cycled T
    target_cycle_images = [sample_t_original, fake_S_from_T, cycled_T_from_S]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    titles = [
        "Original Source", "Source -> Fake Target", "Source -> Fake Target -> Cycled Source",
        "Original Target", "Target -> Fake Source", "Target -> Fake Source -> Cycled Target"
    ]

    all_images_to_display = source_cycle_images + target_cycle_images

    for i, ax in enumerate(axes.flat):
        if i < len(all_images_to_display):
            img_to_show = denormalize_image(all_images_to_display[i]).permute(1, 2, 0)
            ax.imshow(img_to_show)
            ax.set_title(titles[i])
            ax.axis('off')

    plt.suptitle("GAN Inference Demonstration: Image Translation Cycles")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
else:
    print("Skipping inference demonstration due to empty datasets.")

print("Inference usage demonstration complete.")