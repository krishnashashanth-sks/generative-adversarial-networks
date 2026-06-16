from model import VQGAN
import torch
from dataset import train_dataset
from losses import perceptual_loss_fn,recon_loss_fn, adv_loss_fn
import matplotlib.pyplot as plt
from utils import unnormalize
from torch.utils.data import DataLoader

vqgan_model = VQGAN(in_channels=3, num_hiddens=128, num_residual_layers=2, num_residual_hiddens=64,
                    num_embeddings=512, embedding_dim=64, commitment_cost=0.25)

optimizer_g = torch.optim.Adam(vqgan_model.encoder.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizer_g.add_param_group({'params': vqgan_model.pre_quant_conv.parameters()})
optimizer_g.add_param_group({'params': vqgan_model.quantizer.parameters()})
optimizer_g.add_param_group({'params': vqgan_model.post_quant_conv.parameters()})
optimizer_g.add_param_group({'params': vqgan_model.decoder.parameters()})

optimizer_d = torch.optim.Adam(vqgan_model.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))

batch_size = 64
num_workers = 2 # Adjust based on your system's capabilities

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

print(f"DataLoader created with batch size {batch_size} and {num_workers} workers.")
print(f"Number of batches in training loader: {len(train_loader)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vqgan_model.to(device)
perceptual_loss_fn.to(device)
recon_loss_fn.to(device)
adv_loss_fn.to(device)

print(f"Models and loss functions moved to {device}.")

lambda_recon = 1.0
lambda_perceptual = 1.0
lambda_gan = 1.0

vqgan_model.discriminator.train(False)
vqgan_model.train(True)

print("Loss weight constants defined and model training modes set.")

num_epochs=1

g_losses, d_losses = train(num_epochs,train_loader,vqgan_model,recon_loss_fn,perceptual_loss_fn,adv_loss_fn,optimizer_g,optimizer_d)

plt.figure(figsize=(10, 6))
plt.plot(g_losses, label='Generator Loss')
plt.plot(d_losses, label='Discriminator Loss')
plt.title('VQGAN Training Losses Over Batches')
plt.xlabel('Batch Number')
plt.ylabel('Loss Value')
plt.legend()
plt.grid(True)
plt.show()

print("VQGAN training losses plotted.")

vqgan_model.eval()

# Get a single batch of real images
try:
    data, _ = next(iter(train_loader))
except StopIteration:
    # If the current iterator is exhausted, create a new one.
    train_loader_iter = iter(train_loader)
    data, _ = next(train_loader_iter)

real_images_batch = data.to(device)

# Generate reconstructions
with torch.no_grad():
    reconstructions, _, _ = vqgan_model(real_images_batch)

# Move to CPU and convert to numpy for visualization
real_images_np = real_images_batch.cpu().numpy()
reconstructions_np = reconstructions.cpu().numpy()


num_display_images = min(8, real_images_np.shape[0])

plt.figure(figsize=(num_display_images * 2, 4))
for i in range(num_display_images):
    # Original Image
    plt.subplot(2, num_display_images, i + 1)
    original_img = unnormalize(torch.from_numpy(real_images_np[i])).permute(1, 2, 0).numpy()
    plt.imshow(original_img)
    plt.title('Real Image')
    plt.axis('off')

    # Reconstructed Image
    plt.subplot(2, num_display_images, num_display_images + i + 1)
    recon_img = unnormalize(torch.from_numpy(reconstructions_np[i])).permute(1, 2, 0).numpy()
    plt.imshow(recon_img)
    plt.title('Reconstruction')
    plt.axis('off')

plt.tight_layout()
plt.show()

print(f"Displayed {num_display_images} real images and their reconstructions.")
