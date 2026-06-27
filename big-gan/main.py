import matplotlib.pyplot as plt
from generator import BigGANGenerator
from discriminator import BigGANDiscriminator
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from dataset import dataset
import torch
from train import train
from losses import generator_loss,discriminator_loss

# Define parameters for the BigGANGenerator
latent_dim = 128  # Dimension of the latent noise vector
num_classes = 1000 # Number of classes (e.g., ImageNet has 1000 classes)
ch = 64           # Base number of channels
num_classes=10
img_size=128
# Instantiate the BigGANGenerator
generator = BigGANGenerator(latent_dim=latent_dim, num_classes=num_classes, ch=ch, img_size=img_size)

discriminator = BigGANDiscriminator(num_classes=num_classes, ch=ch, img_size=img_size, attention_res=[64, 32])

lr_G = 0.00005
lr_D = 0.0002
betas = (0.0, 0.999)

# Instantiate optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr_G, betas=betas)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_D, betas=betas)

dataloader=DataLoader(dataset,batch_size=1,shuffle=True,num_workers=os.cpu_count())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.to(device)
discriminator.to(device)
print(f"Models moved to {device}")

num_epochs = 1
n_critic = 5
batch_size = dataloader.batch_size # Use updated batch_size

train(num_epochs,dataloader,generator,discriminator,optimizer_D,optimizer_G,discriminator_loss,generator_loss,latent_dim,batch_size,n_critic,num_classes,device)


# 2. Create a dictionary to map CIFAR-10 class indices to their names
cifar10_classes = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

# 3. Set the generator model to evaluation mode
generator.eval()

# 4. Define the number of images to generate per class
num_images_per_class = 5 # Generate 5 images for each of the 10 classes

# 5. Create an empty list to store generated images and their corresponding labels
all_generated_images_data = []

# 6. Loop through each of the num_classes available
for class_idx in range(num_classes):
    # a. Generate latent vectors
    # Use a fixed seed for reproducibility for each class's noise if desired, or random for diversity
    z = torch.randn(num_images_per_class, latent_dim, device=device)

    # b. Create a tensor of class labels for the current class
    labels = torch.full((num_images_per_class,), class_idx, dtype=torch.long, device=device)

    # c. Pass the generated noise and class labels to the generator
    with torch.no_grad():
        fake_images = generator(z, labels).detach().cpu()

    # d. Denormalize the generated images from [-1, 1] to [0, 1]
    denormalized_images = (fake_images + 1) / 2.0

    # e. Append the generated images and their class labels
    for i in range(num_images_per_class):
        all_generated_images_data.append({
            'image': denormalized_images[i],
            'label_idx': class_idx,
            'label_name': cifar10_classes[class_idx]
        })

# 7. Create a figure and subplots to display the generated images
fig, axes = plt.subplots(num_classes, num_images_per_class, figsize=(num_images_per_class * 2, num_classes * 2))
fig.suptitle('Conditional Image Generation by BigGAN', fontsize=16)

# 8. Iterate through the stored images and labels, displaying each image
for i, img_data in enumerate(all_generated_images_data):
    row = img_data['label_idx']
    col = i % num_images_per_class

    ax = axes[row, col]
    # Move channel dimension to last for matplotlib (C, H, W) -> (H, W, C)
    ax.imshow(img_data['image'].permute(1, 2, 0))

    # 9. Add a title to each subplot indicating the class label
    if col == 0: # Only add label to the first column of each row
        ax.set_ylabel(img_data['label_name'], rotation=90, ha='right', fontsize=10)

    # 10. Turn off the axis for each subplot
    ax.axis('off')

# 11. Adjust the layout of the plots and display the figure
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to prevent suptitle overlap
plt.show()

# 12. Set the generator model back to training mode
generator.train()

print("Conditional image generation demonstration complete.")