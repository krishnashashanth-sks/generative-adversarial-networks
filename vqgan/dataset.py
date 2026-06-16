import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Define image transformations
transform = transforms.Compose([
    transforms.Resize(128), # Resize images to 128x128
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize to [-1, 1]
])

# Choose a dataset (CIFAR-10 for simplicity) and download it
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

print("Necessary modules imported, image transformations defined, and CIFAR-10 dataset initialized.")
