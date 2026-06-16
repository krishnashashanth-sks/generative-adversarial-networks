import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
print("PyTorch and torchvision libraries imported successfully.")

source_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3), # Convert 1-channel MNIST to 3-channels
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

target_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("Image transformations defined successfully.")

source_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=source_transform, download=True)
target_dataset=torchvision.datasets.MNIST(root='./data',train=False,tramsform=target_transform,download=True)

print(f"Source dataset (MNIST) loaded successfully with {len(source_dataset)} samples.")

batch_size = 32
source_dataloader = torch.utils.data.DataLoader(source_dataset, batch_size=batch_size, shuffle=True)
target_dataloader = torch.utils.data.DataLoader(target_dataset, batch_size=batch_size, shuffle=True)

print(f"Source Dataloader created with batch size: {batch_size}")
print(f"Target Dataloader created with batch size: {batch_size}")