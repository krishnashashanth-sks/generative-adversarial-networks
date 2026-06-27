import torchvision
import torchvision.transforms as transforms

img_size = 128    # Desired output image size (e.g., 128x128)

transform=transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,5])
])
dataset=torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)