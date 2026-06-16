from torch.utils.data import DataLoader
import os
import torchvision
import torchvision.transforms as transforms

transform=transforms.Compose([
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5]) # Changed from [0.5,0.5,0.5] to [0.5]
])
dataset=torchvision.datasets.MNIST(root='./data',download=True,transform=transform)
dataloader=DataLoader(dataset,shuffle=True,batch_size=64,num_workers=os.cpu_count())
