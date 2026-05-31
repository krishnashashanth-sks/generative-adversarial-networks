from torchvision.transforms import transforms
import torchvision
transform=transforms.Compose([
    transforms.Resize(64), # Add this line to resize images
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])
train_data=torchvision.datasets.MNIST("/root",train=True,transform=transform,download=True)
test_data=torchvision.datasets.MNIST("/root",train=False,transform=transform,download=True)
train_loader=DataLoader(train_data,batch_size=64,shuffle=True,num_workers=0)
test_loader=DataLoader(test_data,batch_size=64,shuffle=False,num_workers=0)
