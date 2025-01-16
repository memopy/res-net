import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

train = torchvision.datasets.CIFAR10(".data/",True,transform=transform)
test = torchvision.datasets.CIFAR10(".data/",False,transform=transform)

train,test = DataLoader(train,100,True),DataLoader(test,100,True)

class block(nn.Module):
    def __init__(self,in_channel,out_channel,stride=1,downsample=None):
        super().__init__()
        self.downsample = downsample

        self.conv1 = nn.Conv2d(in_channel,out_channel,kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel,out_channel,kernel_size=3,stride=stride,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(out_channel,out_channel*4,kernel_size=1,stride=1)
        self.bn3 = nn.BatchNorm2d(out_channel*4)

    def forward(self,x):
        identity = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.downsample:
            identity = self.downsample(identity)
        
        x += identity
        x = F.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self,layers,input_channels,num_classes):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(input_channels,64,kernel_size=7,stride=2,padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1 = self.make_layer(layers[0],64,1)
        self.layer2 = self.make_layer(layers[1],128,2)
        self.layer3 = self.make_layer(layers[2],256,2)
        self.layer4 = self.make_layer(layers[3],512,2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048,num_classes)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = F.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = F.relu(x)
        x = self.fc(torch.flatten(x,1))
        return x

    def make_layer(self,layer_count,out_channels,stride):
        downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * 4:
            downsample = nn.Sequential(nn.Conv2d(self.in_channels,out_channels*4,stride=stride,kernel_size=1))
        
        layers.append(block(self.in_channels,out_channels,stride,downsample))
        self.in_channels = out_channels*4
        
        for i in range(layer_count-1):
            layers.append(block(self.in_channels,out_channels))
        
        return nn.Sequential(*layers)
    
ResNet50 = ResNet((3,4,6,3),3,10).to(device)

lr = 0.0003
epochs = 25

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(ResNet50.parameters(),lr=lr)

def accuracy():
    ResNet50.eval()

    correct = 0
    for images,labels in test:
        images = images.to(device)
        labels = labels.to(device)
        correct += (torch.max(ResNet50(images),1)[1] == labels).sum().item()

    ResNet50.train()
    return correct/100

running_loss = 0
for i in range(1,epochs+1):
    for images,labels in train:
        images = images.to(device)
        labels = labels.to(device)
        loss = criterion(ResNet50(images),labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    running_loss += loss
    print(f"{i}. EPOCH. LOSS : {loss}. RUNNING LOSS : {running_loss/i}.ACCURACY : {accuracy()}")

torch.save(ResNet50.state_dict(),"resnet50_cifar10.pth")
