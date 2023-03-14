import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as mp
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torch.optim
####数据导入
# https://blog.csdn.net/qq_38765642/article/details/109779370?fromshare=blogdetail transform详细解释

epoch1=[]
item=[]

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),# 图像归一化处理 ，同时把图像由HWC→CHW
    transforms.Normalize((0.1307,),(0.3081,))# 使图像分布归于正态分布，更利于收敛，MEAN 均值，STD 标准差
])
train_dataset =datasets.MNIST( root='C:\ProgramData\Anaconda3\Lib\pytorchtrains\线性回归\数据集\mnist\\',
                               train=True,
                               download=True,
                               transform=transform)

train_loader = DataLoader(train_dataset,
                         shuffle=True,
                         batch_size=batch_size)

test_dataset =datasets.MNIST(root='C:\ProgramData\Anaconda3\Lib\pytorchtrains\线性回归\数据集\mnist\\',
                             train=False,
                             download=True,
                             transform=transform)

test_loader = DataLoader(train_dataset,
                        shuffle=False,
                        batch_size=batch_size)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.l1=torch.nn.Linear(784,512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 158)
        self.l4 = torch.nn.Linear(158, 64)

        self.l5 = torch.nn.Linear(64, 32)
        self.l6 = torch.nn.Linear(32, 10)

    def forward(self,x):

        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        x = F.relu(self.l6(x))

        return x

model = Net()

criterion = torch.nn.CrossEntropyLoss()##sofatmax + log +loss

optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

def train(epoch):
    runing_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()
        y_pred = model(inputs)
        loss = criterion(y_pred, target)
        loss.backward()

        optimizer.step()

        print(epoch, i, loss.item())
        runing_loss += loss.item()

        if i % 300 == 299:
            print('[%d,%5d] loss:%.3f'% (epoch + 1,i+1,runing_loss/300))
            running_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy on test set: %d %%' % (100*correct/total))


if __name__ == '__main__':
    for epoch in range(100):
        train(epoch)
        test()


































