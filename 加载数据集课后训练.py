import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as mp
import torchvision
import torch.nn.functional as F

epoch1=[]
item=[]

##泰坦尼克号
class DiabetesDataset(Dataset):
    def __init__(self, filepath):

        xy = pd.read_csv(filepath)

        xy = np.loadtxt(filepath, delimiter=',',skiprows=1, usecols=(2,3,4,5,6,8,1),dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]

    def __len__(self):
        return self.len


dataset = DiabetesDataset('C:\ProgramData\Anaconda3\Lib\pytorchtrains\线性回归\泰坦尼克号\\train.csv')

train_loader = DataLoader(
    dataset = dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2

)


class LogisiticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisiticRegressionModel,self).__init__()
        self.linear = torch.nn.Linear(6,4,True)
        self.linear1 = torch.nn.Linear(4, 2,True)
        self.linear2 = torch.nn.Linear(2, 1,True)

    def forward(self,x):
        x=F.sigmoid(self.linear(x))
        x = F.sigmoid(self.linear1(x))

        y_pred = F.sigmoid(self.linear2(x))

        return y_pred

model = LogisiticRegressionModel()


criterion = torch.nn.BCELoss(reduction='sum')##BCELoss 交叉熵损失

optimizer = torch.optim.SGD(model.parameters(),lr=0.01,)



if __name__ == '__main__':

    for epoch in range(100):
        for i ,data in enumerate(train_loader,0):
            inputs , labels = data

            y_pred = model(inputs)
            loss = criterion(y_pred,labels)



            print(epoch,i,loss.item())

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
        epoch1.append(epoch)
        item.append(loss.item())
    plt.plot(epoch1, item)
    plt.ylabel('item')
    plt.xlabel('epoch')
    plt.show()






