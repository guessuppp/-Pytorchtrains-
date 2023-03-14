import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as mp
import torchvision
import torch.nn.functional as F

epoch1=[]
item=[]
A=[
                     [-0.29,-0.88,-0.06,-0.88,0.00,-0.41,-0.65,0.18,-0.76,-0.06],
                     [0.49,-0.15,0.84,-0.11,0.38,0.17,-0.22,0.16,0.98,0.26],
                     [0.18,0.08,0.05,0.08,-0.34,0.21,-0.18,0.00,0.15,0.57],
                     [-0.29,-0.41,0.00,-0.54,-0.29,0.00,-0.35,0.00,-0.09,0.00],
                     [0.00,0.00,0.00,-0.78,-0.60,0.00,-0.79,0.00,0.28,0.00],
                     [0.00,-0.21,-0.31,-0.16,0.28,-0.24,-0.08,0.05,-0.09,0.00],
                     [-0.53,-0.77,-0.49,-0.92,0.89,-0.89,-0.85,-0.95,-0.93,-0.87],
                     [-0.03,-0.67,-0.63,0.00,-0.60,-0.70,-0.83,-0.73,0.07,0.10]]
B=np.transpose(A)

x_data=torch.Tensor(B)

y_data=torch.Tensor([[0],[1],[0],[1],[0],[1],[0],[1],[0],[0]])


class DiabetesDataset(Dataset):
    def __init__(self, filepath):

        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:,:-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]

    def __len__(self):
        return self.len


dataset = DiabetesDataset('C:\ProgramData\Anaconda3\Lib\pytorchtrains\线性回归\diabetes.csv')

train_loader = DataLoader(
    dataset = dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2

)


class LogisiticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisiticRegressionModel,self).__init__()
        self.linear = torch.nn.Linear(8,6,True)
        self.linear1 = torch.nn.Linear(6, 4, True)
        self.linear2 = torch.nn.Linear(4, 1, True)

    def forward(self,x):
        x=F.sigmoid(self.linear(x))
        x = F.sigmoid(self.linear1(x))

        y_pred = F.sigmoid(self.linear2(x))

        return y_pred

model = LogisiticRegressionModel()


criterion = torch.nn.BCELoss(reduction='sum')##BCELoss 交叉熵损失

optimizer = torch.optim.SGD(model.parameters(),lr=0.01,)



if __name__ == '__main__':
    for epoch in range(1000):
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






