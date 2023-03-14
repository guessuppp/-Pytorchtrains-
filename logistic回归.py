import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as mp
import torchvision
import torch.nn.functional as F

#数据集
#模型选择
#损失函数
#训练循环
#train_set = torchvision.datasets.MNIST(root='E:\pytorch\线性回归\data',train=True,download=True)
#test_set = torchvision.datasets.MNIST(root='E:\pytorch\线性回归\data',train=False,download=True)
##sigmoid functions 激活函数

epoch1=[]
item=[]
x_data=torch.Tensor([[1.0],[2.0],[3.0],[4.0],[5.0],[6.0]])
y_data=torch.Tensor([[0],[0],[0],[1],[1],[1]])
class LogisiticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisiticRegressionModel,self).__init__()
        self.linear = torch.nn.Linear(1,1,True)

    def forward(self,x):
        y_pred=F.sigmoid(self.linear(x))
        return y_pred
model = LogisiticRegressionModel()
criterion = torch.nn.BCELoss(size_average=False)##BCELoss 交叉熵损失
optimizer = torch.optim.SGD(model.parameters(),lr=0.05,)
for epoch in range(1000):
    y_pred= model(x_data)
    loss = criterion(y_pred,y_data)
    print(epoch,loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch1.append(epoch)
    item.append(loss.item())
#plt.plot(epoch1,item)
#plt.ylabel('item')
#plt.xlabel('epoch')
#plt.show()

x = np.linspace(0,10,200)
x_t=torch.Tensor(x).view(200,1)
y_t=model(x_t)
y=y_t.data.numpy()
plt.plot(x,y)
plt.plot([0,10],[0.5,0.5],c='r')

plt.ylabel('item')
plt.xlabel('epoch')
plt.grid()
plt.show()

