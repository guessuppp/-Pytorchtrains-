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
#x_data=torch.Tensor([[1.0,1.0,1.0,1.0,1.0,1.0],[2.0,2.0,2.0,2.0,2.0,2.0],[3.0,3.0,3.0,3.0,3.0,3.0]])
A=[[-0.29,-0.88,-0.06,-0.88,0.00,-0.41,-0.65,0.18,-0.76,-0.06],
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
criterion = torch.nn.BCELoss(size_average=False)##BCELoss 交叉熵损失
optimizer = torch.optim.SGD(model.parameters(),lr=0.01,)
for epoch in range(1000):
    y_pred= model(x_data)
    loss = criterion(y_pred,y_data)
    print(epoch,loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch1.append(epoch)
    item.append(loss.item())
plt.plot(epoch1,item)
plt.ylabel('item')
plt.xlabel('epoch')
plt.show()



