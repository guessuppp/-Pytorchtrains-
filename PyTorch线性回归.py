

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as mp


epoch1=[]
item=[]
x_data=torch.Tensor([[1.0,1.0],[2.0,2.0],[3.0,3.0]])
y_data=torch.Tensor([[2.0],[4.0],[6.0]])

class LinerModel(torch.nn.Module):
    def __init__(self):
        super(LinerModel,self).__init__()
        self.linear = torch.nn.Linear(2,2)

    def forward(self,x):
        y_pred=self.linear(x)
        return y_pred
model = LinerModel()
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(),lr=0.0005,)

for epoch in range(200):
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
