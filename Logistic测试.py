import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as mp
import torchvision
import torch.nn.functional as F
class LogisiticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisiticRegressionModel,self).__init__()
        self.linear = torch.nn.Linear(1,1,True)

    def forward(self,x):
        y_pred=F.sigmoid(self.linear(x))
        return y_pred
model = LogisiticRegressionModel()


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