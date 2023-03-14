import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as mp


epoch1=[]
item=[]
x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]
w = torch.Tensor([1.0])
w.requires_grad = True

def forword(x):
    return x * w
def loss(x,y):
    y_pred = forword(x)
    return (y_pred-y) ** 2
for epoch in range(100):
    for x,y in zip(x_data,y_data):
        l=loss(x,y)
        l.backward()
        print('\tgrad:',x,y,w.grad.item(),'Loss=',l.item())
        print(w.grad.data)
        w.data=w.data-0.01*w.grad.data
        w.grad.data.zero_()
        epoch1.append(epoch)
        item.append(l.item())
    print('progress:',epoch,l.item())


print("predict (after training)",4,forword(4).item())
plt.plot(epoch1,item)
plt.ylabel('item')
plt.xlabel('epoch')
plt.show()


