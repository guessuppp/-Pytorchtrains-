import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as mp

x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]

w = 1
losss=[]
epochs=[]
def forword(x):
    return x * w

def cost(xs,ys):
    cost = 0
    for x,y in zip(xs,ys):
        y_pred = forword(x)
        cost += (y_pred-y)**2
    return cost/len(xs)
def gradient(xs,ys):
    grad = 0
    for x,y in zip(xs,ys):
        grad +=2*x*(x*w-y)
    return grad/len(xs)
for epoch in range (100):
    cost_val = cost(x_data,y_data)
    gral_val=gradient(x_data,y_data)
    w -= 0.007*gral_val
    losss.append(cost_val)
    epochs.append(epoch)
    print('Epoch:',epoch,'w=',w,'loss=',cost_val)
plt.plot(epochs,losss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
