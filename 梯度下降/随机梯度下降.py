import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as mp
import random
number=[0,1,2]
x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]

w = 1
losss=[]
epochs=[]
def forword(x):

    return x * w

def cost(xs,ys):
    y_pred = forword(xs)
    return (y_pred-ys)**2

def gradient(xs,ys):
    grad = 0

    grad +=2*xs*(xs*w-ys)
    return grad

for epoch in range (100):
    chosen = random.sample(number, 1)

    cost_val = cost(x_data[chosen[0]],y_data[chosen[0]])
    gral_val=gradient(x_data[chosen[0]],y_data[chosen[0]])
    w -= 0.007*gral_val 
    losss.append(cost_val)
    epochs.append(epoch)
    print('Epoch:',epoch,'w=',w,'loss=',cost_val)
plt.plot(epochs,losss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
