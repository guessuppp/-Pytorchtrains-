import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as mp

x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]

w = 1
b = 1
def forword(x,b):
    return x * w + b

def cost(xs,ys,b):
    cost = 0
    for x,y in zip(xs,ys):
        y_pred = forword(x,b)
        cost += (y_pred-y)**2
    return cost/len(xs)
def gradient(xs,ys,b):
    grad = 0
    for x,y in zip(xs,ys):
        grad +=2*x*(x*w +b-y)
    return grad/len(xs)
for b in np.arange(-5,5,0.1):
    print('b=',b)
    for epoch in range (100):
        cost_val = cost(x_data,y_data,b)
        gral_val=gradient(x_data,y_data,b)
        w -= 0.007*gral_val
    print('Epoch:',epoch,'w=',w,'loss=',cost_val)