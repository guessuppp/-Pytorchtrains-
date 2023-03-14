import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as mp

x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]
def forword(x,b):
    return x * w +b
def loss(x,y,b):
    y_pred = forword(x,b)
    return (y_pred-y)*(y_pred-y)
w_list =[]
mse_list=[]
b_list=[]
for w in np.arange(0.0,4.1,0.1):
    for b in np.arange(-2.0, 2.1, 0.1):
        print('w=', w)
        print('b',b)
        for x_val,y_val in zip(x_data,y_data):
                l_sum = 0
                y_pred_val=forword(x_val,b)
                loss_val=loss(x_val,y_val,b)
                l_sum+=loss_val
                print('\t',x_val,y_val,b,y_pred_val,loss_val)
                print('Mse=',l_sum/3)
                w_list.append(w)
                mse_list.append(l_sum/3)
                b_list.append(b)
print(min(mse_list))



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = w_list
Y = b_list
Z = mse_list
ax.plot_trisurf(X, Y, Z,cmap="rainbow")
plt.show()










