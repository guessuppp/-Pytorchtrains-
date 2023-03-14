import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# 准备数据集
class DiabetesDataset(Dataset):
    def __init__(self, filepath):  # filepath 表示数据来自什么地方
        # np.loadtxt为读取文本文档的函数
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        # shape为（N,9）元组，取出N的值
        self.len = xy.shape[0]
        # 第一个‘：’是指读取所有行，第二个‘：’是指从第一列开始，最后一列不要
        self.x_data = torch.from_numpy(xy[:, :-1])
        # 要最后一列，且最后得到的是个矩阵，所以要加[]
        self.y_data = torch.from_numpy(xy[:, [-1]])

    # 把里面的x_data[index],y_data[index]的第index条数据给拿出来
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # 把整个数据的数量取出来,返回数据集的数据条数
    def __len__(self):
        return self.len


dataset = DiabetesDataset('C:\ProgramData\Anaconda3\Lib\pytorchtrains\线性回归\diabetes.csv')  # 111.csv.gz数据路径
# 用 DataLoader 构造了一个加载器
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)


# 设计模型
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linearl1 = torch.nn.Linear(8, 6)
        self.linearl2 = torch.nn.Linear(6, 4)
        self.linearl3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linearl1(x))
        x = self.sigmoid(self.linearl2(x))
        x = self.sigmoid(self.linearl3(x))
        return x


model = Model()

# 构造损失函数和优化器
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练
if __name__ == '__main__':
    for epoch in range(100):
        # 对train_loader做迭代，用 enumerate是为了获得当前是第几次迭代
        # 把从train_loader拿出来的（x,y）元组放到data里面
        for i, data in enumerate(train_loader, 0):
            # 在训练之前把x,y从data里面拿出来，inputs=x,labels=y，
            # 此时inputs,labels都已经被自动转换为张量（tensor）
            inputs, labels = data

            # Forward
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())

            # backward
            optimizer.zero_grad()
            loss.backward()
            # update
            optimizer.step()