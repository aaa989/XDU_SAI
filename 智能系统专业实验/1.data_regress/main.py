import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(- np.pi,np.pi,100),dim=1)
y = torch.sin(x) + 0.5*torch.rand(x.size(1))

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.predict = nn.Sequential(
            nn.Linear(1,10),
            nn.ReLU(),
            nn.Linear(10,1)
        )

    def forward(self,x):
        prediction = self.predict(x)
        return prediction

net = Net()
optimizer = torch.optim.SGD(net.parameters(),lr=0.1)
loss_func = nn.MSELoss()

plt.ion()
for epoch in range(10000):
    out = net(x)
    loss = loss_func(out,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch%1000 ==0:
        plt.cla()
        plt.scatter(x.data.numpy(),y.data.numpy())
        plt.plot(x.data.numpy(),y.data.numpy(),'r-',lw=5)
        plt.text(0.5,0,'Epoch = %d,Loss = %.4f'%(epoch,loss.data.numpy()),fontdict={'size':10,"color":"red"})
        plt.pause(0.5)
plt.ioff()
plt.show()

