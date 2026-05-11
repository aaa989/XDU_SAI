import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 生成训练数据
data = torch.ones(100, 2)
x0 = torch.normal(2 * data, 1)
x1 = torch.normal(-2 * data, 1)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)

y0 = torch.zeros(100)
y1 = torch.ones(100)
y = torch.cat((y0, y1), 0).type(torch.LongTensor)

# 定义神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.classify = nn.Sequential(
            nn.Linear(2, 15),
            nn.ReLU(),
            nn.Linear(15, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.classify(x)

net = Net()
optimizer = torch.optim.SGD(net.parameters(), lr=0.03)
loss_func = nn.CrossEntropyLoss()

# 训练过程 + 动态绘图
plt.ion()
for epoch in range(100):
    out = net(x)
    loss = loss_func(out, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        plt.cla()
        classification = torch.max(out, 1)[1]
        class_y = classification.data.numpy()
        target_y = y.data.numpy()   
        
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=class_y, s=100, cmap="RdYlGn")    
        accuracy = sum(class_y == target_y) / 200
        plt.text(1.5, -4, "Accuracy={:.4f}".format(accuracy), fontdict={"size": 20, "color": "red"})
        plt.pause(0.5)

plt.ioff()
plt.show()

# 生成测试数据
test_data = torch.ones(30, 2)
x0_test = torch.normal(2 * test_data[:15], 1)
x1_test = torch.normal(-2 * test_data[15:], 1)
x_test = torch.cat((x0_test, x1_test), 0).type(torch.FloatTensor)
y_test = torch.cat((torch.zeros(15), torch.ones(15)), 0).type(torch.LongTensor)

# 模型测试
net.eval()
with torch.no_grad():
    test_out = net(x_test)
    test_pred = torch.max(test_out, 1)[1]
    test_acc = (test_pred == y_test).sum().item() / len(y_test)
    
    print(f"测试集大小: {len(y_test)}个样本")
    print(f"预测准确率: {test_acc:.2%}")
    
    print("序号\t真实标签\t预测标签\t类别0概率\t类别1概率\t结果")
    print("-" * 70)
    for i in range(len(y_test)):
        prob0 = test_out[i][0].item()
        prob1 = test_out[i][1].item()
        result = "✓" if test_pred[i] == y_test[i] else "✗"
        print(f"{i+1}\t{y_test[i].item()}\t\t{test_pred[i].item()}\t\t{prob0:.4f}\t{prob1:.4f}\t{result}")