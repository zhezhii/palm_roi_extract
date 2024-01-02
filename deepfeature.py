import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import datetime
import numpy as np
import matplotlib.pyplot as plt
 
# 默认用GPU训练，没有GPU则是用CPU寻来，方便电脑里没有GPU的同学做实验，注意用CPU训练的速度会相对慢一点，但由于我们要处理的静脉数据不算特别多，实验时间还是挺短的，但同学们增大模型或者增加数据后，训练时间就会增加
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class VeinNet(nn.Module):
    # 这个模型参数量很少，运算量很小，是为了方便没有GPU的同学做实验，效果未必最好，同学们可以根据自己的知识或通过学习《神经网络与深度学习》课程后优化模型
    def __init__(self):
        super(VeinNet, self).__init__()
        # 以下定义四个卷积层，作用是通过训练后其卷积核具有提取静脉特征的能力
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=0)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0)
        # self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0, groups=32)

        # 以下定义四个batch normalization层，作用是对中间数据做归一化处理
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)
        # self.bn5 = nn.BatchNorm2d(128)

        # 以下定义池化层，作用是对长和宽维度做下采样
        self.pool = nn.MaxPool2d(2, 2)

        # 以下定义激活层，作用是增加神经网络模型的非线性
        self.act = nn.LeakyReLU()

        # 以下定义最后的特征处理层，作用是将神经网络的三维矩阵特征变为一维向量特征后经过全连接层输出分类逻辑
        self.feature = nn.AdaptiveAvgPool2d(1)
        # self.x2c = nn.Linear(64, 4) # 由于给的例程数据是8类，所以这里的输出维度等于类别数是8
        self.x2c = nn.Linear(64, 10) # 由于给的例程数据是8类，所以这里的输出维度等于类别数是8

    def forward(self, x):
        # 第一层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool(x)
        x = self.act(x)
        # 第二层
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool(x)
        x = self.act(x)
        # 第三层
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.pool(x)
        x = self.act(x)
        # 第四层
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.pool(x)
        x = self.act(x)
        # # 第五层
        # x = self.conv5(x)
        # x = self.bn5(x)
        # x = self.pool(x)
        # x = self.act(x)
        # 输出特征
        x = self.feature(x).view(-1, 64)
        c = self.x2c(x)
        return c

def data_read(dir_path):
    with open(dir_path, "r") as f:
        raw_data = f.read()
        data = raw_data[1:-1].split(", ")
    return np.asfarray(data, float)


# 数据格式转化，还能在这里写数据扩增方法，同学们可自行探讨，注意训练集扩增，测试集不用扩增
train_transform = transforms.Compose([transforms.Resize((64, 64)),
                                      transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
                                      transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                                      transforms.ToTensor()])
test_transform = transforms.Compose([transforms.Resize((64, 64)),
                                     transforms.ToTensor()])

 
# 在构建数据集的时候指定我们定义好的transform进行数据格式转换
# 同时指定训练数据和测试数据的位置，默认一个子文件夹下存放同个类别的数据
train_data = torchvision.datasets.ImageFolder(root='./data/train', transform=train_transform)
test_data = torchvision.datasets.ImageFolder(root='./data/test', transform=test_transform)
# 将数据集放到迭代加载器里面去，便于批量调用数据集
train_loader = torch.utils.data.DataLoader(train_data, batch_size=2, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=2, shuffle=True)

# 实例化模型
net = VeinNet().to(device)
 
# 使用交叉熵做损失函数，CrossEntropyLoss
criterion = nn.CrossEntropyLoss()

# 使用随机梯度下降法做优化，学习率lr可调，有讲究，根据实际任务而定，一般不用这么大
optimizer = optim.SGD(net.parameters(), lr=0.03)

# 程序入口
if __name__ == '__main__':
    print("Start Training...")
    start_time = datetime.datetime.now()
    print("start_time:", start_time)
    train_loss = []
    train_acc = []
    num = 0
    # 以下为训练过程，训练轮次这里默认为100，这里同学们要根据实验结果做调整
    for epoch in range(100):
        # 用来存放epoch累加loss
        all_loss = 0.0
        correct = 0
        num = 0
        # 我们的dataloader派上了用场
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            _, pre = torch.max(outputs.data, 1)
            num += labels.size(0)
            correct += (pre == labels).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            all_loss += loss.item()
            # print('loss: ', loss)
        print('Epoch %d mean loss: %.8f' % (epoch+1, all_loss/(i+1)))
        train_loss.append(all_loss/(i+1))
        train_acc.append(correct/num)

    with open("./train_loss.txt", 'w') as train_l:
        train_l.write(str(train_loss))
    with open("./train_acc.txt", 'w') as train_a:
        train_a.write(str(train_acc))


    # 记录结束时间
    end_time = datetime.datetime.now()
    print("end_time:", end_time)
    print("Done Training!")
    # 计算时间
    delta = end_time - start_time
    # 打印时间
    print("total_seconds: ", delta.total_seconds())

    # 以下为测试过程，得到准确率
    num = 0
    correct = 0
    net.eval()
    for i, data in enumerate(test_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, pre = torch.max(outputs.data, 1)
        num += labels.size(0)
        correct += (pre == labels).sum().item()
    print('Test Accuracy: {:.2%}'.format(correct/num))


    # 以下为绘制损失函数代码
    train_loss_path = "./train_loss.txt"
    y_train_loss = data_read(train_loss_path)
    x_train_loss = range(len(y_train_loss))
    plt.figure(1)
    ax = plt.axes()
    plt.xlabel('epochs')    # x轴标签
    plt.ylabel('Loss')     # y轴标签
    plt.plot(x_train_loss, y_train_loss, color='red',linewidth=1, linestyle="solid", label="train loss")
    plt.legend()
    plt.title('Loss curve')


    # 以下为绘制训练准确率代码
    train_acc_path = "./train_acc.txt"
    y_train_acc = data_read(train_acc_path)
    x_train_acc = range(len(y_train_acc))
    plt.figure(2)
    ax = plt.axes()
    plt.xlabel('epochs')    # x轴标签
    plt.ylabel('Train accuracy')     # y轴标签
    plt.plot(x_train_acc, y_train_acc, color='blue',linewidth=1, linestyle="solid", label="train acc")
    plt.legend()
    plt.title('Train accuracy curve')
    plt.show()
