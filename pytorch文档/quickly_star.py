import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

"""
我们有两个处理数据的类：Dataset和DataLoader
Dataset: 代表一个数据集，包含了数据的样本和标签
DataLoader: 代表一个数据加载器，负责将Dataset中的数据分批次加载到模型中进行训练或测试
"""

"""
V2 是一个转换器库，提供了一些常用的数据转换操作，可以方便地对数据进行预处理和增强。
在这个例子中，我们使用了 v2.Compose 来组合多个转换操作，
包括:
v2.ToImage() 将数据转换为图像格式，
v2.ToDtype(torch.float32, scale=True) 将数据类型转换为 float32 并进行缩放。
"""
training_data = datasets.MNIST(
    root="data",    #数据存储路径
    train=True,     #下载训练数据
    download=True,  #如果数据不存在则下载
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]), 
    #对数据进行转换，先将数据转换为图像格式，再将数据类型转换为 float32 并进行缩放
)

test_data = datasets.MNIST(
    root="data",
    train=False,    #下载测试数据
    download=True, 
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)

"""
加载dataloader,并设定batch_size
"""
batch_size = 64 #每个批次的样本数量
train_dataloader = DataLoader(training_data, batch_size=batch_size) #创建训练数据加载器
test_dataloader = DataLoader(test_data, batch_size=batch_size) #创建测试数据加载

"""
查看数据的形状
for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
    
out:
Shape of X [N, C, H, W]: torch.Size([64, 1, 28, 28])    => N=64样本数量, C=1通道数, H=28, W=28 宽度和高度
Shape of y: torch.Size([64]) torch.int64                => 64个样本，数据类型为int64
"""

#设定硬件加速设备
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
"""
之前没请缓存导致一直安的是CPU版的，现在安装了GPU版的torch，终于可以使用GPU了
out:
Using cuda device
"""

# 定义模型
"""
模型的前向传播使用矩阵乘法，输入，输出和中间的矩阵维度需要匹配
"""
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            #nn.Linear(输入维度, 输出维度, bias=True) 自带偏移向量
            nn.Linear(28*28, 512),      #输入层到隐藏层的线性变换，输入维度为28*28=784，输出维度为512
            nn.ReLU(),                  #激活函数，增加模型的非线性能力   
            nn.Linear(512, 512),        #隐藏层到隐藏层的线性变换，输入维度为512，输出维度为512
            nn.ReLU(),                  #激活函数，增加模型的非线性能力
            nn.Linear(512, 10)          #隐藏层到输出层的线性变换，输入维度为512，输出维度为10（对应10个类别）
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

"""
out:
outNeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)
"""


#定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss()#loss:交叉熵损失函数，适用于多分类问题
optimizer = torch.optim.Adam(
    model.parameters(),     #优化器，使用Adam算法，更新模型参数
    lr=0.001,               #学习率，控制参数更新的步长
    betas=(0.9, 0.999),     #Adam优化器的两个超参数，分别控制一阶矩估计和二阶矩估计的指数衰减率
    eps=1e-08,              #小浮点数，防止除0
    weight_decay=0.0,       #权重衰减，L2正则化的系数，防止过拟合
    amsgrad=False           #使用AMSGrad变体的Adam优化器，默认值为False
)#见笔记 [[optimizer 优化器]]

#训练和测试模型
def train(dataloader, model, loss_fn, optimizer):   #训练函数，接受数据加载器、模型、损失函数和优化器作为输入
    size = len(dataloader.dataset)                  #数据集的大小
    model.train()                                   #将模型设置为训练模式，启用dropout和batch normalization等训练特定的行为    
    for batch, (X, y) in enumerate(dataloader):     #遍历所有的data loader中的批次，获取输入数据X和标签y
        X, y = X.to(device), y.to(device)           #迁移到指定的设备（CPU或GPU）

        # 前向传播计算误差C
        pred = model(X)                 #通过模型进行前向传播，得到预测结果pred
        loss = loss_fn(pred, y)         #计算损失函数，比较预测结果pred和真实标签y，得到损失值loss

        # 反向传播
        loss.backward()                 #反向传播计算梯度，自动计算损失函数相对于模型参数的梯度
        optimizer.step()                #更新模型参数，根据计算得到的梯度更新模型参数，执行一次优化步骤
        optimizer.zero_grad()           #清除优化器中的梯度，防止梯度累积，准备下一次迭代的梯度计算
        """
        pytorch的函数设计包含了自动梯度累加设计,需要手动清0
        这么做是为了支持 多 loss 叠加：多个 loss 分别 backward,梯度自然相加
        """
        if batch % 100 == 0:            #每100个批次打印一次当前的损失和进度          
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]") #打印当前的损失值和进度，loss: 当前的损失值，current: 当前处理的样本数量，size: 数据集的总大小


def test(dataloader, model, loss_fn):   #测试函数，接受数据加载器、模型和损失函数作为输入
    size = len(dataloader.dataset)      #数据集的大小
    num_batches = len(dataloader)       #数据加载器中的批次数量
    model.eval()                #将模型设置为评估模式，禁用dropout和batch normalization等训练特定的行为
    test_loss, correct = 0, 0   #初始化测试损失和正确预测的数量
    with torch.no_grad():       #禁用梯度计算，节省内存和计算资源，因为在测试阶段不需要进行反向传播
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)                         #前向传播计算预测结果pred
            test_loss += loss_fn(pred, y).item()    #计算损失函数，比较预测结果pred和真实标签y，得到损失值，并累加到test_loss中
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


#开始训练
epochs = 5 #批次数量，训练模型的轮数
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer) #调用训练函数，传入训练数据加载器、模型、损失函数和优化器
    test(test_dataloader, model, loss_fn)              #调用测试函数，传入测试数据加载器、模型和损失函数
print("Done!")

#保存模型状态字典
torch.save(model.state_dict(), "./data/models/model.pth")
print("Saved PyTorch Model State to model.pth")
"""
加载：
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load( "./data/models/model.pth", weights_only=True))
"""