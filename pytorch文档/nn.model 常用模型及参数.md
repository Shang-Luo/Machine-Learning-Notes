## 如何构建一个神经网络
`torch.nn.Module` 是**所有神经网络模块（层/模型）的父类**，它提供了神经网络训练、推理、参数管理、设备迁移（CPU/GPU）的全套基础功能。
### 核心作用
- **自动管理可训练参数**（`weight`, `bias`）
- **嵌套组合**：可以把小模块拼成大模型（Layer → Block → Model）
- **设备迁移**：一行代码 `model.to('cuda')` 移动所有参数
- **训练/推理模式切换**：`model.train()` / `model.eval()`
- **自动前向传播**：必须实现 `forward()` 方法
- **保存/加载模型**：`torch.save()` / `torch.load()`
### `nn.Module` 示例
```python
import torch
import torch.nn as nn

# 自定义模型必须继承 nn.Module
class MyModel(nn.Module):
    # 1. 初始化：定义所有层
    def __init__(self):
        super().__init__()  # 必须调用父类初始化
        self.linear = nn.Linear(10, 5)  # 线性层
        self.relu = nn.ReLU()            # 激活函数

    # 2. 前向传播：定义数据流动路径
    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return x

# 使用
model = MyModel()
output = model(torch.randn(2, 10))  # 自动调用 forward()
```

> [!NOTE] 数据是如何流动的
> 在`self.__init__`中，我们仅仅放置了小的模块（积木）而没有拼在一起。
> `def forward(self,x)`中才定义了数据的流动方式：数据从`x` 输入，从`return` 输出


## 层的类型
### 线性层（全连接层）
最基础的层，用于**特征映射、分类头**。
#### `nn.Linear(in_features, out_features, bias=True)`
- **功能**：线性变换 $y = xA^T + b$
- **输入**：`(*, in_features)`
- **输出**：`(*, out_features)`
- **常用场景**：MLP、分类最后一层、特征映射

```python
linear = nn.Linear(20, 10)  # 输入20维，输出10维
```

---
### 卷积层（计算机视觉核心）
用于**图像特征提取**，分为 1D/2D/3D，分别处理序列、图像、视频。

#### `nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)`
- **最常用的图像卷积**
- `in_channels`: 输入通道（RGB=3，灰度=1）
- `out_channels`: 输出通道（卷积核数量）
- `kernel_size`: 卷积核大小（3=3x3）
- `stride`:位移
- `padding`:该给边缘补几圈 $0$（防止图像缩小）

```python
conv = nn.Conv2d(3, 16, 3, padding=1)  # 3→16通道，3x3卷积
```

####  其他卷积
- `nn.Conv1d`：时序数据、语音
- `nn.Conv3d`：视频、3D 医学图像
- `nn.ConvTranspose2d`：转置卷积（上采样，图像生成）

> [!NOTE] 卷积通用参数模板
> ```python
> nn.ConvNd(
   > in_channels,   # 输入通道数
   > out_channels,  # 输出通道数（=卷积核个数）
   > kernel_size,   # 核大小
   > stride=1,      # 步幅（移动步长）
   > padding=0,     # 填充（补0圈数）
>)
>  ```
>  #### nn.ConvTranspose2d（转置卷积 = 上采样 = 放大图像）
>用于：**图像变大**（分割、生成、超分）
>  ```python
>  conv_trans = nn.ConvTranspose2d(
>     in_channels=16,
>     out_channels=3,
>     kernel_size=3,
>     stride=2,       # 步幅>1 → 图像放大
>     padding=1,
>     output_padding=1 # 可选：微调输出尺寸
> )
> ```

---
### 池化层（降采样）
**减少参数量、防止过拟合、扩大感受野**

#### 1. `nn.MaxPool2d(kernel_size, stride=None)`
最大池化（最常用）
```python
pool = nn.MaxPool2d(2)  # 2x2池化，尺寸减半
```

#### 2. `nn.AvgPool2d`
平均池化
```python
nn.AvgPool2d(kernel_size, stride=None, padding=0)
```
#### 3. `nn.AdaptiveAvgPool2d(output_size)`
自适应平均池化（**强制输出固定尺寸**，非常好用）
```python
adaptive_pool = nn.AdaptiveAvgPool2d((1,1))  # 任何图变成 1x1 特征
```

---
### 归一化层（训练稳定神器）
对上一层的输出进行归一（到$(0,1)$or$(1,-1)$）
解决**梯度消失、训练不稳定、过拟合**，必须掌握。
$$ 数据 = \frac{数据 - 均值}{标准差} $$

#### 1. nn.BatchNorm2d（CNN 标配）
**按通道归一，跨批次、跨图片、跨像素**
```python
bn = nn.BatchNorm2d(num_features=16)
```

#### 2. nn.LayerNorm（Transformer 标配）
**单个样本内归一，跨所有通道/特征**
```python
ln = nn.LayerNorm(normalized_shape=512)
```

#### 3. nn.InstanceNorm2d（风格迁移/GAN）
**单张图片 + 单个通道 独立归一**
```python
in_layer = nn.InstanceNorm2d(num_features=16)
```

> [!NOTE]
> #### 1. 归一化加在哪？
> **不是最后一层，不是任意层，是【每一层】卷积/全连接后面都加！**
> - CNN：`Conv → BN → 激活`
> - Transformer：`LN → Linear`
> #### 2. 为什么要层层加？
> 只有**层层归一**，才能把每层特征锁在稳定范围，**才能用大学习率，训练快、不崩**。
> #### 3. 为什么要在 **Sigmoid 前用归一**？（核心重点）
> **Sigmoid 函数两头是平的，输入太大/太小，梯度直接变 0（梯度消失）！**
> 
> - 没归一：输入很大 → Sigmoid 输出卡在 0 或 1 → **梯度没了，网络学不动**
> - 加归一：把输入强行拉到 **0 附近** → Sigmoid 处于**斜率最大、梯度最稳**的区域 → **训练正常、不消失**

---
### 激活函数（非线性能力）
没有激活函数，网络就是线性模型，**无法拟合复杂特征**。

全部继承自 `nn.Module`：
```python
nn.ReLU()        # 最常用
nn.Sigmoid()     # 二分类
nn.Tanh()        # 输出 -1~1
nn.GELU()        # Transformer 标配
nn.Softmax(dim=-1) # 多分类概率
```

> [!NOTE] 常见激活表
>#### nn.ReLU()
>$$
> \text{ReLU}(x) = \max(0, x)
> $$
最常用，CNN/网络标配，解决梯度消失 
> #### nn.Sigmoid()
> $$
> \text{Sigmoid}(x) = \frac{1}{1+e^{-x}}
> $$
> 输出 0~1，二分类，必须配合归一化使用
> 
> #### nn.Tanh()
> $$
> \text{Tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
> $$
> 输出 -1~1，特征中心化
> 
> #### nn.GELU()
> $$
> \text{GELU}(x) = x \cdot \Phi(x)
> $$
> $\boldsymbol{\Phi(x)}$：标准正态分布累积分布函数
> $$
> \Phi(x)=\frac1{\sqrt{2\pi}}\int_{-\infty}^x e^{-\frac{t^2}{2}}dt
> $$
> Transformer 标配，更平滑稳定
> 
> #### nn.Softmax(dim=-1)
> $$
> \text{Softmax}(x_i) = \frac{e^{x_i}}{\sum e^{x_j}}
> $$
> 多分类输出概率，总和=1
> 


---

### 循环层（时序/自然语言处理）
处理**序列数据**：文本、语音、时间序列。

#### 1. `nn.LSTM(input_size, hidden_size, num_layers, batch_first=False)`
长短期记忆网络（解决梯度消失）

#### 2. `nn.GRU`
简化版 LSTM，更快

#### 3. `nn.RNN`
基础循环网络（少用）

```python
lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
```

---

### Dropout 层（防止过拟合）
训练时随机让部分神经元失活。每个节点/通道独立关闭

```python
nn.Dropout(p=0.5)    # 标准 dropout       随机关闭节点
nn.Dropout2d(p=0.5)  # 图像通道 dropout    随机关闭通道
```