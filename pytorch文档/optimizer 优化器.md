## Adam 完整参数详解
### 函数原型
```python
torch.optim.Adam(
    params,
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=0.0,
    amsgrad=False
)
```

逐参数解释：
1. **`params`**
   固定填 `model.parameters()`，代表需要优化的网络权重/偏置。

2. **`lr` 学习率**
   - 默认值：`0.001`（`1e-3`）
   - 控制参数更新步长；Adam 常用区间：`1e-4 ~ 1e-3`

3. **`betas=(β1, β2)`** 动量系数（核心）
   - `β1=0.9`：一阶矩（梯度均值）平滑系数，相当于动量，**平滑梯度、减弱震荡**
   - `β2=0.999`：二阶矩（梯度平方均值）平滑系数，**自适应调整每个参数的学习率**
   - 绝大多数场景**使用默认值即可，不用修改**

4. **`eps` 极小值**
   - 默认 `1e-8`
   - 防止分母为 0，数值稳定用，一般不改动。

5. **`weight_decay` 权重衰减（L2 正则化）**
   - 默认 `0`（不开启）
   - 作用：抑制**过拟合**，给权重加惩罚；分类任务可设 `1e-4` 左右。
   示例：
   ```python
   optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
   ```

6. **`amsgrad`**
   - 默认 `False`
   - 增强版 Adam，解决 Adam 后期收敛变差问题；普通任务不用开启。

---

# 主流常见优化器 + 区别、适用场景
结合你的 **FashionMNIST 图像分类**，对比最常用 5 款：
SGD、带动量 SGD、Adam、AdamW、RMSprop

## 1. SGD 随机梯度下降（原教程使用）
```python
torch.optim.SGD(model.parameters(), lr=1e-3)
```
- 特点：
  只用**当前批次梯度**更新参数，无自适应学习率。
- 优点：泛化能力强、占用显存小。
- 缺点：收敛慢、易震荡、对学习率敏感。
- 适用：大数据集、需要强泛化、调参经验丰富的场景。

## 2. SGD + Momentum 带动量 SGD
```python
torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
```
- 在 SGD 基础上加**历史梯度累积**，像小球带惯性下滑。
- 优点：收敛更快、震荡更小。
- 适用：传统 CV 模型、竞赛常用。

## 3. Adam（自适应矩估计，**新手首选**）
```python
torch.optim.Adam(model.parameters(), lr=1e-3)
```
- 结合「动量 + 自适应学习率」，**每个参数单独调整学习步长**。
- 优点：
  收敛速度远快于原生 SGD；对初始学习率不那么敏感；调参简单。
- 缺点：泛化能力略弱于纯 SGD。
- 适用：**绝大多数深度学习任务、入门练习、快速实验**（你的场景首选）。

## 4. AdamW（带权重衰减的 Adam，现在工业界主流）
```python
torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
```
- 修正了 Adam 权重衰减的实现缺陷，正则化效果更标准。
- 目前 **Transformer、CV、NLP 大模型标配**。
- 推荐：正式训练、想进一步防过拟合优先选 AdamW。

## 5. RMSprop
```python
torch.optim.RMSprop(model.parameters(), lr=1e-3)
```
- 早期自适应学习率优化器，思路和 Adam 二阶矩类似。
- 适用：RNN、时序任务；现在图像分类用得少。
## 横向对比总结表
| 优化器 | 核心特点 | 收敛速度 | 调参难度 | 推荐场景 |
|--------|----------|----------|----------|----------|
| SGD | 无自适应、简单 | 慢 | 高 | 大数据、追求泛化 |
| SGD+Momentum | 加惯性 | 中 | 中 | 传统CV、竞赛 |
| Adam | 自适应学习率+动量 | 快 | 低 | 入门、快速实验 |
| AdamW | 优化权重衰减的Adam | 快 | 低 | 工程落地、大模型 |
| RMSprop | 梯度平方平滑 | 中快 | 中 | 循环网络RNN |

---

# 运行方案示例
## 方案1：标准 Adam
```python
import torch
from torch import nn

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

## 方案2：Adam + 权重衰减
```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

## 方案3：工业级 AdamW
```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

---
