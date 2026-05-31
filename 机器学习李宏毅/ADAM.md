## 普通的梯度下降法
$$
\theta_{t+1} = \theta_t - \eta \nabla J(\theta_t)~~~~~~\eta\text{为学习率}
$$

![[A0000-0008.png]]

---
## Part 1: Momentum(动量法)
现实世界中的小球不会一下停止，考虑加上他的上一刻速度向量
$$
\begin{align*}
v_{t+1} &= \beta v_t + \nabla J(\theta_t) \\
\theta_{t+1} &= \theta_t - \eta v_{t+1}\\
\text{参数:} \beta ,\eta
\end{align*}
$$
![[A0000-0009.png]]
迭代版本：
$$ \begin{cases} m^0 = 0 \\ m^1 = -\eta g^0 \\ m^2 = -\lambda \eta g^0 - \eta g^1 \\ m^3 = -\lambda^2 \eta g^0 - \lambda \eta g^1 - \eta g^2 \\ \quad \vdots \\ m^i = -\eta \sum_{k=0}^{i-1} \lambda^{i-1-k} g^k \end{cases} $$

---
## Part 2: adaptive learning rate(动态学习率)

标准梯度下降更新公式：
$$
\theta_{t+1} = \theta_t - \eta \cdot g_t
$$
其中：
- $\theta_t$：$t$ 时刻模型参数
- $\eta$：全局固定学习率
- $g_t$：$t$ 时刻损失对参数的梯度

固定学习率无法适配不同参数、不同训练阶段的更新需求，由此衍生出自适应学习率优化器。

---

### 二、AdaGrad
#### 1. 定义
AdaGrad 是**逐参数自适应学习率**优化器，对每个参数累积历史梯度平方和，动态缩放该参数的学习率；梯度大的参数更新步长变小，梯度小的参数更新步长变大。

#### 2. 公式
1. 计算当前梯度
$$
g_t = \nabla_\theta \mathcal{L}(\theta_t)
$$

2. 累积梯度平方和
$$
G_t = G_{t-1} + g_t^2
$$
$G_t$ 为从训练开始到当前步的梯度平方累加值，初始 $G_0 = 0$。

3. 参数更新
$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t} + \epsilon} \cdot g_t
$$
- $\epsilon$：极小常数（通常取 $10^{-8}$），防止分母为0
- 缺陷：$G_t$ 持续单调递增，训练后期学习率会无限趋近于0，模型提前停止更新

---

### 三、RMSProp
#### 1. 定义
RMSProp 改进了 AdaGrad 的缺陷，**使用指数移动平均(EMA)替代全量累积**，弱化久远梯度的影响，避免梯度平方和无限增大，保证训练后期仍有有效学习率，同样为逐参数自适应学习率优化器。

#### 2. 公式
1. 计算当前梯度
$$
g_t = \nabla_\theta \mathcal{L}(\theta_t)
$$

2. 梯度平方的指数移动平均
$$
E[g^2]_t = \alpha \cdot E[g^2]_{t-1} + (1-\alpha) \cdot g_t^2
$$
- $\alpha$：衰减系数，常用取值 $0.9$
- $E[g^2]_t$：$t$ 时刻梯度平方的滑动平均值

3. 参数更新
$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t} + \epsilon} \cdot g_t
$$
- $\epsilon$：防除零极小值，一般取 $10^{-8}$


## ADAM: Momentum + RMSProp

![[A0000-0010.png]]