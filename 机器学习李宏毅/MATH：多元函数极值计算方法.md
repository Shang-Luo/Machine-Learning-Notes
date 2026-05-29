
> [!NOTE] 根据研究
>其实在绝大多数情况下，**梯度为0**时还会有方向**可以**进行更新
>![[A0000-0001.png]]
## 梯度为0如何更新
梯度下降仅依靠一阶梯度更新参数，求解 $$\vec{\theta}^* = \arg\min_{\vec{\theta}} L(\vec{\theta})$$其一阶信息存在局限：**只能找到驻点** ,会出现**梯度为0**但不是极值点的问题。一阶梯度无法区分极小值、极大值、鞍点，这是优化收敛异常的根本原因

参数为 $\vec{\theta}$，损失函数梯度定义：

$$\nabla L(\vec{\theta}) = \left( \frac{\partial L}{\partial \theta_1},\frac{\partial L}{\partial \theta_2},...,\frac{\partial L}{\partial \theta_n} \right)^T$$

驻点满足 $\nabla L(\vec{\theta}_0) = \vec{0}$，该条件仅为极值的**必要非充分条件**。一阶信息无法判定驻点类别，因此引入二阶泰勒展开分析。

设参数增量向量 $\Delta\vec{\theta} = \vec{\theta}-\vec{\theta}_0$，多元函数在驻点$\vec{\theta}_0$ 处的**矩阵形式二阶泰勒展开**：

$$L(\vec{\theta}_0+\Delta\vec{\theta}) = L(\vec{\theta}_0) + \nabla L(\vec{\theta}_0)^T \Delta\vec{\theta} + \frac{1}{2} \begin{pmatrix} \Delta\theta_1 & \Delta\theta_2 & \dots & \Delta\theta_n \end{pmatrix} \boldsymbol{H}(\vec{\theta}_0) \begin{pmatrix} \Delta\theta_1 \\ \Delta\theta_2 \\ \vdots \\ \Delta\theta_n \end{pmatrix} + o(\|\Delta\vec{\theta}\|^2)$$

代入驻点矩阵条件 $\nabla L(\vec{\theta}_0)=\boldsymbol{0}$，舍弃高阶无穷小项，得到矩阵形式损失增量：

$$\Delta L = L(\vec{\theta}_0+\Delta\vec{\theta}) - L(\vec{\theta}_0) \approx \frac{1}{2}\Delta\vec{\theta}^T \boldsymbol{H}(\vec{\theta}_0) \Delta\vec{\theta}$$

$\nabla^2 L(\vec{\theta})$ 为**海森矩阵**，是损失函数二阶偏导数构成的对称矩阵：

$$\boldsymbol{H}(\vec{\theta}) = \nabla^2 L(\vec{\theta}) = \left[ \frac{\partial^2 L}{\partial \theta_i \partial \theta_j} \right]_{n \times n} = \begin{pmatrix} \frac{\partial^2 L}{\partial \theta_1^2} & \frac{\partial^2 L}{\partial \theta_1 \partial \theta_2} & \dots & \frac{\partial^2 L}{\partial \theta_1 \partial \theta_n} \\ \frac{\partial^2 L}{\partial \theta_2 \partial \theta_1} & \frac{\partial^2 L}{\partial \theta_2^2} & \dots & \frac{\partial^2 L}{\partial \theta_2 \partial \theta_n} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial^2 L}{\partial \theta_n \partial \theta_1} & \frac{\partial^2 L}{\partial \theta_n \partial \theta_2} & \dots & \frac{\partial^2 L}{\partial \theta_n^2} \end{pmatrix}$$

驻点类型由海森矩阵二次型 $\Delta\vec{\theta}^T \boldsymbol{H} \Delta\vec{\theta}$ 符号唯一确定，判定规则：

$$ \begin{cases} \boldsymbol{H} \text{ 正定},\Delta\vec{\theta}^T \boldsymbol{H} \Delta\vec{\theta} > 0 \implies \vec{\theta}_0 \text{ 为严格极小值点} \\ \boldsymbol{H} \text{ 负定},\Delta\vec{\theta}^T \boldsymbol{H} \Delta\vec{\theta} < 0 \implies \vec{\theta}_0 \text{ 为严格极大值点} \\ \boldsymbol{H} \text{ 不定},\exists\Delta\vec{\theta} \text{ 使二次型符号可变} \implies \vec{\theta}_0 \text{ 为鞍点} \end{cases} $$

### 即：如果H在该点是正定的，那么我们就可以认为他到达了极小值点