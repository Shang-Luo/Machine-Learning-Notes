我们在前面的部分讨论了深度学习的原理和神经网络的架构，现在我们来讨论如何进行回归分析和聚类分析
回归分析很简单，我们的输出就是回归值
## 聚类分析
### 函数及返回值定义
聚类分析我们使用one-hot编码和softmax来通过回归得到的向量来聚类分析
![[A0000-0011.png]]
再加上一个**SoftMax**函数进行有理化
$$p_i=\frac{e^{z_i}}{\sum_{k=1}^C e^{z_k}}$$ 约束：$\displaystyle\sum_{i=1}^C p_i=1,\;0<p_i<1$ 
> [!NOTE] 为什么我们使用Softmax
> ### 1.定义
> 1.分类类别$c\in\{1,2,...,C\}$，类别先验$P(Y=c)=\pi_c$；
> 2.类条件分布：$p(\boldsymbol x|Y=c)\sim \mathcal N(\boldsymbol\mu_c,\Sigma)$，**全部类别共用协方差$\Sigma$**；
> 3.贝叶斯后验：$\displaystyle P(Y=c|\boldsymbol x)=\frac{p(\boldsymbol x|Y=c)\pi_c}{\sum_{k=1}^C p(\boldsymbol x|Y=k)\pi_k}$。
> ### 2.简要推导
> 对高斯密度取对数并展开二次项，拆分变量相关项与全局常数项，整理为线性形式$\boldsymbol w_c^\top\boldsymbol x+b_c$，指数化后代入贝叶斯公式，分子分母公共常数抵消。
> ### 3.结果分析
> 1.最终后验概率为$\displaystyle P(Y=c|\boldsymbol x)=\frac{e^{\boldsymbol w_c^\top \boldsymbol x+b_c}}{\sum_{k=1}^C e^{\boldsymbol w_k^\top \boldsymbol x+b_k}}$，即Softmax表达式；
> 2.共用$\Sigma$是线性Softmax成立的关键，协方差不同则退化为二次判别；
> 3.$C=2$时公式退化为Sigmoid，契合二分类逻辑回归；
> 4.输出为合法概率分布，天然适配One-hot标签与交叉熵损失。

![[A0000-0012.png]]

### Loss 选择
我们常使用cross-entropy

> [!NOTE] 为什么使用cross-entropy
>最小化**cross-entropy（交叉熵）** 等价于最大化 **likelihood（似然估计）** 

多分类交叉熵（标签$\boldsymbol y$为one-hot，预测$\boldsymbol p$） $$\mathcal L=-\sum_{i=1}^C y_i\ln p_i$$
![[A0000-0013.png]]
改loss function居然可以改变优化的难度