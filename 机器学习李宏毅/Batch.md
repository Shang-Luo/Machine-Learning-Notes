## batch是什么？
batch 是一种将资料拆分为每个**epoch轮次**内每个batch**依次**梯度下降一次的技术
Batch = 一次前向 / 反向传播时，一起喂给模型的一组数据
![[A0000-0002.png]]

## 为什么需要batch?
Batch 不是必须，但用了能让**训练更快、更稳、更省资源**，是深度学习 / 机器学习的 “最优解”
1. 显存 / 内存装不下全部数据
2. 让梯度更稳定，训练不抖动
3. GPU 天生擅长并行计算
4. 带来更好的泛化能力

## 不同Batch size的效果
![[A0000-0003.png]]
 - `Batch_size = N` ：batch很少，但是很大：一次更新的时间长，但是准确
 - `Batch_size = 1` ：batch很多，但是很小：一次更新的时间短，但是不准确


## 不同Batch size的计算时间
由于存在CPU**并行计算**的能力，所以有时候大的Batch_size的运行时间和Batch_size小的单次运行时间**差不多**，甚至由于小Batch的次数多，反而运行时间大
![[A0000-0004.png]]

## 不同Batch size的optimization效果
我们发现 batchsize越大，正确率越低
![[A0000-0005.png]]
为什么？
因为每个batch使用的数据不同，每一次loss都有一些不同。遇到saddle可以在下一个batch里面规避掉
![[A0000-0006.png|657]]

## 额外发现：小的batch_size可以增加train的表现
![[A0000-0007.png]]
  小的batch因为会瞎走，更容易跳出sharp minima，大的batch不容易跳出去