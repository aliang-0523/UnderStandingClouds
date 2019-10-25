## 脚本keras_seg_boost_like_a_cat重要函数,自定义类型说明
- `AdamAccumulate`继承自`Optimizer`,针对乞丐实验机,对梯度进行累加后再进行反向传播,`keras`框架目前不是特别熟悉,在`backend`中具体是怎么进行的,但是这种方法对于含有BN的网络效果肯定没有直接增大batch_size来的明显.可以类比`torch`中如下过程:  
```python
loss+=creterion(label,outputs)/accumulated_iter
loss.backward()
if (iter+1)%accumulated_iter==0:
    optim.step()
    optim.zero_grad()
```  

- post_process:对预测结果进行处理,检查结果中的连通域,若连通域面积过小,则去除这些噪音区域
- early_stopping:继承自Callback,在训练过程中若验证集的dice_val_loss在指定patience情况下未提升min_delta后提前终止训练过程,防止过拟合,缩短训练过程
- ReduceLROnPlateau:暂时还没完全理解透,大概就是在达到一定的瓶颈后减小学习率

## 当前的ensemble过程由于内存问题无法实现在一个完整的代码运行过程中保存多个predict结果,所以暂时将每一个train_test_split在测试集上的预测结果保存为pickle,在下一次运行时再单独进行处理
