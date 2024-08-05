这个项目实现了一些经典的分割模型，方便大家学习和参考。
# UNet
[参考代码1:https://github.com/milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)<br/>
[参考代码2：https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets/tree/master](https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets/tree/master)
注意：<br/>
(1) 原论文的模型结构图中，卷积的padding为0，这里为了方便跳层连接，在具体实现时，padding设置为了1，使得特征层在卷积前和卷积后大小一致，仅改变通道数<br/>
(2) 原论文发表在2015年,因此BatchNorm还未发表，在本代码中，增加了BatchNorm,当batch较小时，也可以替换为InstanceNorm


# 
