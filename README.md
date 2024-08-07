这个项目实现了一些经典的分割模型，方便大家学习和参考。
# UNet
[论文地址：https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)<br>
[参考代码1:https://github.com/milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)<br/>
[参考代码2：https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets/tree/master](https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets/tree/master)
注意：<br/>
(1) 原论文的模型结构图中，卷积的padding为0，这里为了方便跳层连接，在具体实现时，padding设置为了1，使得特征层在卷积前和卷积后大小一致，仅改变通道数<br/>
(2) 原论文发表在2015年,因此BatchNorm还未发表，在本代码中，增加了BatchNorm,当batch较小时，也可以替换为InstanceNorm
(3) 在本实现中，上采样使用的转置卷积

# ResNet

# ViT

# U-Transformer
[论文地址:https://arxiv.org/abs/2103.06104#](https://arxiv.org/abs/2103.06104#)<br/>
在UNet的基础上增加self-attention和cross-attention
# ResUNet
在UNet的DouConv基础上增加一个残差连接即可
[参考博客:https://www.cnblogs.com/alex-bn-lee/p/15224922.html](https://www.cnblogs.com/alex-bn-lee/p/15224922.html)<br/>
[参考代码：https://github.com/Nishanksingla/UNet-with-ResBlock/blob/master/resnet34_unet_model.py](https://github.com/Nishanksingla/UNet-with-ResBlock/blob/master/resnet34_unet_model.py)<br/>
[参考代码:https://github.com/alexbnlee/ResUNet](https://github.com/alexbnlee/ResUNet)
# TransUNet



