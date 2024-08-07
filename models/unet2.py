import  torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init


#基础模块，特征层大小不变，特征层的通道数变多,和原论文不同,padding设置为1，并且使用了BatchNorm
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,padding=1,bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,padding=1,bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

#先最大池化，然后下采样
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2),DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)

#先进行上采样，然后通道数减半
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,diffY // 2, diffY - diffY // 2])
        x=torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet2(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.conv = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)     #  //2
        self.down2 = Down(128, 256)    #  //4
        self.down3 = Down(256, 512)    #  //8
        self.down4 = Down(512, 1024)   #  //16
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.out=nn.Conv2d(64, n_classes, kernel_size=1,stride=1,padding=0)

        self.__init_weights()

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)
        return x

    #模型参数量计算
    def calc_params(self):
        count= sum( p.numel() for p in self.parameters())
        print(f"Total params: {count/1e6:.2f} M")

    #参数初始化
    def  __init_weights(self, init_type='kaiming',init_gain=0.02):
        for m in self.modules():
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find(
                    'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
                init.normal_(m.weight.data, 1.0, init_gain)
                init.constant_(m.bias.data, 0.0)



if __name__ == '__main__':
    model = UNet2(in_channels=3, n_classes=1).cuda()
    model.calc_params()
    x=torch.randn((1,3,512,512)).cuda()
    out=model(x)
    print(out.shape)
