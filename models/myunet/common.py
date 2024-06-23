
from torch import nn
import torch
import torch.nn.functional as F
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,act=nn.ReLU()):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1, stride=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1, stride=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.act = act

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.norm2(x)
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels,act=nn.ReLU()):
        super(Down, self).__init__()
        self.conv=ConvBlock(in_channels, out_channels, act=act)
        self.maxpool=nn.MaxPool2d(2)
    def forward(self, x):
        x=self.conv(x)
        x=self.maxpool(x)
        return x




class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.transpose = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=(2,2), stride=2)
        self.conv=ConvBlock(in_channels, out_channels)

    def forward(self, x1,x2):
        x1=self.transpose(x1)
        diffy=x2.size()[2]-x1.size()[2]
        diffx=x2.size()[3]-x1.size()[3]
        x1=F.pad(x1, (diffx//2, diffx-diffx//2, diffy//2, diffy-diffy//2))
        x1=torch.cat([x2,x1],dim=1)
        return self.conv(x1)
