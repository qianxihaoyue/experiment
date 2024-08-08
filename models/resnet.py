import torch
import torch.nn as nn
import torch.nn.functional as F



class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.sequential = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
                                        nn.BatchNorm2d(planes),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                                        nn.BatchNorm2d(planes),
                                        nn.ReLU(inplace=True))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride,padding=0, bias=False),
                                          nn.BatchNorm2d(self.expansion*planes))
    def forward(self, x):
        return self.sequential(x)+ self.shortcut(x)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.sequential = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,padding=1,bias=False),
                                        nn.BatchNorm2d(planes),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                                        nn.BatchNorm2d(planes),
                                        nn.ReLU(inplace=True))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut=nn.Sequential(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride,padding=0, bias=False),
                                        nn.BatchNorm2d(self.expansion*planes))
    def forward(self, x):
        return self.sequential(x)+ self.shortcut(x)



class ResNet(nn.Module):
    def __init__(self,block,num_blocks,num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1=nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(inplace=True))
        self.conv2=self.make_layer(block,64,num_blocks[0],stride=1)
        self.conv3=self.make_layer(block,128,num_blocks[1],stride=2)
        self.conv4=self.make_layer(block,256,num_blocks[2],stride=2)
        self.conv5=self.make_layer(block,512,num_blocks[3],stride=2)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512*block.expansion,num_classes)

    def make_layer(self, block, planes, num_block,stride=1):
        strides=[stride]+[1]*(num_block-1)
        layers=[]
        for stride in strides:
            layers.append(block(self.in_planes,planes,stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def resnet18():
    return ResNet(BasicBlock, [2,2,2,2])

def resnet34():
    return ResNet(BasicBlock, [3,4,6,3])

def resnet50():
    return ResNet(Bottleneck, [3,4,6,3])



if __name__ == '__main__':
    model = resnet18()
    x=torch.randn(1, 3, 224, 224)
    y=model(x)
    print(y.shape)