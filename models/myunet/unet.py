from torch import nn
from models.myunet.common import Up,Down
import torch
from models.myunet.common import ConvBlock
class  Unet(nn.Module):
    def __init__(self,in_channels=3,n_classes=1,layers=4):
        super().__init__()
        self.channels =32
        self.down_layers = [ConvBlock(in_channels=in_channels,out_channels=self.channels)]
        self.up_layers=[]
        for i in range(layers):
            self.down_layers.append(Down(in_channels=self.channels,out_channels=self.channels*2))
            self.channels=self.channels*2
        for i in range(layers):
            self.up_layers.append(Up(in_channels=self.channels,out_channels=self.channels//2))
            self.channels=self.channels//2

        self.encoder = nn.ModuleList(self.down_layers)
        self.decoder=nn.ModuleList(self.up_layers)
        self.final_conv=nn.Conv2d(self.channels,n_classes,kernel_size=1)

    def forward(self,x):
        self.features=[]
        for e in self.encoder:
            x=e(x)
            self.features.append(x)
        for i,d in enumerate(self.decoder):
            x=d(x,self.features[-i-2])
        logist=self.final_conv(x)
        return logist


if __name__=='__main__':
    x=torch.rand(1,3,512,512)
    net=Unet(in_channels=3,n_classes=1)
    logist=net(x)



