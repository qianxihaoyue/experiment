import torch
from torch  import nn

import numpy  as np

import math
import  torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
from einops import rearrange,repeat

def get_positional_encoding(max_seq_len, embed_dim):
    # 初始化一个positional encoding
    # embed_dim: 字嵌入的维度
    # max_seq_len: 最大的序列长度
    positional_encoding = np.array([
        [pos / np.power(10000, 2 * i / embed_dim) for i in range(embed_dim)]
        if pos != 0 else np.zeros(embed_dim) for pos in range(max_seq_len)])

    positional_encoding[1:, 0::2] = np.sin(positional_encoding[1:, 0::2])  # dim 2i 偶数
    positional_encoding[1:, 1::2] = np.cos(positional_encoding[1:, 1::2])  # dim 2i+1 奇数
    return torch.tensor(positional_encoding,dtype=torch.float32)





class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x, num_heads) :
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x) :
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q, k, v) :
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out



class  SA(nn.Module):
    def __init__(self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1):
        super().__init__()
        self.attention = Attention(embedding_dim, num_heads, downsample_rate)


    def forward(self,x):
        b,c,h,w=x.shape
        x= rearrange(x, 'b c h w -> b (h w) c')
        print(x.dtype)

        self.postion_embedding =get_positional_encoding(h*w,c)
        self.postion_embedding=repeat(self.postion_embedding, 'l c -> b l c', b=b)
        x=x+self.postion_embedding.cuda()
        print(x.dtype)
        x = self.attention(x,x,x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x




class CA(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 num_heads: int,
                 downsample_rate: int = 1):
        super().__init__()
        self.attention = Attention(embedding_dim, num_heads, downsample_rate)

        self.conv1=nn.Sequential(nn.Conv2d(embedding_dim,embedding_dim,kernel_size=1,stride=2,padding=0),nn.BatchNorm2d(embedding_dim),nn.ReLU()) #[2h,2w,d]->[h,w,d] front
        self.conv2=nn.Sequential(nn.Conv2d(embedding_dim*2,embedding_dim,kernel_size=1,stride=1,padding=0),nn.BatchNorm2d(embedding_dim),nn.ReLU()) #[h,w,2d]->[h,w,d]
        self.conv3=nn.Sequential(nn.Conv2d(embedding_dim,embedding_dim,kernel_size=1,stride=1,padding=0),nn.BatchNorm2d(embedding_dim),nn.Sigmoid(),nn.ConvTranspose2d(embedding_dim,embedding_dim,kernel_size=2,stride=2))
    def forward(self,behind,front):
        b1, c1, h1, w1 = front.shape
        b2,c2,h2,w2 = behind.shape

        self.postion_embedding_1 =get_positional_encoding(h1*w1,c1)
        self.postion_embedding_2 =get_positional_encoding(h2*w2,c2)

        self.postion_embedding_1 = repeat(self.postion_embedding_1, 'l c -> b l c', b=b1).cuda()
        self.postion_embedding_2 = repeat(self.postion_embedding_2, 'l c -> b l c', b=b2).cuda()
        self.postion_embedding_1=self.postion_embedding_1.reshape(b1,c1,h1,w1)
        self.postion_embedding_2=self.postion_embedding_2.reshape(b2,c2,h2,w2)

        front += self.postion_embedding_1
        behind += self.postion_embedding_2

        front_tmp=self.conv1(front)
        behind_tmp=self.conv2(behind)

        front_tmp = rearrange(front_tmp, 'b c h w -> b (h w) c')
        behind_tmp = rearrange(behind_tmp, 'b c h w -> b (h w) c')
        front_tmp = self.attention(behind_tmp, behind_tmp, front_tmp)
        front_tmp = rearrange(front_tmp, 'b (h w) c -> b c h w', h=h2, w=w2)

        front *= self.conv3(front_tmp)


        # behind = rearrange(behind, 'b (h w) c -> b c h w', h=h2, w=w2)




        return behind,front






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
        self.ca=CA(out_channels,num_heads=8,downsample_rate=1)

    def forward(self, behind, front):
        behind,front=self.ca(behind,front)
        behind = self.upsample(behind)
        diffY = front.size()[2] - behind.size()[2]
        diffX = front.size()[3] - behind.size()[3]
        behind = F.pad(behind, [diffX // 2, diffX - diffX // 2,diffY // 2, diffY - diffY // 2])
        x=torch.cat([front, behind], dim=1)
        return self.conv(x)


class UTransformer(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.conv = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)     #  //2
        self.down2 = Down(128, 256)    #  //4
        self.down3 = Down(256, 512)    #  //8
        self.down4 = Down(512, 1024)   #  //16
        self.sa=SA(1024, 8,downsample_rate=1)
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
        b, c, h, w = x5.size()
        # x5 =x5.reshape(b, c, h*w).permute(0, 2, 1)
        # x5 = self.attn(x5,x5,x5)
        # x5=x5.permute(0, 2, 1).reshape(b, c, h, w)
        # x5 = rearrange(x5, 'b c h w -> b (h w) c')
        x5 = self.sa(x5)
        # x5 = rearrange(x5, 'b (h w) c -> b c h w', h=h, w=w)
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
    model = UTransformer(in_channels=3, n_classes=1).cuda()
    model.calc_params()
    x=torch.randn((1,3,256,256)).cuda()
    out=model(x)
    print(out.shape)




