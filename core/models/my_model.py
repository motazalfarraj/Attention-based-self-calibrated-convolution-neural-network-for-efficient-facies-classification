import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import trunc_normal_, DropPath
import torch.nn.functional as F
from functools import partial
from einops.layers.torch import Rearrange
from einops import reduce, rearrange
import core.models.layers as layers
import core.models.gates as gates


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels,
                 stride=1, groups=1, width_per_group=64, rate=0.3, sd=0.0,
                 reduction=16, **block_kwargs):
        super(BasicBlock, self).__init__()

        if groups != 1 or width_per_group != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        width = int(channels * (width_per_group / 64.)) * groups

        self.rate = rate

        self.shortcut = []
        if stride != 1 or in_channels != channels * self.expansion:
            self.shortcut.append(layers.conv1x1(in_channels, channels * self.expansion, stride=stride))
            self.shortcut.append(layers.bn(channels * self.expansion))
        self.shortcut = nn.Sequential(*self.shortcut)

        self.conv1 = nn.Sequential(
            layers.conv3x3(in_channels, width, stride=stride),
            layers.bn(width),
            nn.PReLU(),
        )
        self.conv2 = nn.Sequential(
            layers.conv3x3(width, channels * self.expansion),
            layers.bn(channels * self.expansion),
        )

        self.relu = nn.PReLU()
        self.sd = layers.DropPath(sd) if sd > 0.0 else nn.Identity()
        self.gate = gates.ChannelGate(channels * self.expansion, reduction, max_pool=False)

    def forward(self, x):
        skip = self.shortcut(x)

        x = self.conv1(x)
        x = F.dropout(x, p=self.rate)
        x = self.conv2(x)
        x = self.gate(x)

        x = self.sd(x) + skip
        x = self.relu(x)

        return x

    def extra_repr(self):
        return "rate=%.3e" % self.rate

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels,
                 stride=1, groups=1, width_per_group=64, rate=0.3, sd=0.0,
                 reduction=16, **block_kwargs):
        super(Bottleneck, self).__init__()

        width = int(channels * (width_per_group / 64.)) * groups

        self.rate = rate

        self.shortcut = []
        if stride != 1 or in_channels != channels * self.expansion:
            self.shortcut.append(layers.conv1x1(
                in_channels, channels * self.expansion, stride=stride))
            self.shortcut.append(layers.bn(channels * self.expansion))
        self.shortcut = nn.Sequential(*self.shortcut)

        self.conv1 = nn.Sequential(
            layers.conv1x1(in_channels, width),
            layers.bn(width),
            nn.PReLU(),
        )
        self.conv2 = nn.Sequential(
            layers.conv3x3(width, width, stride=stride, groups=groups),
            layers.bn(width),
            nn.PReLU(),
        )
        self.conv3 = nn.Sequential(
            layers.conv1x1(width, channels * self.expansion),
            layers.bn(channels * self.expansion),
        )

        self.relu = nn.PReLU()
        self.sd = layers.DropPath(sd) if sd > 0.0 else nn.Identity()
        self.gate = gates.ChannelGate(channels * self.expansion, reduction, max_pool=False)

    def forward(self, x):
        skip = self.shortcut(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = F.dropout(x, p=self.rate)
        x = self.conv3(x)
        x = self.gate(x)

        x = self.sd(x) + skip
        x = self.relu(x)

        return x

    def extra_repr(self):
        return "rate=%.3e" % self.rate
    
## This piece of code is obtained from: https://github.com/zhaohengyuan1/PAN with modifications 
class PAConv(nn.Module):
    def __init__(self, nf, k_size=3):
        super(PAConv, self).__init__()
        self.k2 = nn.Conv2d(nf, nf, 1)  # 1x1 convolution nf->nf
        self.sigmoid = nn.Sigmoid()
        self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 3x3 convolution
        self.k4 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 3x3 convolution
    def forward(self, x):
        y = self.k2(x)
        y = self.sigmoid(y)
        out = torch.mul(self.k3(x), y)
        out = self.k4(out)
        return out

class SCPA(nn.Module):
    """SCPA is modified from SCNet (Jiang-Jiang Liu et al. Improving Convolutional Networks with Self-Calibrated Convolutions. In CVPR, 2020)
        Github: https://github.com/MCG-NKU/SCNet
    """
    def __init__(self, nf, reduction=2, stride=1, dilation=1, drop_path=0.):
        super(SCPA, self).__init__()
        group_width = nf // reduction
        self.conv1_a = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
        self.conv1_b = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
        self.k1 = nn.Sequential(
            nn.Conv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                bias=False)
        )
        self.PAConv = PAConv(group_width)
        self.conv3 = nn.Conv2d(
            group_width * reduction, nf, kernel_size=1, bias=False)

        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()
        self.prelu4 = nn.PReLU()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        residual = x
        out_a = self.conv1_a(x)
        out_a = self.prelu1(out_a)
        out_a = self.k1(out_a)
        out_a = self.prelu2(out_a)

        out_b = self.conv1_b(x)
        out_b = self.prelu3(out_b)
        out_b = self.PAConv(out_b)
        out_b = self.prelu4(out_b)
        
        out = residual + self.drop_path(self.conv3(torch.cat([out_a, out_b], dim=1)))
        return out


## The code below is my own implementation
    
class MyModel(nn.Module):
    def __init__(self, n_classes=4):
        super(MyModel, self).__init__()
        self.n_classes = n_classes
        self.drop_path = 0.05

        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True)
        self.unpool =  nn.MaxUnpool2d(2, stride=2)

        self.conv_1 = nn.Sequential(BasicBlock(1, 32, sd=self.drop_path, rate=0.0), 
                                    SCPA(32, drop_path=self.drop_path))

        self.conv_2 = nn.Sequential(BasicBlock(32, 64, sd=self.drop_path, rate=0.0), 
                                    SCPA(64, drop_path=self.drop_path),
                                    SCPA(64, drop_path=self.drop_path))

        self.conv_3 = nn.Sequential(SCPA(64, drop_path=self.drop_path),
                                    SCPA(64, drop_path=self.drop_path))
        
        self.conv_4 = nn.Sequential(BasicBlock(64, 128, sd=self.drop_path, rate=0.0), 
                                    SCPA(128, drop_path=self.drop_path),
                                    SCPA(128, drop_path=self.drop_path),
                                    SCPA(128, drop_path=self.drop_path))
        
        self.conv_5 = nn.Sequential(SCPA(128, drop_path=self.drop_path),
                                    SCPA(128, drop_path=self.drop_path),
                                    SCPA(128, drop_path=self.drop_path))

        self.bottleneck = nn.Sequential(Bottleneck(in_channels=128, channels=128, sd=self.drop_path, rate=0.0),
                                        SCPA(512, drop_path=self.drop_path),
                                        SCPA(512, drop_path=self.drop_path),
                                        BasicBlock(in_channels=512, channels=128, sd=self.drop_path, rate=0.0))

        self.dconv_5 = nn.Sequential(SCPA(128, drop_path=self.drop_path),
                                     SCPA(128, drop_path=self.drop_path),
                                     SCPA(128, drop_path=self.drop_path))
        
        self.dconv_4 = nn.Sequential(BasicBlock(128, 64, sd=self.drop_path, rate=0.0),
                                     SCPA(64, drop_path=self.drop_path),
                                     SCPA(64, drop_path=self.drop_path),
                                     SCPA(64, drop_path=self.drop_path))
        
        self.dconv_3 = nn.Sequential(SCPA(64, drop_path=self.drop_path),
                                     SCPA(64, drop_path=self.drop_path)) 
        
        self.dconv_2 = nn.Sequential(BasicBlock(64, 32, sd=self.drop_path, rate=0.0),
                                     SCPA(32, drop_path=self.drop_path),
                                     SCPA(32, drop_path=self.drop_path),) 
        
        self.dconv_1 = SCPA(32, drop_path=self.drop_path)

        self.classify = nn.Conv2d(32, self.n_classes, kernel_size=(1,1))

        self.optimizer = torch.optim.Adam(self.parameters(), amsgrad=True)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', factor=0.5,
                                                                    patience=5,
                                                                    verbose=True)
    def forward(self, x):

        conv_1 = self.conv_1(x)
        pool_1, ind1 = self.pool(conv_1)

        conv_2 = self.conv_2(pool_1)
        pool_2, ind2 = self.pool(conv_2)

        conv_3 = self.conv_3(pool_2)
        pool_3, ind3 = self.pool(conv_3)

        conv_4 = self.conv_4(pool_3)
        pool_4, ind4 = self.pool(conv_4)

        conv_5 = self.conv_5(pool_4)
        pool_5, ind5 = self.pool(conv_5)

        bottleneck = self.bottleneck(pool_5)

        unpool_5 = self.unpool(bottleneck, ind5,conv_5.size())
        dconv_5 = self.dconv_5(unpool_5) + pool_4

        unpool_4 = self.unpool(dconv_5, ind4, conv_4.size())
        dconv_4 = self.dconv_4(unpool_4) + pool_3

        unpool_3= self.unpool(dconv_4, ind3, conv_3.size())
        dconv_3 = self.dconv_3(unpool_3) + pool_2

        unpool_2 = self.unpool(dconv_3  , ind2, conv_2.size())
        dconv_2 = self.dconv_2(unpool_2) + pool_1

        unpool_1 = self.unpool(dconv_2, ind1, conv_1.size())
        dconv_1 = self.dconv_1(unpool_1)

        out = self.classify(dconv_1)
        return out


