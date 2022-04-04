import torch.nn as nn
import torch 
import numpy as np
from torch.nn.functional import pad
import torch.nn.functional as F

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
    def __init__(self, nf, reduction=2, stride=1, dilation=1):
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
        self.prelu = nn.PReLU()

    def forward(self, x):
        residual = x
        out_a = self.conv1_a(x)
        out_b = self.conv1_b(x)
        out_a = self.prelu(out_a)
        out_b = self.prelu(out_b)
        out_a = self.k1(out_a)
        out_b = self.PAConv(out_b)
        out_a = self.prelu(out_a)
        out_b = self.prelu(out_b)
        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        out += residual
        return out

class ConvBlock(nn.Module):
    def __init__(self, ch_in=128, kernel_size=3, padding = None, ch_up=True, ds_factor = 2):

        super(ConvBlock, self).__init__()
        self.ch_up = ch_up

        if padding is None:
            padding = (kernel_size-1)//2

        if ch_up:
            self.block = nn.Sequential(
                SCPA(ch_in),
                nn.PixelShuffle(upscale_factor=ds_factor),
            )

        else:
            self.block = nn.Sequential(
                nn.PixelUnshuffle(downscale_factor=ds_factor),
                SCPA(ch_in*ds_factor**2),
            )


    def forward(self, x):
        return self.block(x)


class PSNet(nn.Module):
    def __init__(self, n_classes=4):
        super(PSNet, self).__init__()
        self.ch_out = n_classes
        self.ch_in = n_classes
        self.depth = 4
        self.kernel_size = 3
        self.ds_factor = 2

        self.padding = (self.kernel_size - 1) // 2

        self.conv_in = nn.Sequential(
            nn.Conv2d(1, self.ch_in, kernel_size=(self.kernel_size, self.kernel_size), padding=self.padding),
            SCPA(self.ch_in),
            nn.Conv2d(self.ch_in, self.ch_in, kernel_size=(self.kernel_size, self.kernel_size), padding=self.padding)
        )

        layers_down = []
        for i in range(self.depth):
            layers_down.append(
                ConvBlock(ch_in=self.ds_factor ** (2 * i) * self.ch_in, kernel_size=self.kernel_size, ch_up=False,
                          ds_factor=self.ds_factor))

        self.block_down = nn.Sequential(*layers_down)

        layers_up = []
        for i in range(self.depth):
            layers_up.append(
                ConvBlock(ch_in=self.ds_factor ** (2 * (self.depth - i)) * self.ch_in, kernel_size=self.kernel_size,
                          ch_up=True, ds_factor=self.ds_factor))

        self.block_up = nn.Sequential(*layers_up)

        self.conv_out = nn.Sequential(
            nn.Conv2d(self.ch_in, self.ch_out, kernel_size=(1, 1)),
            SCPA(self.ch_out),
            nn.Conv2d(self.ch_out, self.ch_out, kernel_size=(1, 1))
        )

    def forward(self, x):
        x_size = x.size()
        pad_h = int(np.ceil(x_size[-2] / (self.ds_factor ** self.depth)) * (self.ds_factor ** self.depth) - x_size[-2])
        pad_w = int(np.ceil(x_size[-1] / (self.ds_factor ** self.depth)) * (self.ds_factor ** self.depth) - x_size[-1])
        x = pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        out = self.conv_in(x)
        out = self.block_down(out)

        out = self.block_up(out)
        out = self.conv_out(out)
        out = out[..., :out.shape[-2] - pad_h, :out.shape[-1] - pad_w]
        return out