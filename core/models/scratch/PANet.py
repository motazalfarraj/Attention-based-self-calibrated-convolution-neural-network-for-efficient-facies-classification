from core.models.scratch.PA import *
import numpy as np
from torch.nn.functional import pad


class ConvBlock(nn.Module):
    def __init__(self, ch_in, scale=2, depth=1):
        super(ConvBlock, self).__init__()
        self.pool = nn.PixelUnshuffle(scale)
        self.scpa = SCPA(scale**2*ch_in)
        self.unpool= nn.PixelShuffle(scale)
        layers = [SCPA(ch_in) for i in range(depth-1)]
        self.conv = nn.Sequential(*layers)
        self.scale = scale

    def forward(self, x):
        x_size = x.size()
        pad_h = int(np.ceil(x_size[-2] / (self.scale)) * (self.scale) - x_size[-2])
        pad_w = int(np.ceil(x_size[-1] / (self.scale)) * (self.scale) - x_size[-1])
        x = pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        if self.scale>1:
            x_pool = self.pool(x)
            x_conv = self.scpa(x_pool)
            x_unpool = self.unpool(x_conv)
            out = self.conv(x_unpool)
        else:
            out = self.conv(x)

        out = out[..., :out.shape[-2] - pad_h, :out.shape[-1] - pad_w]

        return out


class PANet(nn.Module):
    def __init__(self, n_classes=4):
        super(PANet, self).__init__()
        self.n_classes = n_classes
        self.BlockDepth = 10

        self.conv_in = nn.Sequential(nn.Conv2d(1, self.n_classes, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(self.n_classes, eps=1e-05, momentum=0.1, affine=True),
                                     nn.PReLU(),
                                     ConvBlock(ch_in=self.n_classes, scale=1, depth=1))

        self.conv_1 = ConvBlock(ch_in=self.n_classes, scale=2, depth=self.BlockDepth)
        self.conv_2 = ConvBlock(ch_in=self.n_classes, scale=4, depth=self.BlockDepth)
        self.conv_3 = ConvBlock(ch_in=self.n_classes, scale=8, depth=self.BlockDepth)
        self.conv_4 = ConvBlock(ch_in=self.n_classes, scale=16, depth=self.BlockDepth)

        self.combine =  ConvBlock(ch_in=n_classes, scale=1, depth=self.BlockDepth)

        self.classify = nn.Conv2d(self.n_classes, self.n_classes, kernel_size=(1,1))

    def forward(self, x):

        conv_in = self.conv_in(x)
        conv_1 = self.conv_1(conv_1)
        conv_2 = self.conv_2(conv_2)
        conv_3 = self.conv_3(conv_3)
        conv_4 = self.conv_4(conv_4)

        combine = conv_in+conv_1+conv_2+conv_3+conv_4
        combine = self.combine(combine)+combine
        out = self.classify(combine)

        return out
