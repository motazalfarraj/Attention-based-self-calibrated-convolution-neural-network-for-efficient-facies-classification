import torch.nn as nn
import torch
import numpy as np
from torch.nn.functional import pad

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
    def __init__(self, nf, reduction=2, stride=1, dilation=1, split=False):
        super(SCPA, self).__init__()
        group_width = nf // reduction
        self.split = split
        self.conv1_a = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
        self.conv1_b = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)

        self.k1 = nn.Conv2d( group_width, group_width, kernel_size=3, stride=stride,
                             padding=dilation, dilation=dilation, bias=False)
        self.PAConv = PAConv(group_width)
        self.conv3 = nn.Conv2d(
            group_width * reduction, nf, kernel_size=1, bias=False)

        self.act_a = nn.PReLU()
        self.act_b = nn.PReLU()

        self.bn = nn.BatchNorm2d(nf, eps=1e-05, momentum=0.1, affine=True)
        self.act_out = nn.PReLU()

    def forward(self, x):
        out_a = self.conv1_a(x)
        out_a = self.act_a(out_a)
        out_a = self.k1(out_a)
        out_a = self.act_a(out_a)

        out_b = self.conv1_b(x)
        out_b = self.act_b(out_b)
        out_b = self.PAConv(out_b)
        out_b = self.act_b(out_b)

        if self.split:
            return out_a, out_b
        else:
            return self.act_out(self.bn(self.conv3(torch.cat((out_a, out_b), dim=1))+ x))

class ModSCPA(nn.Module):
    def __init__(self, ch_in, kernel_size=3, depth=1, split=False):
        super(ModSCPA, self).__init__()
        self.kernel_size = kernel_size
        self.split = split

        if self.split:
            scpa_block = nn.Sequential(*[SCPA(ch_in) for _ in range(depth - 1)], SCPA(ch_in, split=True))
        else:
            scpa_block = nn.Sequential(*[SCPA(ch_in) for _ in range(depth)])

        self.layer = scpa_block

    def forward(self, x):
        if self.split:
            out_a, out_b = self.layer(x)
            return out_a, out_b
        else:
            out = self.layer(x)
            return out



class PSNet(nn.Module):
    def __init__(self, n_classes=4):
        super(PSNet, self).__init__()
        self.n_classes = n_classes
        self.activation = "leakyrelu"
        self.input_channels = 32
        self.PU = nn.PixelUnshuffle(downscale_factor=2)
        self.PS = nn.PixelShuffle(upscale_factor=2)
        self.block_depth = 2
        self.scale = 2**4

        self.conv_in = nn.Sequential(nn.Conv2d(1, self.input_channels, kernel_size=3, padding=1, bias=False),
                                    nn.BatchNorm2d(self.input_channels, eps=1e-05, momentum=0.1, affine=True),
                                    nn.PReLU())

        self.conv_1 = ModSCPA(ch_in=self.input_channels*4*1, depth=self.block_depth, split=True)
        self.conv_2 = ModSCPA(ch_in=self.input_channels*4*2, depth=self.block_depth, split=True)
        self.conv_3 = ModSCPA(ch_in=self.input_channels*4*4, depth=self.block_depth, split=True)
        self.conv_4 = ModSCPA(ch_in=self.input_channels*4*8, depth=self.block_depth, split=False)

        self.dconv_4 = ModSCPA(ch_in=self.input_channels*4*8, depth=self.block_depth, split=False)
        self.dconv_3 = ModSCPA(ch_in=self.input_channels*4*4, depth=self.block_depth, split=False)
        self.dconv_2 = ModSCPA(ch_in=self.input_channels*4*2, depth=self.block_depth, split=False)
        self.dconv_1 = ModSCPA(ch_in=self.input_channels*4*1, depth=self.block_depth, split=False)

        self.classify = nn.Conv2d(self.input_channels, self.n_classes, kernel_size=(1,1))


        self.optimizer = torch.optim.Adam(self.parameters(), amsgrad=True)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', factor=0.5, patience=5,
                                                                    verbose=True)


    def forward(self, x):
        x_size = x.size()

        pad_h = int(np.ceil(x_size[-2] / (self.scale)) * (self.scale) - x_size[-2])
        pad_w = int(np.ceil(x_size[-1] / (self.scale)) * (self.scale) - x_size[-1])
        x = pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        conv_in = self.conv_in(x)

        conv_1_pu = self.PU(conv_in)
        conv_1a, conv_1b = self.conv_1(conv_1_pu)

        conv_2_pu = self.PU(conv_1b)
        conv_2a, conv_2b = self.conv_2(conv_2_pu)

        conv_3_pu = self.PU(conv_2b)
        conv_3a, conv_3b = self.conv_3(conv_3_pu)

        conv_4_pu = self.PU(conv_3b)
        conv_4 = self.conv_4(conv_4_pu)

        dconv_4 = self.dconv_4(conv_4)
        dconv_4_ps = self.PS(dconv_4)

        dconv_3 = self.dconv_3(torch.cat((dconv_4_ps, conv_3a), dim=1))
        dconv_3_ps = self.PS(dconv_3)

        dconv_2 = self.dconv_2(torch.cat((dconv_3_ps, conv_2a), dim=1))
        dconv_2_ps = self.PS(dconv_2)

        dconv_1 = self.dconv_1(torch.cat((dconv_2_ps, conv_1a), dim=1))
        dconv_1_ps = self.PS(dconv_1)

        out = self.classify(dconv_1_ps)

        out = out[..., :out.shape[-2] - pad_h, :out.shape[-1] - pad_w]
        return out




