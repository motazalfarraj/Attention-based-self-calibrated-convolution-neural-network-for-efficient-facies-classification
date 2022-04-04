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
    def __init__(self, nf, reduction=2, stride=1, dilation=1, activation="relu", residual=True):
        super(SCPA, self).__init__()
        self.residual=residual
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
        if activation.lower()=="prelu":
            self.activation = nn.PReLU()

        elif activation.lower()=="leakyrelu":
            self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out_a = self.conv1_a(x)
        out_a = self.activation(out_a)
        out_a = self.k1(out_a)
        out_a = self.activation(out_a)

        out_b = self.conv1_b(x)
        out_b = self.activation(out_b)
        out_b = self.PAConv(out_b)
        out_b = self.activation(out_b)

        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        if self.residual:
            out += residual
        return out


class ModSCPA(nn.Module):
    def __init__(self, ch_in, ch_out=None, kernel_size=3, depth=1, activation="relu", dilation=1, residual=True):
        super(ModSCPA, self).__init__()
        self.kernel_size = kernel_size
        if ch_out is None:
            ch_out = ch_in

        scpa_block = nn.Sequential(*[SCPA(ch_out, activation=activation, dilation=dilation, residual=residual) for _ in range(depth)])

        if activation.lower() == "prelu":
            self.activation = nn.PReLU()

        elif activation.lower() == "leakyrelu":
            self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)

        if ch_in == ch_out:
            self.layer = scpa_block
        elif ch_in<ch_out:
            self.layer = nn.Sequential(
                            nn.Conv2d(ch_in, ch_out, kernel_size = self.kernel_size, padding=(self.kernel_size-1)//2, bias=False),
                            nn.BatchNorm2d(ch_out, eps=1e-05, momentum=0.1, affine=True),
                            self.activation,
                            scpa_block)
        else:
            self.layer = nn.Sequential(
                            nn.ConvTranspose2d(ch_in, ch_out, kernel_size = self.kernel_size, padding=(self.kernel_size-1)//2,  bias=False),
                            nn.BatchNorm2d(ch_out, eps=1e-05, momentum=0.1, affine=True),
                            self.activation,
                            scpa_block)

    def forward(self, x):
        out = self.layer(x)
        return out

class PSNet(nn.Module):
    def __init__(self, n_classes=4):
        super(PSNet, self).__init__()
        self.n_classes = n_classes
        self.activation = "prelu"
        self.block_depth = 1
        self.scale = 2
        self.in_channels=1

        self.PU = nn.PixelUnshuffle(downscale_factor=self.scale)
        self.PS = nn.PixelShuffle(upscale_factor=self.scale)

        self.conv1 = ModSCPA(ch_in=self.in_channels*self.scale**2, ch_out=self.in_channels*self.scale**2*self.n_classes,
                             activation=self.activation,
                             depth=self.block_depth)

        self.conv2 = ModSCPA(ch_in=self.in_channels*self.scale**4*self.n_classes, activation=self.activation,
                             depth=self.block_depth)

        self.conv3 = ModSCPA(ch_in=self.in_channels*self.scale**6*self.n_classes, activation=self.activation,
                             depth=self.block_depth)

        self.conv4 = ModSCPA(ch_in=self.in_channels*self.scale**8*self.n_classes, activation=self.activation,
                             depth=self.block_depth)

        self.dconv4 = ModSCPA(ch_in=self.in_channels*self.scale**8*self.n_classes, activation=self.activation,
                              depth=self.block_depth)

        self.dconv3 = ModSCPA(ch_in=self.in_channels*self.scale**6*self.n_classes, activation=self.activation,
                              depth=self.block_depth)
        self.dconv2 = ModSCPA(ch_in=self.in_channels*self.scale**4*self.n_classes, activation=self.activation,
                              depth=self.block_depth)
        self.dconv1 = ModSCPA(ch_in=self.in_channels*self.scale**2*self.n_classes, activation=self.activation,
                              depth=self.block_depth)

        self.optimizer = torch.optim.Adam(self.parameters(), amsgrad=True)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', factor=0.5, patience=10,
                                                                    verbose=True)

    def forward(self, x):
        x_size = x.size()
        pad_h = int(np.ceil(x_size[-2] / (self.scale**8)) * (self.scale**8) - x_size[-2])
        pad_w = int(np.ceil(x_size[-1] / (self.scale**8)) * (self.scale**8) - x_size[-1])
        x = pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        x1 = self.PU(x)
        conv1 = self.conv1(x1)

        x2 = self.PU(conv1)
        conv2 = self.conv2(x2)

        x3 = self.PU(conv2)
        conv3 = self.conv3(x3)

        x4 = self.PU(conv3)
        conv4 = self.conv4(x4)

        dconv4 = self.dconv4(conv4)
        dx3 = self.PS(dconv4)

        dconv3 = self.dconv3(dx3+conv3)
        dx3 = self.PS(dconv3)

        dconv2 = self.dconv2(dx3+conv2)
        dx2 = self.PS(dconv2)

        dconv1 = self.dconv1(dx2+conv1)
        dx1 = self.PS(dconv1)

        out = dx1[..., :dx1.shape[-2] - pad_h, :dx1.shape[-1] - pad_w]
        return out
