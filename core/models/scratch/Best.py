import torch.nn as nn
import torch

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
        self.k1 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=dilation,
                            dilation=dilation,bias=False)
        self.PAConv = PAConv(group_width)
        self.conv3 = nn.Conv2d(group_width * reduction, nf, kernel_size=1, bias=False)

        self.act1_a = nn.PReLU()
        self.act2_a = nn.PReLU()
        self.act1_b = nn.PReLU()
        self.act2_b = nn.PReLU()

    def forward(self, x):
        out_a = self.conv1_a(x)
        out_a = self.act1_a(out_a)
        out_a = self.k1(out_a)
        out_a = self.act2_a(out_a)

        out_b = self.conv1_b(x)
        out_b = self.act1_b(out_b)
        out_b = self.PAConv(out_b)
        out_b = self.act2_b(out_b)
        out = self.conv3(torch.cat([out_a, out_b], dim=1)) + x

        return out


class ModSCPA(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, depth=1):
        super(ModSCPA, self).__init__()
        self.kernel_size = kernel_size

        if ch_in == ch_out:
            self.in_layer = None
        elif ch_in<ch_out:
            self.in_layer = nn.Sequential(
                            nn.Conv2d(ch_in, ch_out, kernel_size=self.kernel_size,
                                      padding=(self.kernel_size - 1) // 2),
                            nn.BatchNorm2d(ch_out, eps=1e-5, momentum=0.1, affine=True),
                            nn.PReLU())
        else:
            self.in_layer = nn.Sequential(
                            nn.ConvTranspose2d(ch_in, ch_out,kernel_size=self.kernel_size,
                                               padding=(self.kernel_size - 1) // 2),
                            nn.BatchNorm2d(ch_out, eps=1e-5, momentum=0.1, affine=True),
                            nn.PReLU())
        self.scpa_block = nn.Sequential(*[SCPA(ch_out) for _ in range(depth)])
        # self.bn = nn.BatchNorm2d(ch_out, eps=1e-5, momentum=0.1, affine=True)
        # self.act = nn.PReLU()
    def forward(self, x):
        if self.in_layer is not None:
            x = self.in_layer(x)
        out = self.scpa_block(x)
        return out

class SCPANet_skip(nn.Module):
    def __init__(self, n_classes=4):
        super(SCPANet_skip, self).__init__()
        self.n_classes = n_classes
        self.channels_conv = [32,  64, 64, 128, 128, 512]
        self.block_depth = 1
        self.bottleneck_depth = 4

        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True)
        self.unpool =  nn.MaxUnpool2d(2, stride=2)

        self.conv_in = ModSCPA(ch_in=1,                     ch_out=self.channels_conv[0], depth=self.block_depth)

        self.conv_1 = ModSCPA(ch_in=self.channels_conv[0], ch_out=self.channels_conv[1], depth=self.block_depth)
        self.conv_2 = ModSCPA(ch_in=self.channels_conv[1], ch_out=self.channels_conv[2], depth=self.block_depth)
        self.conv_3 = ModSCPA(ch_in=self.channels_conv[2], ch_out=self.channels_conv[3], depth=self.block_depth)
        self.conv_4 = ModSCPA(ch_in=self.channels_conv[3], ch_out=self.channels_conv[4], depth=self.block_depth)
        self.conv_5 = ModSCPA(ch_in=self.channels_conv[4], ch_out=self.channels_conv[5], depth=self.block_depth)

        self.bottleneck = ModSCPA(ch_in=self.channels_conv[5], ch_out=self.channels_conv[5], depth=self.bottleneck_depth)

        self.dconv_5 = ModSCPA(ch_in=2*self.channels_conv[5], ch_out=self.channels_conv[4], depth=self.block_depth)
        self.dconv_4 = ModSCPA(ch_in=2*self.channels_conv[4], ch_out=self.channels_conv[3], depth=self.block_depth)
        self.dconv_3 = ModSCPA(ch_in=2*self.channels_conv[3], ch_out=self.channels_conv[2], depth=self.block_depth)
        self.dconv_2 = ModSCPA(ch_in=2*self.channels_conv[2], ch_out=self.channels_conv[1], depth=self.block_depth)
        self.dconv_1 = ModSCPA(ch_in=2*self.channels_conv[1], ch_out=self.channels_conv[0], depth=self.block_depth)

        self.classify = nn.Conv2d(self.channels_conv[0], self.n_classes, kernel_size=(1,1))

        self.optimizer = torch.optim.Adam(self.parameters(), amsgrad=True, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', factor=0.5,
                                                                    patience=5,
                                                                    verbose=True)

    def forward(self, x):

        conv_in = self.conv_in(x)

        conv_1 = self.conv_1(conv_in)
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
        dconv_5 = self.dconv_5(torch.cat((unpool_5,conv_5), dim=1))

        unpool_4 = self.unpool(dconv_5, ind4, conv_4.size())
        dconv_4 = self.dconv_4(torch.cat((unpool_4,conv_4), dim=1))

        unpool_3= self.unpool(dconv_4, ind3, conv_3.size())
        dconv_3 = self.dconv_3(torch.cat((unpool_3,conv_3), dim=1))

        unpool_2 = self.unpool(dconv_3  , ind2, conv_2.size())
        dconv_2 = self.dconv_2(torch.cat((unpool_2,conv_2), dim=1))

        unpool_1 = self.unpool(dconv_2, ind1, conv_1.size())
        dconv_1 = self.dconv_1(torch.cat((unpool_1,conv_1), dim=1))

        out = self.classify(dconv_1)
        return out
