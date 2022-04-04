import torch.nn as nn
import torch
import torchvision

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
    def __init__(self, nf, reduction=2, stride=1, dilation=1, activation="relu"):
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
        out += residual
        return out


class ModSCPA(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, depth=1, activation="relu"):
        super(ModSCPA, self).__init__()
        self.kernel_size = kernel_size

        scpa_block = nn.Sequential(*[SCPA(ch_out, activation=activation) for _ in range(depth)])

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
                            nn.Conv2d(ch_in, ch_out, kernel_size = self.kernel_size, padding=(self.kernel_size-1)//2),
                            nn.BatchNorm2d(ch_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                            scpa_block)
        else:
            self.layer = nn.Sequential(
                            nn.ConvTranspose2d(ch_in, ch_out, kernel_size = self.kernel_size, padding=(self.kernel_size-1)//2, stride=(2,2)),
                            nn.BatchNorm2d(ch_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                            scpa_block)

    def forward(self, x):
        out = self.layer(x)
        return out



class ResNet_SCPANet(nn.Module):
    def __init__(self, n_classes=4):
        super(ResNet_SCPANet, self).__init__()
        self.n_classes = n_classes
        self.activation = "leakyrelu"
        self.channels_conv = [64,128,256,512]
        self.block_depth = [2,2,2,2,2]
        self.bottleneck_dim = 512
        self.bottleneck_depth = 4

        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True)
        self.unpool =  nn.MaxUnpool2d(2,stride=2)

        self.resnet = torchvision.models.resnet34(False)
        self.resnet.load_state_dict(torch.load("/home/motaz/facies/core/ms_models/resnet34-b627a593.pth"))
        # self.resnet.layer1
        self.conv_in = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu)

        self.conv_1 = nn.Sequential(self.resnet.layer1[0], SCPA(64))
        self.conv_2 = nn.Sequential(self.resnet.layer2[0],
                                    SCPA(128))
        self.conv_3 = nn.Sequential(self.resnet.layer3[0],
                                    SCPA(256))
        self.conv_4 = nn.Sequential(self.resnet.layer4[0],
                                    SCPA(512))

        self.bottleneck = ModSCPA(ch_in=self.bottleneck_dim, ch_out=self.bottleneck_dim, activation=self.activation, depth=self.bottleneck_depth)

        self.dconv_4 = ModSCPA(ch_in=self.channels_conv[-1], ch_out=self.channels_conv[-2], activation=self.activation, depth=self.block_depth[0])
        self.dconv_3 = ModSCPA(ch_in=self.channels_conv[-2], ch_out=self.channels_conv[-3], activation=self.activation, depth=self.block_depth[1])
        self.dconv_2 = ModSCPA(ch_in=self.channels_conv[-3], ch_out=self.channels_conv[-4], activation=self.activation, depth=self.block_depth[2])
        self.dconv_1 = ModSCPA(ch_in=self.channels_conv[-4], ch_out=self.channels_conv[-5], activation=self.activation, depth=self.block_depth[3])

        self.classify = nn.Conv2d(self.channels_conv[-5], self.n_classes, kernel_size=(1,1))

        self.optimizer = torch.optim.Adam(self.parameters(), amsgrad=True)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', factor=0.5,
                                                                    patience=5,
                                                                    verbose=True)

    def forward(self, x):

        conv_in = self.conv_in(x)
        conv_1 = self.conv_1(conv_in)
        conv_2 = self.conv_2(conv_1)
        conv_3 = self.conv_3(conv_2)
        conv_4 = self.conv_4(conv_3)

        bottleneck = self.bottleneck(conv_4)

        dconv_4 = self.dconv_4(bottleneck)
        dconv_3 = self.dconv_3(dconv_4)
        dconv_2 = self.dconv_2(dconv_3)
        dconv_1 = self.dconv_1(dconv_2)
        out = self.classify(dconv_1)

        print(x.shape)
        print(conv_in.shape)
        print(conv_2.shape)
        print(conv_3.shape)
        print(conv_4.shape)
        print(dconv_4.shape)
        print(dconv_3.shape)
        print(dconv_2.shape)
        print(dconv_1.shape)
        print(out.shape)

        return out
