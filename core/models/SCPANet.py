import torch.nn as nn
import torch
from timm.models.layers import trunc_normal_, DropPath
from core.models.ViT.models.preresnet_dnn_block import Bottleneck, BasicBottleneck

class ConvNeXtBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

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

        self.activation = nn.GELU()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

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

        out = residual + self.drop_path(self.conv3(torch.cat([out_a, out_b], dim=1)))

        return out


class ModSCPA(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, depth=1, drop_path=0.):
        super(ModSCPA, self).__init__()
        self.kernel_size = kernel_size

        scpa_block = nn.Sequential(*[SCPA(ch_out, drop_path=drop_path) for _ in range(depth)])

        self.activation = nn.GELU()

        if ch_in == ch_out:
            self.layer = scpa_block
        elif ch_in<ch_out:
            self.layer = nn.Sequential(
                            nn.Conv2d(ch_in, ch_out, kernel_size = self.kernel_size, padding=(self.kernel_size-1)//2),
                            nn.BatchNorm2d(ch_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                            self.activation,
                            scpa_block)
        else:
            self.layer = nn.Sequential(
                            nn.ConvTranspose2d(ch_in, ch_out, kernel_size = self.kernel_size, padding=(self.kernel_size-1)//2),
                            nn.BatchNorm2d(ch_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                            self.activation,
                            scpa_block)

    def forward(self, x):
        out = self.layer(x)
        return out

class SCPANet_skip(nn.Module):
    def __init__(self, n_classes=4):
        super(SCPANet_skip, self).__init__()
        self.n_classes = n_classes
        self.channels_conv = [32,  64, 64, 128, 128]
        self.block_depth = [1,2,2,3,3]
        self.drop = 0.1

        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True)
        self.unpool =  nn.MaxUnpool2d(2, stride=2)

        self.conv_1 = ModSCPA(ch_in=1,                     ch_out=self.channels_conv[0], depth=self.block_depth[0],drop_path=self.drop)
        self.conv_2 = ModSCPA(ch_in=self.channels_conv[0], ch_out=self.channels_conv[1], depth=self.block_depth[1],drop_path=self.drop)
        self.conv_3 = ModSCPA(ch_in=self.channels_conv[1], ch_out=self.channels_conv[2], depth=self.block_depth[2],drop_path=self.drop)
        self.conv_4 = ModSCPA(ch_in=self.channels_conv[2], ch_out=self.channels_conv[3], depth=self.block_depth[3],drop_path=self.drop)
        self.conv_5 = ModSCPA(ch_in=self.channels_conv[3], ch_out=self.channels_conv[4], depth=self.block_depth[4],drop_path=self.drop)

        self.bottleneck = nn.Sequential(Bottleneck(in_channels=self.channels_conv[4], channels=self.channels_conv[4]),
                                        ModSCPA(ch_in=4*self.channels_conv[4],ch_out=4*self.channels_conv[4], depth=4),
                                        BasicBottleneck(in_channels=4*self.channels_conv[4], channels=self.channels_conv[4]))

        self.dconv_5 = ModSCPA(ch_in=self.channels_conv[-1], ch_out=self.channels_conv[-2], depth=self.block_depth[0],drop_path=self.drop)
        self.dconv_4 = ModSCPA(ch_in=self.channels_conv[-2], ch_out=self.channels_conv[-3], depth=self.block_depth[1],drop_path=self.drop)
        self.dconv_3 = ModSCPA(ch_in=self.channels_conv[-3], ch_out=self.channels_conv[-4], depth=self.block_depth[2],drop_path=self.drop)
        self.dconv_2 = ModSCPA(ch_in=self.channels_conv[-4], ch_out=self.channels_conv[-5], depth=self.block_depth[3],drop_path=self.drop)
        self.dconv_1 = ModSCPA(ch_in=self.channels_conv[-5], ch_out=self.channels_conv[-5], depth=self.block_depth[4],drop_path=self.drop)

        self.classify = nn.Conv2d(self.channels_conv[-5], self.n_classes, kernel_size=(1,1))

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



# import torch.nn as nn
# import torch
# import torchvision.models as models
#
# class PAConv(nn.Module):
#     def __init__(self, nf, k_size=3):
#         super(PAConv, self).__init__()
#         self.k2 = nn.Conv2d(nf, nf, 1)  # 1x1 convolution nf->nf
#         self.sigmoid = nn.Sigmoid()
#         self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 3x3 convolution
#         self.k4 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 3x3 convolution
#     def forward(self, x):
#         y = self.k2(x)
#         y = self.sigmoid(y)
#         out = torch.mul(self.k3(x), y)
#         out = self.k4(out)
#         return out
#
# class SCPA(nn.Module):
#     """SCPA is modified from SCNet (Jiang-Jiang Liu et al. Improving Convolutional Networks with Self-Calibrated Convolutions. In CVPR, 2020)
#         Github: https://github.com/MCG-NKU/SCNet
#     """
#     def __init__(self, nf, reduction=2, stride=1, dilation=1):
#         super(SCPA, self).__init__()
#         group_width = nf // reduction
#         self.conv1_a = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
#         self.conv1_b = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
#         self.k1 = nn.Sequential(
#             nn.Conv2d(
#                 group_width, group_width, kernel_size=3, stride=stride,
#                 padding=dilation, dilation=dilation,
#                 bias=False)
#         )
#         self.PAConv = PAConv(group_width)
#         self.conv3 = nn.Conv2d(
#             group_width * reduction, nf, kernel_size=1, bias=False)
#
#         self.activation = nn.GELU()
#
#     def forward(self, x):
#         residual = x
#         out_a = self.conv1_a(x)
#         out_a = self.activation(out_a)
#         out_a = self.k1(out_a)
#         out_a = self.activation(out_a)
#
#         out_b = self.conv1_b(x)
#         out_b = self.activation(out_b)
#         out_b = self.PAConv(out_b)
#         out_b = self.activation(out_b)
#
#         out = self.conv3(torch.cat([out_a, out_b], dim=1))
#         out += residual
#         return out
#
#
# class ModSCPA(nn.Module):
#     def __init__(self, ch_in, ch_out, kernel_size=3, depth=1):
#         super(ModSCPA, self).__init__()
#         self.kernel_size = kernel_size
#         self.activation = nn.GELU()
#         scpa_block = nn.Sequential(*[SCPA(ch_out) for _ in range(depth)])
#
#         if ch_in == ch_out:
#             self.layer = scpa_block
#         elif ch_in<ch_out:
#             self.layer = nn.Sequential(
#                             nn.Conv2d(ch_in, ch_out, kernel_size = self.kernel_size, padding=(self.kernel_size-1)//2),
#                             nn.BatchNorm2d(ch_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#                             self.activation,
#                             scpa_block)
#         else:
#             self.layer = nn.Sequential(
#                             nn.ConvTranspose2d(ch_in, ch_out, kernel_size = self.kernel_size, padding=(self.kernel_size-1)//2),
#                             nn.BatchNorm2d(ch_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#                             self.activation,
#                             scpa_block)
#
#     def forward(self, x):
#         out = self.layer(x)
#         return out
#
#
#
# class SCPANet_skip(nn.Module):
#     def __init__(self, n_classes=4):
#         super(SCPANet_skip, self).__init__()
#         self.n_classes = n_classes
#         self.channels_conv = [32,  64, 128, 128, 256]
#         self.block_depth = [1,2,3,3,4]
#         self.bottleneck_dim = self.channels_conv[-1]
#         self.bottleneck_depth = 4
#         vgg16 = models.vgg16()
#         self.pool = nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True)
#         self.unpool =  nn.MaxUnpool2d(2, stride=2)
#
#         self.conv_1 = ModSCPA(ch_in=1,                     ch_out=self.channels_conv[0], depth=self.block_depth[0])
#         self.conv_2 = ModSCPA(ch_in=self.channels_conv[0], ch_out=self.channels_conv[1], depth=self.block_depth[1])
#         self.conv_3 = ModSCPA(ch_in=self.channels_conv[1], ch_out=self.channels_conv[2], depth=self.block_depth[2])
#         self.conv_4 = ModSCPA(ch_in=self.channels_conv[2], ch_out=self.channels_conv[3], depth=self.block_depth[3])
#         self.conv_5 = ModSCPA(ch_in=self.channels_conv[3], ch_out=self.channels_conv[4], depth=self.block_depth[4])
#
#         self.bottleneck = ModSCPA(ch_in=self.bottleneck_dim, ch_out=self.bottleneck_dim, depth=self.bottleneck_depth)
#
#         self.dconv_5 = ModSCPA(ch_in=self.channels_conv[-1], ch_out=self.channels_conv[-2], depth=self.block_depth[0])
#         self.dconv_4 = ModSCPA(ch_in=self.channels_conv[-2], ch_out=self.channels_conv[-3], depth=self.block_depth[1])
#         self.dconv_3 = ModSCPA(ch_in=self.channels_conv[-3], ch_out=self.channels_conv[-4], depth=self.block_depth[2])
#         self.dconv_2 = ModSCPA(ch_in=self.channels_conv[-4], ch_out=self.channels_conv[-5], depth=self.block_depth[3])
#         self.dconv_1 = ModSCPA(ch_in=self.channels_conv[-5], ch_out=self.channels_conv[-5], depth=self.block_depth[4])
#
#         self.classify = nn.Conv2d(self.channels_conv[-5], self.n_classes, kernel_size=(1,1))
#
#         self.optimizer = torch.optim.Adam(self.parameters(), amsgrad=True)
#         self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', factor=0.5,
#                                                                     patience=5,
#                                                                     verbose=True)
#     def forward(self, x):
#
#         conv_1 = self.conv_1(x)
#         pool_1, ind1 = self.pool(conv_1)
#
#         conv_2 = self.conv_2(pool_1)
#         pool_2, ind2 = self.pool(conv_2)
#
#         conv_3 = self.conv_3(pool_2)
#         pool_3, ind3 = self.pool(conv_3)
#
#         conv_4 = self.conv_4(pool_3)
#         pool_4, ind4 = self.pool(conv_4)
#
#         conv_5 = self.conv_5(pool_4)
#         pool_5, ind5 = self.pool(conv_5)
#
#         bottleneck = self.bottleneck(pool_5)
#
#         unpool_5 = self.unpool(bottleneck, ind5,conv_5.size())
#         dconv_5 = self.dconv_5(unpool_5) + pool_4
#
#         unpool_4 = self.unpool(dconv_5, ind4, conv_4.size())
#         dconv_4 = self.dconv_4(unpool_4) + pool_3
#
#         unpool_3= self.unpool(dconv_4, ind3, conv_3.size())
#         dconv_3 = self.dconv_3(unpool_3) + pool_2
#
#         unpool_2 = self.unpool(dconv_3  , ind2, conv_2.size())
#         dconv_2 = self.dconv_2(unpool_2) + pool_1
#
#         unpool_1 = self.unpool(dconv_2, ind1, conv_1.size())
#         dconv_1 = self.dconv_1(unpool_1)
#
#         out = self.classify(dconv_1)
#         return out


# import torch.nn as nn
# import torch
# from timm.models.layers import trunc_normal_, DropPath
# import numpy as np
# from torch.nn.functional import pad
# # class Stem(nn.Module):
# #     def __init__(self, ch_in, ch_out, kernel_size=3, prenorm=False):
# #         super().__init__()
# #         if ch_in <= ch_out:
# #             self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, padding=int((kernel_size-1)/2))
# #         else:
# #             self.conv = nn.ConvTranspose2d(ch_in, ch_out, kernel_size=kernel_size, padding=int((kernel_size-1)/2))
# #
# #         self.prenorm = prenorm
# #         if prenorm:
# #             self.norm = nn.LayerNorm(ch_in, eps=1e-6)
# #         else:
# #             self.norm = nn.LayerNorm(ch_out, eps=1e-6)
# #     def forward(self, x):
# #         if self.prenorm:
# #             x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
# #             x = self.norm(x)
# #             x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
# #             x = self.conv(x)
# #         else:
# #             x = self.conv(x)
# #             x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
# #             x = self.norm(x)
# #             x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
# #
# #         return x
#
# class ConvNeXtBlock(nn.Module):
#     r""" ConvNeXt Block. There are two equivalent implementations:
#     (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
#     (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
#     We use (2) as we find it slightly faster in PyTorch
#
#     Args:
#         dim (int): Number of input channels.
#         drop_path (float): Stochastic depth rate. Default: 0.0
#         layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
#     """
#
#     def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
#         super().__init__()
#         self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
#         self.norm = nn.LayerNorm(dim, eps=1e-6)
#         self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
#         self.act = nn.GELU()
#         self.pwconv2 = nn.Linear(4 * dim, dim)
#         self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
#                                   requires_grad=True) if layer_scale_init_value > 0 else None
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#
#     def forward(self, x):
#         input = x
#         x = self.dwconv(x)
#         x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
#         x = self.norm(x)
#         x = self.pwconv1(x)
#         x = self.act(x)
#         x = self.pwconv2(x)
#         if self.gamma is not None:
#             x = self.gamma * x
#         x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
#
#         x = input + self.drop_path(x)
#         return x
#
# class PAConv(nn.Module):
#     def __init__(self, nf, k_size=3):
#         super(PAConv, self).__init__()
#         self.k2 = nn.Conv2d(nf, nf, 1)  # 1x1 convolution nf->nf
#         self.sigmoid = nn.Sigmoid()
#         self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 3x3 convolution
#         self.k4 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 3x3 convolution
#     def forward(self, x):
#         y = self.k2(x)
#         y = self.sigmoid(y)
#         out = torch.mul(self.k3(x), y)
#         out = self.k4(out)
#         return out
#
# class SCPA(nn.Module):
#     """SCPA is modified from SCNet (Jiang-Jiang Liu et al. Improving Convolutional Networks with Self-Calibrated Convolutions. In CVPR, 2020)
#         Github: https://github.com/MCG-NKU/SCNet
#     """
#     def __init__(self, nf, reduction=2, stride=1, dilation=1):
#         super(SCPA, self).__init__()
#         group_width = nf // reduction
#         self.conv1_a = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
#         self.conv1_b = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
#         self.k1 = nn.Sequential(
#             nn.Conv2d(
#                 group_width, group_width, kernel_size=3, stride=stride,
#                 padding=dilation, dilation=dilation,
#                 bias=False)
#         )
#         self.convnext = ConvNeXtBlock(dim=group_width)
#         self.PAConv = PAConv(group_width)
#         self.conv3 = nn.Conv2d(
#             group_width * reduction, nf, kernel_size=1, bias=False)
#
#         self.norm = nn.LayerNorm(group_width, eps=1e-6)
#         self.activation = nn.GELU()
#
#     def forward(self, x):
#         residual = x
#         out_a = self.conv1_a(x)
#         out_a = self.convnext(out_a)
#
#         out_b = self.conv1_b(x)
#         out_b = out_b.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
#         out_b = self.norm(out_b)
#         out_b = out_b.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
#         out_b = self.PAConv(out_b)
#         out_b = self.activation(out_b)
#
#         out = self.conv3(torch.cat([out_a, out_b], dim=1))
#         out += residual
#         return out
#
#
# class ModSCPA(nn.Module):
#     def __init__(self, ch_in, ch_out, depth=1, l_type=None):
#         super(ModSCPA, self).__init__()
#         self.attn = nn.Sequential(*[SCPA(ch_out) for _ in range(depth)])
#         self.norm = nn.LayerNorm(ch_in, eps=1e-6)
#
#         if l_type.lower() == "downsample":
#             self.resample = nn.Conv2d(ch_in, ch_out, stride=2, kernel_size=3, padding=1)
#         elif l_type.lower() == "upsample":
#             self.resample = nn.ConvTranspose2d(ch_in, ch_out, stride=2, kernel_size=3, padding=1, output_padding=1)
#         else:
#             self.resample = None
#
#
#     def forward(self, x):
#         if self.resample is not None:
#             x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
#             x = self.norm(x)
#             x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
#             x = self.resample(x)
#
#         out = self.attn(x)
#         return out
#
#
#
# class SCPANet_skip(nn.Module):
#     def __init__(self, n_classes=4):
#         super(SCPANet_skip, self).__init__()
#         self.n_classes = n_classes
#         self.channels_conv = [32,  32, 64, 64, 128,128]
#
#         self.input = nn.Conv2d(in_channels=1, out_channels=self.channels_conv[0], kernel_size=3, padding=1)
#
#         self.conv_1 = ModSCPA(ch_in=self.channels_conv[0], ch_out=self.channels_conv[1], l_type="downsample")
#         self.conv_2 = ModSCPA(ch_in=self.channels_conv[1], ch_out=self.channels_conv[2], l_type="downsample")
#         self.conv_3 = ModSCPA(ch_in=self.channels_conv[2], ch_out=self.channels_conv[3], l_type="downsample", depth=2)
#         self.conv_4 = ModSCPA(ch_in=self.channels_conv[3], ch_out=self.channels_conv[4], l_type="downsample", depth=3)
#         self.conv_5 = ModSCPA(ch_in=self.channels_conv[4], ch_out=self.channels_conv[5], l_type="downsample", depth=4)
#
#         self.bn = nn.Sequential(ConvNeXtBlock(self.channels_conv[5]),
#                                 ConvNeXtBlock(self.channels_conv[5]),
#                                 ConvNeXtBlock(self.channels_conv[5]))
#
#         self.dconv_5 = ModSCPA(ch_in=self.channels_conv[-1], ch_out=self.channels_conv[-2], l_type="upsample")
#         self.dconv_4 = ModSCPA(ch_in=self.channels_conv[-2], ch_out=self.channels_conv[-3], l_type="upsample")
#         self.dconv_3 = ModSCPA(ch_in=self.channels_conv[-3], ch_out=self.channels_conv[-4], l_type="upsample",depth=2)
#         self.dconv_2 = ModSCPA(ch_in=self.channels_conv[-4], ch_out=self.channels_conv[-5], l_type="upsample",depth=3)
#         self.dconv_1 = ModSCPA(ch_in=self.channels_conv[-5], ch_out=self.channels_conv[-6], l_type="upsample",depth=4)
#
#         self.classify = nn.Conv2d(self.channels_conv[-6], self.n_classes, kernel_size=(1,1))
#
#         self.optimizer = torch.optim.Adam(self.parameters(), amsgrad=True)
#         self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', factor=0.5,
#                                                                     patience=5,
#                                                                     verbose=True)
#     def forward(self, x):
#         x_size = x.size()
#
#         pad_h = int(np.ceil(x_size[-2] / (32)) * (32) - x_size[-2])
#         pad_w = int(np.ceil(x_size[-1] / (32)) * (32) - x_size[-1])
#         x = pad(x, (0, pad_w, 0, pad_h), mode="reflect")
#
#         conv_0 = self.input(x)
#         conv_1 = self.conv_1(conv_0)
#         conv_2 = self.conv_2(conv_1)
#         conv_3 = self.conv_3(conv_2)
#         conv_4 = self.conv_4(conv_3)
#         conv_5 = self.conv_5(conv_4)
#
#         bottleneck = self.bn(conv_5)
#
#         dconv_5 = self.dconv_5(bottleneck) + conv_4
#         dconv_4 = self.dconv_4(dconv_5)+conv_3
#         dconv_3 = self.dconv_3(dconv_4)+conv_2
#         dconv_2 = self.dconv_2(dconv_3)+conv_1
#         dconv_1 = self.dconv_1(dconv_2)
#
#         out = self.classify(dconv_1)
#         out = out[..., :out.shape[-2] - pad_h, :out.shape[-1] - pad_w]
#         return out


# import torch.nn as nn
# import torch
# from timm.models.layers import trunc_normal_, DropPath
#
#
# class Stem(nn.Module):
#     def __init__(self, ch_in, ch_out, kernel_size=3, prenorm=True):
#         super().__init__()
#         if ch_in <= ch_out:
#             self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, padding=1)
#         else:
#             self.conv = nn.ConvTranspose2d(ch_in, ch_out, kernel_size=kernel_size, padding=1)
#
#         self.prenorm = prenorm
#         if prenorm:
#             self.norm = nn.LayerNorm(ch_in, eps=1e-6)
#         else:
#             self.norm = nn.LayerNorm(ch_out, eps=1e-6)
#     def forward(self, x):
#         if self.prenorm:
#             x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
#             x = self.norm(x)
#             x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
#             x = self.conv(x)
#         else:
#             x = self.conv(x)
#             x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
#             x = self.norm(x)
#             x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
#
#         return x
#
# class ConvNeXtBlock(nn.Module):
#     r""" ConvNeXt Block. There are two equivalent implementations:
#     (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
#     (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
#     We use (2) as we find it slightly faster in PyTorch
#
#     Args:
#         dim (int): Number of input channels.
#         drop_path (float): Stochastic depth rate. Default: 0.0
#         layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
#     """
#
#     def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
#         super().__init__()
#         self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
#         self.norm = nn.LayerNorm(dim, eps=1e-6)
#         self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
#         self.act = nn.GELU()
#         self.pwconv2 = nn.Linear(4 * dim, dim)
#         self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
#                                   requires_grad=True) if layer_scale_init_value > 0 else None
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#
#     def forward(self, x):
#         input = x
#         x = self.dwconv(x)
#         x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
#         x = self.norm(x)
#         x = self.pwconv1(x)
#         x = self.act(x)
#         x = self.pwconv2(x)
#         if self.gamma is not None:
#             x = self.gamma * x
#         x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
#
#         x = input + self.drop_path(x)
#         return x
#
# class PAConv(nn.Module):
#     def __init__(self, nf, k_size=3):
#         super(PAConv, self).__init__()
#         self.k2 = nn.Conv2d(nf, nf, 1)  # 1x1 convolution nf->nf
#         self.sigmoid = nn.Sigmoid()
#         self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 3x3 convolution
#         self.k4 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 3x3 convolution
#     def forward(self, x):
#         y = self.k2(x)
#         y = self.sigmoid(y)
#         out = torch.mul(self.k3(x), y)
#         out = self.k4(out)
#         return out
#
# class SCPA(nn.Module):
#     """SCPA is modified from SCNet (Jiang-Jiang Liu et al. Improving Convolutional Networks with Self-Calibrated Convolutions. In CVPR, 2020)
#         Github: https://github.com/MCG-NKU/SCNet
#     """
#     def __init__(self, nf, reduction=2, stride=1, dilation=1, activation=None):
#         super(SCPA, self).__init__()
#         group_width = nf // reduction
#         self.conv1_a = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
#         self.conv1_b = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
#         self.k1 = nn.Sequential(
#             nn.Conv2d(
#                 group_width, group_width, kernel_size=3, stride=stride,
#                 padding=dilation, dilation=dilation,
#                 bias=False)
#         )
#         self.PAConv = PAConv(group_width)
#         self.conv3 = nn.Conv2d(
#             group_width * reduction, nf, kernel_size=1, bias=False)
#
#         if activation is None:
#             self.activation = nn.GELU()
#         else:
#             self.activation = activation
#
#     def forward(self, x):
#         residual = x
#         out_a = self.conv1_a(x)
#         out_a = self.activation(out_a)
#         out_a = self.k1(out_a)
#         out_a = self.activation(out_a)
#
#         out_b = self.conv1_b(x)
#         out_b = self.activation(out_b)
#         out_b = self.PAConv(out_b)
#         out_b = self.activation(out_b)
#
#         out = self.conv3(torch.cat([out_a, out_b], dim=1))
#         out += residual
#         return out
#
#
# class ModSCPA(nn.Module):
#     def __init__(self, ch_in, ch_out, activation=None):
#         super(ModSCPA, self).__init__()
#         drop = 0.1
#
#         if activation is None:
#             self.activation = nn.GELU()
#         else:
#             self.activation = activation
#
#         self.conv = ConvNeXtBlock(ch_out, drop_path=drop)
#         self.attn = SCPA(ch_out, activation=activation)
#
#         if ch_in == ch_out:
#             self.layer = None
#         else:
#             self.layer = Stem(ch_in, ch_out)
#
#
#     def forward(self, x):
#         if self.layer is not None:
#             x = self.layer(x)
#
#         # x = self.conv(x)
#         out = self.attn(x)
#         return out
#
#
#
# class SCPANet_skip(nn.Module):
#     def __init__(self, n_classes=4):
#         super(SCPANet_skip, self).__init__()
#         self.n_classes = n_classes
#         self.activation = nn.LeakyReLU(negative_slope=0.2)
#         self.channels_conv = [64,  64, 64, 128, 128,128]
#
#         self.pool = nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True)
#         self.unpool =  nn.MaxUnpool2d(2, stride=2)
#
#         self.input = Stem(ch_in=1, ch_out=self.channels_conv[0],prenorm=False)
#
#         self.conv_1 = ModSCPA(ch_in=self.channels_conv[0], ch_out=self.channels_conv[1], activation=self.activation)
#         self.conv_2 = ModSCPA(ch_in=self.channels_conv[1], ch_out=self.channels_conv[2], activation=self.activation)
#         self.conv_3 = ModSCPA(ch_in=self.channels_conv[2], ch_out=self.channels_conv[3], activation=self.activation)
#         self.conv_4 = ModSCPA(ch_in=self.channels_conv[3], ch_out=self.channels_conv[4], activation=self.activation)
#         self.conv_5 = ModSCPA(ch_in=self.channels_conv[4], ch_out=self.channels_conv[5], activation=self.activation)
#
#         self.bn = nn.Sequential(ConvNeXtBlock(self.channels_conv[5]),
#                                 ConvNeXtBlock(self.channels_conv[5]),
#                                 ConvNeXtBlock(self.channels_conv[5]))
#
#         self.dconv_5 = ModSCPA(ch_in=self.channels_conv[-1], ch_out=self.channels_conv[-2], activation=self.activation)
#         self.dconv_4 = ModSCPA(ch_in=self.channels_conv[-2], ch_out=self.channels_conv[-3], activation=self.activation)
#         self.dconv_3 = ModSCPA(ch_in=self.channels_conv[-3], ch_out=self.channels_conv[-4], activation=self.activation)
#         self.dconv_2 = ModSCPA(ch_in=self.channels_conv[-4], ch_out=self.channels_conv[-5], activation=self.activation)
#         self.dconv_1 = ModSCPA(ch_in=self.channels_conv[-5], ch_out=self.channels_conv[-6], activation=self.activation)
#
#         self.classify = nn.Conv2d(self.channels_conv[-6], self.n_classes, kernel_size=(1,1))
#
#         self.optimizer = torch.optim.Adam(self.parameters(), amsgrad=True)
#         self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', factor=0.5,
#                                                                     patience=5,
#                                                                     verbose=True)
#     def forward(self, x):
#         conv_0 = self.input(x)
#
#         conv_1 = self.conv_1(conv_0)
#         pool_1, ind1 = self.pool(conv_1)
#
#         conv_2 = self.conv_2(pool_1)
#         pool_2, ind2 = self.pool(conv_2)
#
#         conv_3 = self.conv_3(pool_2)
#         pool_3, ind3 = self.pool(conv_3)
#
#         conv_4 = self.conv_4(pool_3)
#         pool_4, ind4 = self.pool(conv_4)
#
#         conv_5 = self.conv_5(pool_4)
#         pool_5, ind5 = self.pool(conv_5)
#
#         bottleneck = self.bn(pool_5) + pool_5
#
#         unpool_5 = self.unpool(bottleneck, ind5,conv_5.size())
#         dconv_5 = self.dconv_5(unpool_5) + pool_4
#
#         unpool_4 = self.unpool(dconv_5, ind4, conv_4.size())
#         dconv_4 = self.dconv_4(unpool_4) + pool_3
#
#         unpool_3= self.unpool(dconv_4, ind3, conv_3.size())
#         dconv_3 = self.dconv_3(unpool_3) + pool_2
#
#         unpool_2 = self.unpool(dconv_3  , ind2, conv_2.size())
#         dconv_2 = self.dconv_2(unpool_2)
#
#         unpool_1 = self.unpool(dconv_2, ind1, conv_1.size())
#         dconv_1 = self.dconv_1(unpool_1)
#
#         out = self.classify(dconv_1)
#         return out



# import torch.nn as nn
# import torch
# from timm.models.layers import trunc_normal_, DropPath
#
#
# class Stem(nn.Module):
#     def __init__(self, ch_in, ch_out, prenorm=True):
#         super().__init__()
#         if ch_in <= ch_out:
#             self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1)
#         else:
#             self.conv = nn.ConvTranspose2d(ch_in, ch_out, kernel_size=3, padding=1)
#
#         self.prenorm = prenorm
#         if prenorm:
#             self.norm = nn.LayerNorm(ch_in, eps=1e-6)
#         else:
#             self.norm = nn.LayerNorm(ch_out, eps=1e-6)
#     def forward(self, x):
#         if self.prenorm:
#             x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
#             x = self.norm(x)
#             x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
#             x = self.conv(x)
#         else:
#             x = self.conv(x)
#             x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
#             x = self.norm(x)
#             x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
#
#         return x
#
# class ConvNeXtBlock(nn.Module):
#     r""" ConvNeXt Block. There are two equivalent implementations:
#     (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
#     (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
#     We use (2) as we find it slightly faster in PyTorch
#
#     Args:
#         dim (int): Number of input channels.
#         drop_path (float): Stochastic depth rate. Default: 0.0
#         layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
#     """
#
#     def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
#         super().__init__()
#         self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
#         self.norm = nn.LayerNorm(dim, eps=1e-6)
#         self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
#         self.act = nn.GELU()
#         self.pwconv2 = nn.Linear(4 * dim, dim)
#         self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
#                                   requires_grad=True) if layer_scale_init_value > 0 else None
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#
#     def forward(self, x):
#         input = x
#         x = self.dwconv(x)
#         x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
#         x = self.norm(x)
#         x = self.pwconv1(x)
#         x = self.act(x)
#         x = self.pwconv2(x)
#         if self.gamma is not None:
#             x = self.gamma * x
#         x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
#
#         x = input + self.drop_path(x)
#         return x
#
# class PAConv(nn.Module):
#     def __init__(self, nf, k_size=3):
#         super(PAConv, self).__init__()
#         self.k2 = nn.Conv2d(nf, nf, 1)  # 1x1 convolution nf->nf
#         self.sigmoid = nn.Sigmoid()
#         self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 3x3 convolution
#         self.k4 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 3x3 convolution
#     def forward(self, x):
#         y = self.k2(x)
#         y = self.sigmoid(y)
#         out = torch.mul(self.k3(x), y)
#         out = self.k4(out)
#         return out
#
# class SCPA(nn.Module):
#     """SCPA is modified from SCNet (Jiang-Jiang Liu et al. Improving Convolutional Networks with Self-Calibrated Convolutions. In CVPR, 2020)
#         Github: https://github.com/MCG-NKU/SCNet
#     """
#     def __init__(self, nf, reduction=2, stride=1, dilation=1, activation=None):
#         super(SCPA, self).__init__()
#         group_width = nf // reduction
#         self.conv1_a = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
#         self.conv1_b = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
#         self.k1 = nn.Sequential(
#             nn.Conv2d(
#                 group_width, group_width, kernel_size=3, stride=stride,
#                 padding=dilation, dilation=dilation,
#                 bias=False)
#         )
#         self.PAConv = PAConv(group_width)
#         self.conv3 = nn.Conv2d(
#             group_width * reduction, nf, kernel_size=1, bias=False)
#
#         if activation is None:
#             self.activation = nn.GELU()
#         else:
#             self.activation = activation
#
#     def forward(self, x):
#         residual = x
#         out_a = self.conv1_a(x)
#         out_a = self.activation(out_a)
#         out_a = self.k1(out_a)
#         out_a = self.activation(out_a)
#
#         out_b = self.conv1_b(x)
#         out_b = self.activation(out_b)
#         out_b = self.PAConv(out_b)
#         out_b = self.activation(out_b)
#
#         out = self.conv3(torch.cat([out_a, out_b], dim=1))
#         out += residual
#         return out
#
#
# class ModSCPA(nn.Module):
#     def __init__(self, ch_in, ch_out, kernel_size=3, activation=None):
#         super(ModSCPA, self).__init__()
#         self.kernel_size = kernel_size
#         drop = 0.1
#
#         if activation is None:
#             self.activation = nn.GELU()
#         else:
#             self.activation = activation
#
#         self.conv = ConvNeXtBlock(ch_out, drop_path=drop)
#         self.attn = SCPA(ch_out, activation=activation)
#
#         if ch_in == ch_out:
#             self.layer = None
#         else:
#             self.layer = Stem(ch_in, ch_out)
#
#
#     def forward(self, x):
#         if self.layer is not None:
#             x = self.layer(x)
#
#         x = self.conv(x)
#         out = self.attn(x)
#         return out
#
#
#
# class SCPANet_skip(nn.Module):
#     def __init__(self, n_classes=4):
#         super(SCPANet_skip, self).__init__()
#         self.n_classes = n_classes
#         self.activation = nn.GELU()
#         self.channels_conv = [32,  32, 64, 64, 128,128]
#
#         self.pool = nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True)
#         self.unpool =  nn.MaxUnpool2d(2, stride=2)
#
#         self.input = Stem(ch_in=1, ch_out=self.channels_conv[0],prenorm=False)
#
#         self.conv_1 = ModSCPA(ch_in=self.channels_conv[0], ch_out=self.channels_conv[1], activation=self.activation)
#         self.conv_2 = ModSCPA(ch_in=self.channels_conv[1], ch_out=self.channels_conv[2], activation=self.activation)
#         self.conv_3 = ModSCPA(ch_in=self.channels_conv[2], ch_out=self.channels_conv[3], activation=self.activation)
#         self.conv_4 = ModSCPA(ch_in=self.channels_conv[3], ch_out=self.channels_conv[4], activation=self.activation)
#         self.conv_5 = ModSCPA(ch_in=self.channels_conv[4], ch_out=self.channels_conv[5], activation=self.activation)
#
#         self.bn = nn.Sequential(nn.Conv2d(in_channels=self.channels_conv[5], out_channels=4*self.channels_conv[5], kernel_size=3, padding=1),
#                                 ModSCPA(ch_in=4 * self.channels_conv[5], ch_out=4 * self.channels_conv[5]),
#                                 ModSCPA(ch_in=4 * self.channels_conv[5], ch_out=4 * self.channels_conv[5]),
#                                 nn.Conv2d(in_channels=4*self.channels_conv[5], out_channels=self.channels_conv[5], kernel_size=1))
#
#         self.dconv_5 = ModSCPA(ch_in=self.channels_conv[-1], ch_out=self.channels_conv[-2], activation=self.activation)
#         self.dconv_4 = ModSCPA(ch_in=self.channels_conv[-2], ch_out=self.channels_conv[-3], activation=self.activation)
#         self.dconv_3 = ModSCPA(ch_in=self.channels_conv[-3], ch_out=self.channels_conv[-4], activation=self.activation)
#         self.dconv_2 = ModSCPA(ch_in=self.channels_conv[-4], ch_out=self.channels_conv[-5], activation=self.activation)
#         self.dconv_1 = ModSCPA(ch_in=self.channels_conv[-5], ch_out=self.channels_conv[-6], activation=self.activation)
#
#         self.classify = nn.Conv2d(self.channels_conv[-6], self.n_classes, kernel_size=(1,1))
#
#         self.optimizer = torch.optim.Adam(self.parameters(), amsgrad=True)
#         self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', factor=0.5,
#                                                                     patience=5,
#                                                                     verbose=True)
#     def forward(self, x):
#         conv_0 = self.input(x)
#
#         conv_1 = self.conv_1(conv_0)
#         pool_1, ind1 = self.pool(conv_1)
#
#         conv_2 = self.conv_2(pool_1)
#         pool_2, ind2 = self.pool(conv_2)
#
#         conv_3 = self.conv_3(pool_2)
#         pool_3, ind3 = self.pool(conv_3)
#
#         conv_4 = self.conv_4(pool_3)
#         pool_4, ind4 = self.pool(conv_4)
#
#         conv_5 = self.conv_5(pool_4)
#         pool_5, ind5 = self.pool(conv_5)
#
#         bottleneck = self.bn(pool_5)
#
#         unpool_5 = self.unpool(bottleneck, ind5,conv_5.size())
#         dconv_5 = self.dconv_5(unpool_5) + pool_4
#
#         unpool_4 = self.unpool(dconv_5, ind4, conv_4.size())
#         dconv_4 = self.dconv_4(unpool_4) + pool_3
#
#         unpool_3= self.unpool(dconv_4, ind3, conv_3.size())
#         dconv_3 = self.dconv_3(unpool_3) + pool_2
#
#         unpool_2 = self.unpool(dconv_3  , ind2, conv_2.size())
#         dconv_2 = self.dconv_2(unpool_2) + pool_1
#
#         unpool_1 = self.unpool(dconv_2, ind1, conv_1.size())
#         dconv_1 = self.dconv_1(unpool_1)
#
#         out = self.classify(dconv_1)
#         return out
