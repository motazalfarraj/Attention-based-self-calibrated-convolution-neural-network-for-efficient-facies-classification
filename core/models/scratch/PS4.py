from torch.nn.functional import pad
from core.models.scratch.PA import *
import numpy as np

class ModSCPA_Down(nn.Module):
    def __init__(self, ch_in):
        super(ModSCPA_Down, self).__init__()
        self.scpa = SCPA(ch_in)
        self.PU = nn.PixelUnshuffle(downscale_factor=2)
    def forward(self, x):
        # x: ch_in
        out = self.scpa(x) #ch_in
        out = self.PU(out) #ch_in*4
        return out

class ModSCPA_Up(nn.Module):
    def __init__(self, ch_in):
        super(ModSCPA_Up, self).__init__()
        self.PS = nn.PixelShuffle(upscale_factor=2)
        self.scpa = SCPA(ch_in//4)

    def forward(self, x, x_skip):
        out = self.PS(x) + x_skip #ch_in//4
        out = self.scpa(out)
        return out

class PS4(nn.Module):

    def __init__(self, n_classes=4):
        super(PS4, self).__init__()
        self.n_classes = n_classes
        self.start_ch = 8
        self.ds_factor = 2
        self.depth = 4

        self.conv_in= nn.Sequential(nn.Conv2d(1,self.start_ch, kernel_size=(3,3), padding=1),
                                    SCPA(self.start_ch)) #[M,N,start_ch]
        self.down_1 = ModSCPA_Down(self.start_ch)  #[M/2,N/2, start_ch*4]
        self.down_2 = ModSCPA_Down(4*self.start_ch) #[M/4,N/4, start_ch*16]
        self.down_3 = ModSCPA_Down(16*self.start_ch) #[M/8,N/8, start_ch*64]
        self.down_4 = ModSCPA_Down(64*self.start_ch) #[M/16,N/16, start_ch*256]
        self.bottleneck = PAConv(256*self.start_ch,k_size=1)
        self.up_5 = ModSCPA_Up(256*self.start_ch)    #[M/8, N/8, start_ch*64]
        self.up_6 = ModSCPA_Up(64*self.start_ch)     #[M/4, N/4, start_ch*16]
        self.up_7 = ModSCPA_Up(16*self.start_ch)     #[M/2, N/2, start_ch*4]
        self.up_8 = ModSCPA_Up(4*self.start_ch)      #[M, N, start_ch]

        self.conv_out = nn.Sequential(SCPA(self.start_ch),
                                      nn.Conv2d(self.start_ch, self.n_classes, kernel_size=(1,1)))



    def forward(self, x):
        x_size = x.size()
        pad_h = int(np.ceil(x_size[-2] / (self.ds_factor ** self.depth)) * (self.ds_factor ** self.depth) - x_size[-2])
        pad_w = int(np.ceil(x_size[-1] / (self.ds_factor ** self.depth)) * (self.ds_factor ** self.depth) - x_size[-1])
        x = pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        out1 = self.conv_in(x) # ch
        out2 = self.down_1(out1) #ch*4
        out3 = self.down_2(out2) #ch*16
        out4 = self.down_3(out3) #ch*64
        out5 = self.down_4(out4) #ch*256

        bottleneck = self.bottleneck(out5)

        out6 = self.up_5(bottleneck, out4) #ch*64
        out7 = self.up_6(out6, out3) #ch*16
        out8 = self.up_7(out7, out2) #ch*4
        out9 = self.up_8(out8, out1) #ch
        out10 = self.conv_out(out9)

        out = out10[..., :out10.shape[-2] - pad_h, :out10.shape[-1] - pad_w]
        return out




