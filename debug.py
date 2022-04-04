from core.models.SCPANet import *
from core.models.scratch.ResNet_SCPA import ResNet_SCPANet


import torchvision

net = ResNet_SCPANet(n_classes=6).cuda()
x = torchvision.models.resnet34(False)


#%%
from thop import clever_format
from thop import profile


input = torch.randn(1, 3,256,256)
macs, params = profile(model, inputs=(input, ))
macs, params = clever_format([macs, params], "%.3f")
print(macs, params)

# deconv 35.376G 84.031M
# PS Net 12.507G 15.634M
# SCPA_Skip 5.211G 12.188M