import torchvision.models as models
from core.models.section_deconvnet import *
from core.models.SCPANet import *
from core.models.scratch.PS import PSNet
from core.models.scratch.UNet import DilatedUNet
from core.models.ViTNet import dnn_18


def get_model(name, pretrained, n_classes):
    model = _get_model_instance(name)

    if name in ['section_deconvnet']:
        model = model(n_classes=n_classes)
        vgg16 = models.vgg16(pretrained=pretrained)
        model.init_vgg16_params(vgg16)
    else:
        model = model(n_classes=n_classes)
    return model

def _get_model_instance(name):
    try:
        return {
            'section_deconvnet': section_deconvnet,
            'section_deconvnet_skip': section_deconvnet_skip,
            'SCPANet_skip':SCPANet_skip,
            'UNet': DilatedUNet,
            'PSNet':PSNet,}[name]
    except:
        print(f'Model {name} not available')
