from .resnet34 import ResNet34
from .unet import UNet

def get_backbone(model_name, num_classes):
    if model_name == 'resnet34':
        model = ResNet34(num_classes=num_classes)
    elif model_name == 'UNet':
        model = UNet(n_channels=3, n_classes=num_classes)
    return model