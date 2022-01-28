from .mlp_mixer import mixer_s16

def get_backbone(model_name, num_classes):
    if model_name == "mixer":
        model = mixer_s16(num_classes=num_classes)
    else:
        pass
    return model