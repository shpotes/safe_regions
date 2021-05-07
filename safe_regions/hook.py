import torch.nn as nn

def update_safe_region(region_cls):
    def hook(layer, input_tensor, _):
        (input_tensor,) = input_tensor
        if layer.training:
            if layer.__has_region:
                layer.__region.update_region(input_tensor.detach())
            else:
                layer.__has_region = True
                layer.__region = region_cls(input_tensor.detach())

    return hook

def track_safe_region(model, region_cls, target_module=nn.ReLU):
    children = list(model.children())

    for i, layer in enumerate(children):
        if not hasattr(layer, '__has_region'):
            layer.__has_region = False

        if isinstance(layer, target_module) and i > 0:
            layer.register_forward_hook(update_safe_region(region_cls))

        if list(layer.children()):
            track_safe_region(
                layer,
                region_cls,
                target_module,
            )
