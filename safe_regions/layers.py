import torch
import torch.nn as nn
import torch.nn.functional as F

class ReLU(nn.Module):
    def __init__(self, region_cls, inplace: bool = False):
        super().__init__()
        self.region = region_cls()
        self.inplace = inplace

    def forward(self, input_tensor):
        if self.training:
            self.region.update_region(input_tensor.clone())

        return F.relu(input_tensor, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

def track_safe_region(model, region_cls):
    for layer_name, layer in model.named_children():
        if isinstance(layer, nn.ReLU):
            new_relu = ReLU(region_cls, layer.inplace)
            setattr(model, layer_name, new_relu)
        else:
            track_safe_region(layer, region_cls)
