import torch
import torch.nn as nn
import torch.nn.functional as F

class ReLU(nn.Module):
    def __init__(self, region_cls):
        super().__init__()
        self.region = region_cls()

    def forward(self, input_tensor):
        if self.training:
            self.region.update_region(input_tensor.detach().clone())

        return F.relu(input_tensor)

def track_safe_region(model, region_cls):
    for layer_name, layer in model.named_children():
        if isinstance(layer, nn.ReLU):
            new_relu = ReLU(region_cls)
            setattr(model, layer_name, new_relu)
        else:
            track_safe_region(layer, region_cls)
