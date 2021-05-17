import torch
import torch.nn as nn
import torch.nn.functional as F
from .region import Region

class ReLU(nn.Module):
    def __init__(self, region_cls):
        super().__init__()
        self.region = region_cls()

    def forward(self, input_tensor):
        if self.training:
            self.region.update_region(input_tensor.detach().clone())

        return F.relu(input_tensor)

    def get_state(self):
        return self.region.get_state()

    @classmethod
    def from_state(cls, region_cls, state):
        relu = cls(region_cls)
        relu.region = region_cls.from_state(state)

        return relu
