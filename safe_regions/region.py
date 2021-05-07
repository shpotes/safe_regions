import torch

class Region:
    def __init__(self):
        pass

    def update_region(self, detached_tensor):
        raise NotImplementedError

    def evaluate_membership(self, detached_tensor):
        raise NotImplementedError

class MinMaxRegion(Region):
    def __init__(self):
        self._max = None
        self._min = None

    def update_region(self, detached_tensor: torch.Tensor):
        batch_max, _ = detached_tensor.max(dim=0)
        batch_min, _ = detached_tensor.min(dim=0)

        if self._max is None:
            self._max = batch_max
            self._min = batch_min
        else:
            self._max = torch.max(self._max, batch_max)
            self._min = torch.min(self._min, batch_min)

    def evaluate_membership(self, detached_tensor):
        return (self._max < detached_tensor) &  (detached_tensor < self._min)
