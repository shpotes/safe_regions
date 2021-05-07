import torch

class Region:
    def __init__(self, detached_tensor):
        self.shape = detached_tensor[0].shape

    def update_region(self, detached_tensor):
        raise NotImplementedError

    def evaluate_membership(self, detached_tensor):
        raise NotImplementedError

    def __str__(self):
        return f'region of {self.shape}'

    def __repr__(self):
        return f'region of {self.shape.__repr__()}'


class MinMaxRegion(Region):
    def __init__(self, detached_tensor):
        super().__init__(detached_tensor)

        batch_max, _ = detached_tensor.max(dim=0)
        batch_min, _ = detached_tensor.min(dim=0)

        self._max = batch_max
        self._min = batch_min

    def update_region(self, detached_tensor: torch.Tensor):
        batch_max, _ = detached_tensor.max(dim=0)
        batch_min, _ = detached_tensor.min(dim=0)

        self._max = torch.max(self._max, batch_max)
        self._min = torch.min(self._min, batch_min)

    def evaluate_membership(self, detached_tensor):
        return (self._max < detached_tensor) &  (detached_tensor < self._min)
