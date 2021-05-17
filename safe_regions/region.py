import torch

class Region:
    def __init__(self):
        pass

    def update_region(self, detached_tensor):
        raise NotImplementedError

    def evaluate_membership(self, detached_tensor):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    @classmethod
    def from_state(cls, state):
        raise NotImplementedError

class MinMaxRegion(Region):
    def __init__(self):
        self._max = None
        self._min = None

    def update_region(self, detached_tensor: torch.Tensor):
        batch_max, _ = detached_tensor.max(dim=0)
        batch_min, _ = detached_tensor.min(dim=0)

        if self._max is None:
            self._max = batch_max.cpu()
            self._min = batch_min.cpu()
        else:
            self._max = torch.max(self._max, batch_max.cpu())
            self._min = torch.min(self._min, batch_min.cpu())

    def evaluate_membership(self, detached_tensor):
        return (self._max < detached_tensor) &  (detached_tensor < self._min)

    def get_state(self):
        return {
            '_max': self._max.clone() if self._max is not None else None,
            '_min': self._min.clone() if self._min is not None else None,
        }

    @classmethod
    def from_state(cls, state):
        region = cls()
        region._max = state['_max']
        region._min = state['_min']

        return region
