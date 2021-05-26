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
        return (self._min < detached_tensor) &  (detached_tensor < self._max)

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

class RollingMeanRegion(Region):
    def __init__(self, evaluation_sigmas=2, momentum=0.1):
        self._mean = None
        self._std = None
        self.momentum = momentum
        self.sigmas = evaluation_sigmas

    def update_region(self, detached_tensor: torch.Tensor):
        batch_mean = detached_tensor.mean(dim=0).cpu()
        batch_std = detached_tensor.std(dim=0).cpu()

        if self._mean is None:
            self._mean = batch_mean
            self._std = batch_std
        else:
            self._mean = self.momentum * self._mean + (1 - self.momentum) * batch_mean
            self._std = self.momentum * self._std + (1 - self.momentum) * batch_std

    def evaluate_membership(self, detached_tensor):
        min_val = self._mean - self.sigmas * self._std
        max_val = self._mean + self.sigmas * self._std
        
        return (min_val < detached_tensor) &  (detached_tensor < max_val)

    def get_state(self):
        return {
            '_mean': self._mean.clone() if self._max is not None else None,
            '_std': self._std.clone() if self._min is not None else None,
            'params': {
                'momentum': self.momentum,
                'evaluation_sigmas': self.sigmas
            }
        }

    @classmethod
    def from_state(cls, state):
        region = cls(**state['params'])
        region._mean = state['_mean']
        region._std = state['_std']

        return region
