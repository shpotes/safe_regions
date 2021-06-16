from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import torchmetrics.functional as M

from safe_regions.fixup_experiment.fixup_resnet_cifar import fixup_resnet20
from safe_regions.region import Region
from safe_regions.layers import ReLU

class FixupResNet(pl.LightningModule):
    def __init__(
            self,
            region_cls: Optional[Region] = None,
            num_classes: int = 10,
            learning_rate: float = 3e-4,
            decay: float = 1e-4,
    ):
        super().__init__()

        self.learning_rate = learning_rate
        self.decay = decay
        self.region_cls = region_cls

        self.model = fixup_resnet20(num_classes=num_classes)
        self.save_hyperparameters()

        if self.region_cls is not None:
            self._relu_map = {}
            self._track_regions(self.model, ('model',))

    def forward(self, input_tensor):
        return self.model(input_tensor)

    def _common_step(self, batch):
        input_tensor, target = batch
        output = self(input_tensor)
        _, pred = torch.max(output, axis=1)

        loss = F.cross_entropy(output, target)
        acc = M.accuracy(pred, target)

        return loss, acc

    def training_step(self, batch, _):
        loss, acc = self._common_step(batch)

        self.log('train_loss', loss)
        self.log('train_acc', acc)

        return loss

    def training_step(self, batch, _):
        loss, acc = self._common_step(batch)

        self.log('train_loss', loss)
        self.log('train_acc', acc)

        return loss

    def validation_step(self, batch, _):
        loss, acc = self._common_step(batch)

        self.log('val_loss', loss)
        self.log('val_acc', acc)

    def configure_optimizers(self):
        parameters_bias = [p[1] for p in self.model.named_parameters() if 'bias' in p[0]]
        parameters_scale = [p[1] for p in self.model.named_parameters() if 'scale' in p[0]]
        parameters_others = [p[1] for p in self.model.named_parameters() if not ('bias' in p[0] or 'scale' in p[0])]

        optimizer = optim.SGD(
            [{'params': parameters_bias, 'lr': self.base_lr/10.},
             {'params': parameters_scale, 'lr': self.base_lr/10.},
             {'params': parameters_others}],
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=self.decay,
        )

        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        return optimizer
    #{
    #        'optimizer': optimizer,
    #        'lr_schedler': {
    #            'scheduler': scheduler,
    #            'monitor': 'val_loss'
    #        }
    #    }

    def _track_regions(self, module, parent_name=('',)):
        for idx, (layer_name, layer) in enumerate(module.named_children()):
            if isinstance(layer, nn.ReLU):
                new_relu = ReLU(self.region_cls)
                setattr(module, layer_name, new_relu)
                self._relu_map[(idx, *parent_name)] = new_relu
            else:
                self._track_regions(layer, (*parent_name, layer_name))

    def _recover_regions(self, module, parent_name=('',)):
        for idx, (layer_name, layer) in enumerate(module.named_children()):
            if isinstance(layer, ReLU):
                new_relu = self._relu_map[(idx, *parent_name)]
                setattr(module, layer_name, new_relu)
            else:
                self._recover_regions(layer, (*parent_name, layer_name))

    def on_save_checkpoint(self, checkpoint):
        relu_map = {k: v.get_state() for k, v in self._relu_map.items()}
        checkpoint['relu_map'] = relu_map

    def on_load_checkpoint(self, checkpoint):
        self._relu_map = {k: ReLU.from_state(self.region_cls, v)
                          for k, v in checkpoint['relu_map'].items()}
        self._recover_regions(self.model, ('model',))
