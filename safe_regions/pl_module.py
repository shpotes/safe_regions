import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import torchmetrics.functional as M

from torchvision import models

from safe_regions.region import Region
from safe_regions.layers import track_safe_region

class ResNet(pl.LightningModule):
    def __init__(
            self,
            region_cls: Region,
            learning_rate: float = 3e-4,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.region_cls = region_cls
        self.model = models.resnet18()

    def track_regions(self):
        track_safe_region(self.model, self.region_cls)

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
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        return {
            'optimizer': optimizer,
            'lr_schedler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }
