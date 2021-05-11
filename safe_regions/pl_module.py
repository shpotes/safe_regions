import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics.functional as M
from torchvision import models
from safe_regions.region import Region
from safe_regions.layers import track_safe_region

class ResNet(pl.LightningModule):
    def __init__(
            self,
            num_output: int,
            learning_rate: float,
            track_regions: bool,
            region_cls: Region
    ):
        super().__init__()

        self.save_hyperparameters()
        self.model = models.resnet18()
        self.model.fc = nn.Linear(512, num_output)

        if track_regions:
            track_safe_region(self.model, region_cls)

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
        optim = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=200)

        return {
            'optimizer': optim,
            'lr_schedler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }
