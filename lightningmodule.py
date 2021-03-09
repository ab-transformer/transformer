import pytorch_lightning as pl
import torch as th
from pytorch_lightning.metrics import MeanAbsoluteError
from torch.nn import functional as F

from models import MULTModel


class MULTModelWarped(pl.LightningModule):
    def __init__(self, hyp_params, target_names):
        super().__init__()
        self.model = MULTModel(hyp_params)
        self.save_hyperparameters(hyp_params)
        self.learning_rate = hyp_params.lr
        self.target_names = target_names

        self.mae_1 = 1 - MeanAbsoluteError()

    def forward(self, *args):
        if len(args) == 3:
            text, audio, face = args
        else:
            text, audio, face = args[0]
        return self.model(text, audio, face)[0]

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        metric_values = self._calc_loss_metrics(batch)
        metric_values = {f"train_{k}": v for k, v in metric_values.items()}
        self.log_dict(
            metric_values, on_step=True, on_epoch=True, prog_bar=False, logger=True
        )
        self.log(
            "train_loss",
            metric_values["train_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=False,
        )
        self.log(
            "train_1mae",
            metric_values["train_1mae"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=False,
        )
        return metric_values["train_loss"]

    def validation_step(self, batch, batch_idx):
        metric_values = self._calc_loss_metrics(batch)
        metric_values = {f"valid_{k}": v for k, v in metric_values.items()}
        self.log_dict(metric_values, prog_bar=False, logger=True)
        self.log(
            "valid_loss", metric_values["valid_loss"], prog_bar=True, logger=False,
        )
        self.log(
            "valid_1mae", metric_values["valid_1mae"], prog_bar=True, logger=False,
        )
        return metric_values

    def test_step(self, batch, batch_idx):
        metric_values = self._calc_loss_metrics(batch)
        metric_values = {f"test_{k}": v for k, v in metric_values.items()}
        self.log_dict(metric_values, prog_bar=False, logger=True)
        self.log(
            "test_loss", metric_values["test_loss"], prog_bar=True, logger=False,
        )
        self.log(
            "test_1mae", metric_values["test_1mae"], prog_bar=True, logger=False,
        )
        return metric_values

    def _calc_loss_metrics(self, batch):
        audio, face, text, y = batch
        y_hat = self(text, audio, face)
        loss = F.mse_loss(y_hat, y)
        metric_values = self._calc_mae1_columnwise(y_hat, y)
        metric_values["loss"] = loss
        metric_values["1mae"] = self.mae_1(y_hat, y)
        return metric_values

    def _calc_mae1_columnwise(self, y_hat, y):
        return {
            f"1mae_{name}": self.mae_1(y_hat[:, i], y[:, i])
            for i, name in enumerate(self.target_names)
        }
