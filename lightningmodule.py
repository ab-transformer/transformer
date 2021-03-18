import pytorch_lightning as pl
import torch as th
from pytorch_lightning.metrics import MeanAbsoluteError
from torch.nn import functional as F

from models import MULTModel
from loss import bell_loss, bell_mse_mae_loss
from datasets import load_impressionv2_dataset_all

loss_dict = {"L2": F.mse_loss, "Bell": bell_loss, "BellL1L2": bell_mse_mae_loss}


class MULTModelWarped(pl.LightningModule):
    def __init__(self, hyp_params, target_names, early_stopping):
        super().__init__()
        self.model = MULTModel(hyp_params)
        self.save_hyperparameters(hyp_params)
        self.learning_rate = hyp_params.lr
        self.weight_decay = hyp_params.weight_decay
        self.target_names = target_names

        self.mae_1 = 1 - MeanAbsoluteError()
        self.loss = loss_dict[hyp_params.loss_fnc]

        self.early_stopping = early_stopping

    def forward(self, *args):
        if len(args) == 3:
            text, audio, face = args
        else:
            text, audio, face = args[0]
        return self.model(text, audio, face)[0]

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        metric_values = self._calc_loss_metrics(batch)
        metric_values = {f"train_{k}": v for k, v in metric_values.items()}
        metric_values[
            "debug_early_stopping_wait_count"
        ] = self.early_stopping.wait_count
        metric_values[
            "debug_early_stopping_best_score"
        ] = self.early_stopping.best_score
        self.log_dict(
            metric_values, on_step=True, on_epoch=True, prog_bar=False, logger=True
        )
        return metric_values["train_loss"]

    def validation_step(self, batch, batch_idx):
        metric_values = self._calc_loss_metrics(batch)
        metric_values = {f"valid_{k}": v for k, v in metric_values.items()}
        self.log_dict(metric_values, prog_bar=False, logger=True)
        return metric_values

    def validation_epoch_end(
        self, validation_step_outputs
    ):  # This method needs to be override in order for early stopping to work properly (pytorch lighning bug)
        pass

    def test_step(self, batch, batch_idx):
        metric_values = self._calc_loss_metrics(batch)
        metric_values = {f"test_{k}": v for k, v in metric_values.items()}
        self.log_dict(metric_values, prog_bar=False, logger=True)
        return metric_values

    def _calc_loss_metrics(self, batch):
        audio, face, text, y = batch
        y_hat = self(text, audio, face)
        loss = self.loss(y_hat, y)
        metric_values = self._calc_mae1_columnwise(y_hat, y)
        metric_values["loss"] = loss
        metric_values["1mae"] = self.mae_1(y_hat, y)
        return metric_values

    def _calc_mae1_columnwise(self, y_hat, y):
        return {
            f"1mae_{name}": self.mae_1(y_hat[:, i], y[:, i])
            for i, name in enumerate(self.target_names)
        }


class MULTModelWarpedAll(MULTModelWarped):
    def __init__(self, hyp_params, early_stopping):
        super().__init__(hyp_params, None, early_stopping)
        self.batch_size = hyp_params.batch_size
        self.shuffle = hyp_params.shuffle

    def prepare_data(self):
        (
            [self.train_ds, self.valid_ds, self.test_ds],
            self.target_names,
        ) = load_impressionv2_dataset_all()

    def train_dataloader(self):
        return th.utils.data.DataLoader(
            self.train_ds,
            num_workers=1,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=self.shuffle,
        )

    def val_dataloader(self):
        return th.utils.data.DataLoader(
            self.valid_ds, num_workers=1, batch_size=self.batch_size, pin_memory=True,
        )

    def test_dataloader(self):
        return th.utils.data.DataLoader(
            self.test_ds, num_workers=1, batch_size=self.batch_size, pin_memory=True,
        )
