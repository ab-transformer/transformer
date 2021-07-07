import pytorch_lightning as pl
import torch as th
import numpy as np
from torch import optim
from torchmetrics import MeanAbsoluteError, Accuracy, F1, PearsonCorrcoef
from torch.nn import functional as F

from models import MULTModel
from loss import bell_loss, bell_mse_mae_loss
from datasets import (
    load_impressionv2_dataset_all,
    load_resampled_impressionv2_dataset_all,
    load_report_impressionv2_dataset_all,
    load_report_mosi_dataset_all,
    load_report_mosei_dataset_all,
    load_report_mosei_sent_dataset_all
)

loss_dict = {"L2": F.mse_loss, "Bell": bell_loss, "BellL1L2": bell_mse_mae_loss}
opt_dict = {"Adam": optim.Adam, "SGD": optim.SGD}


class MULTModelWarped(pl.LightningModule):
    def __init__(self, hyp_params, target_names, early_stopping):
        super().__init__()
        self.model = MULTModel(hyp_params)
        self.save_hyperparameters(hyp_params)
        self.learning_rate = hyp_params.lr
        self.weight_decay = hyp_params.weight_decay
        self.target_names = target_names

        self.mae_1 = 1 - MeanAbsoluteError()
        self.acc2 = Accuracy()
        self.acc7 = Accuracy(multiclass=True)
        self.f1 = F1()
        self.coef = PearsonCorrcoef()
        self.loss = loss_dict[hyp_params.loss_fnc]
        self.opt = opt_dict[hyp_params.optim]

        self.early_stopping = early_stopping

    def forward(self, *args):
        if len(args) == 3:
            text, audio, face = args
        else:
            text, audio, face = args[0]
        return self.model(text, audio, face)[0]

    def configure_optimizers(self):
        optimizer = self.opt(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
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
        metric_values["acc2"] = self._calc_acc2(y_hat, y)
        metric_values["acc7"] = self._calc_acc7(y_hat, y)
        metric_values["f1"] = self._calc_f1(y_hat, y)
        metric_values["corr"] = self._calc_corr(y_hat, y)
        return metric_values

    def _calc_mae1_columnwise(self, y_hat, y):
        if self.target_names is None:
            return {}

        return {
            f"1mae_{name}": self.mae_1(y_hat[:, i], y[:, i])
            for i, name in enumerate(self.target_names)
        }

    def _y2bin(self, y_hat, y):
        mask = y_hat != 0
        y_hat_bin = y_hat[mask] > 0.0
        y_bin = y[mask] > 0.0
        return y_hat_bin, y_bin

    def y2r(self, y_hat, y):
        y_hat_r = th.round(y_hat).int()
        y_r = th.round(y).int()
        return y_hat_r, y_r

    def _calc_acc2(self, y_hat, y):
        y_hat_bin, y_bin = self._y2bin(y_hat, y)
        return self.acc2(y_hat_bin, y_bin)

    def _calc_acc7(self, y_hat, y):
        y_hat_r, y_r = self.y2r(y_hat, y)
        return self.acc7(y_hat_r + 3, y_r + 3)

    def _calc_f1(self, y_hat, y):
        y_hat_bin, y_bin = self._y2bin(y_hat, y)
        return self.f1(y_hat_bin, y_bin)

    def _calc_corr(self, y_hat, y):
        return np.corrcoef(y.view(-1).cpu().detach().numpy(), y_hat.view(-1).cpu().detach().numpy())[0][1]


class MULTModelWarpedAll(MULTModelWarped):
    def __init__(self, hyp_params, early_stopping):
        super().__init__(
            hyp_params, self.get_target_names(hyp_params.dataset), early_stopping
        )
        self.batch_size = hyp_params.batch_size
        self.shuffle = hyp_params.shuffle

        self.srA = hyp_params.a_sample
        self.srF = hyp_params.v_sample
        self.srT = hyp_params.l_sample
        self.is_random = hyp_params.random_sample
        self.audio_emb = hyp_params.audio_emb
        self.face_emb = hyp_params.face_emb
        self.text_emb = hyp_params.text_emb
        self.is_resampled_dataset = hyp_params.resampled
        self.dataset = hyp_params.dataset
        self.is_norm = hyp_params.norm

        if self.is_resampled_dataset:
            assert self.srA is None
            assert self.srF is None
            assert self.srT is None
            assert not self.is_random
            assert self.audio_emb == "wav2vec2"
            assert self.face_emb == "ig65m"
            assert self.text_emb == "bert"

    def get_target_names(self, dataset):
        if dataset == "impressionV2":
            raise "not implemented!"
        elif dataset == "report":
            return "OCEAN"
        elif dataset == "mosi":
            return None
        elif dataset == "mosei":
            return ["happy", "sad", "anger", "surprise", "disgust", "fear"]
        elif dataset == "mosei_sent":
            return None
        else:
            raise "Dataset not supported!"

    def prepare_data(self):
        if self.dataset == "impressionV2":
            if self.is_resampled_dataset:
                (
                    [self.train_ds, self.valid_ds, self.test_ds],
                    self.target_names,
                ) = load_resampled_impressionv2_dataset_all()
            else:
                (
                    [self.train_ds, self.valid_ds, self.test_ds],
                    self.target_names,
                ) = load_impressionv2_dataset_all(
                    self.srA,
                    self.srF,
                    self.srT,
                    self.is_random,
                    self.audio_emb,
                    self.face_emb,
                    self.text_emb,
                )
        elif self.dataset == "report":
            (
                self.train_ds,
                self.valid_ds,
                self.test_ds,
            ) = load_report_impressionv2_dataset_all(self.is_norm)
        elif self.dataset == "mosi":
            self.train_ds, self.valid_ds, self.test_ds = load_report_mosi_dataset_all(
                self.is_norm
            )
        elif self.dataset == "mosei":
            self.train_ds, self.valid_ds, self.test_ds = load_report_mosei_dataset_all(
                self.is_norm
            )
        elif self.dataset == "mosei_sent":
            self.train_ds, self.valid_ds, self.test_ds = load_report_mosei_sent_dataset_all(
                self.is_norm
            )
        else:
            raise "Dataset not supported!"

    def train_dataloader(self):
        return th.utils.data.DataLoader(
            self.train_ds,
            num_workers=0,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=self.shuffle,
        )

    def val_dataloader(self):
        return th.utils.data.DataLoader(
            self.valid_ds, num_workers=0, batch_size=self.batch_size, pin_memory=True,
        )

    def test_dataloader(self):
        return th.utils.data.DataLoader(
            self.test_ds, num_workers=0, batch_size=self.batch_size, pin_memory=True,
        )
