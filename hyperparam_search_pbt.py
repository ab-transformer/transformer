import os

import torch as th
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, CometLogger

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

from train import hyp_params
from lightningmodule import MULTModelWarpedAll


def train_mult(config, checkpoint_dir=None):
    hyp_params.attn_dropout = config["attn_dropout"]
    hyp_params.attn_dropout_a = config["attn_dropout_a"]
    hyp_params.attn_dropout_v = config["attn_dropout_v"]
    hyp_params.embed_dropout = config["embed_dropout"]
    hyp_params.out_dropout = config["out_dropout"]
    hyp_params.relu_dropout = config["relu_dropout"]
    hyp_params.res_dropout = config["res_dropout"]

    # hyp_params.layers = int(config["layers"])
    # hyp_params.num_heads = int(config["num_heads"])
    # hyp_params.project_dim = int(config["num_heads"]) * int(config["head_dim"])
    hyp_params.lr = 10 ** config["lr_log"]
    hyp_params.weight_decay = 10 ** config["weight_decay_log"]

    comet_logger = CometLogger(
        api_key="cgss7piePhyFPXRw1J2uUEjkQ",
        workspace="transformer",
        project_name=hyp_params.project_name,
        save_dir="logs/comet_ml",
    )
    experiement_key = comet_logger.experiment.get_key()
    csv_logger = CSVLogger("logs/csv", name=experiement_key)
    early_stopping = EarlyStopping(
        monitor="valid_1mae", patience=10, verbose=True, mode="max"
    )
    checkpoint = ModelCheckpoint(save_top_k=1, monitor="valid_1mae", mode="max")
    # tune_reporter = TuneReportCallback(["valid_loss", "valid_1mae"])
    tune_checkpoint_reporter = TuneReportCheckpointCallback(
        metrics=["valid_loss", "valid_1mae"]
    )

    model = MULTModelWarpedAll(hyp_params, early_stopping=early_stopping)
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=hyp_params.num_epochs,
        log_every_n_steps=1,
        callbacks=[early_stopping, checkpoint, tune_checkpoint_reporter],
        logger=[csv_logger, comet_logger],
        limit_train_batches=hyp_params.limit,
        limit_val_batches=hyp_params.limit,
        weights_summary="full",
        weights_save_path="logs/weights",
        progress_bar_refresh_rate=0,
    )

    if checkpoint_dir is not None:
        ck = th.load(os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(ck["state_dict"])
        trainer.current_epoch = ck["epoch"]

    trainer.fit(model)

    ck = th.load(checkpoint.best_model_path)
    model.load_state_dict(ck["state_dict"])

    trainer.test(model)


config = {
    "attn_dropout": tune.quniform(0, 1, 0.1),
    "attn_dropout_a": tune.quniform(0, 1, 0.1),
    "attn_dropout_v": tune.quniform(0, 1, 0.1),
    "embed_dropout": tune.quniform(0, 1, 0.1),
    "out_dropout": tune.quniform(0, 1, 0.1),
    "relu_dropout": tune.quniform(0, 1, 0.1),
    "res_dropout": tune.quniform(0, 1, 0.1),
    # "project_dim": tune.choice([40, 50, 60, 70]),
    "lr": tune.loguniform(1e-6, 1e-3),
    "weight_decay": tune.loguniform(1e-10, 1e-2),
}

previous_best = {
    "attn_dropout": 0.3,
    "attn_dropout_a": 0.5,
    "attn_dropout_v": 0.0,
    "embed_dropout": 0.0,
    "out_dropout": 0.2,
    "relu_dropout": 0.5,
    "res_dropout": 0.1,
    "layers": 5,
    "num_heads": 6,
    "head_dim": 14,
    "lr_log": -4,
    "weight_decay_log": -10,
}

scheduler = PopulationBasedTraining(
    perturbation_interval=4, hyperparam_mutations=config
)

reporter = CLIReporter(metric_columns=["valid_1mae", "training_iteration"])
analysis = tune.run(
    train_mult,
    metric="valid_1mae",
    mode="max",
    resources_per_trial={"cpu": 16, "gpu": 1},
    config=config,
    num_samples=500,
    scheduler=scheduler,
    progress_reporter=reporter,
    name="tune_pbt",
)
# python hyperparam_search_pbt.py --num_epochs 100 --project_dim 50 --batch_size 4 --optim SGD --project_name pbt
