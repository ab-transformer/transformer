import numpy as np
import torch as th
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, CometLogger

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback

from train import hyp_params
from lightningmodule import MULTModelWarpedAll


def train_mult(config):
    hyp_params.attn_dropout = config["attn_dropout"]
    hyp_params.attn_dropout_a = config["attn_dropout_a"]
    hyp_params.attn_dropout_v = config["attn_dropout_v"]
    hyp_params.embed_dropout = config["embed_dropout"]
    hyp_params.out_dropout = config["out_dropout"]
    hyp_params.relu_dropout = config["relu_dropout"]
    hyp_params.res_dropout = config["res_dropout"]

    hyp_params.layers = config["layers"]
    hyp_params.num_heads = config["num_heads"]
    hyp_params.project_dim = config["num_heads"] * config["head_dim"]
    hyp_params.lr = config["lr"]

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
    tune_reporter = TuneReportCallback(["valid_loss", "valid_1mae"])
    model = MULTModelWarpedAll(hyp_params, early_stopping=early_stopping)
    trainer = pl.Trainer(
        # gpus=1,
        max_epochs=hyp_params.num_epochs,
        log_every_n_steps=1,
        callbacks=[early_stopping, checkpoint, tune_reporter],
        logger=[csv_logger, comet_logger],
        limit_train_batches=hyp_params.limit,
        limit_val_batches=hyp_params.limit,
        weights_summary="full",
        weights_save_path="logs/weights",
        progress_bar_refresh_rate=0,
    )
    trainer.fit(model)

    ck = th.load(checkpoint.best_model_path)
    model.load_state_dict(ck["state_dict"])

    trainer.test(model)


config = {
    "attn_dropout": tune.choice(np.arange(0, 1, 0.1)),
    "attn_dropout_a": tune.choice(np.arange(0, 1, 0.1)),
    "attn_dropout_v": tune.choice(np.arange(0, 1, 0.1)),
    "embed_dropout": tune.choice(np.arange(0, 1, 0.1)),
    "out_dropout": tune.choice(np.arange(0, 1, 0.1)),
    "relu_dropout": tune.choice(np.arange(0, 1, 0.1)),
    "res_dropout": tune.choice(np.arange(0, 1, 0.1)),
    "layers": tune.choice([4, 5, 6]),
    "num_heads": tune.choice([4, 5, 6]),
    "head_dim": tune.randint(8, 15),
    # "project_dim": tune.choice([40, 50, 60, 70]),
    "lr": tune.loguniform(1e-5, 1e-3),
}

scheduler = ASHAScheduler(
    metric="valid_1mae", mode="max", max_t=hyp_params.num_epochs, grace_period=3
)

reporter = CLIReporter(
    metric_columns=["valid_1mae", "valid_loss", "training_iteration"]
)
# from ray.tune.integration.docker import DockerSyncer
# sync_config = tune.SyncConfig(
#     sync_to_driver=DockerSyncer)
analysis = tune.run(
    train_mult,
    resources_per_trial={"cpu": 1, "gpu": 0},
    # metric="valid_1mae",
    # mode="max",
    config=config,
    num_samples=1,
    scheduler=scheduler,
    progress_reporter=reporter,
    name="tune_lonly_asha",
    # sync_config=sync_config
)
# python hyperparam_search.py --lonly --num_epochs 30 --project_name lonly_asha
