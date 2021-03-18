import comet_ml
import torch as th
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, CometLogger

from lightningmodule import MULTModelWarpedAll
from config import hyp_params


if __name__ == "__main__":
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
    model = MULTModelWarpedAll(hyp_params, early_stopping=early_stopping)
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=hyp_params.num_epochs,
        log_every_n_steps=1,
        callbacks=[early_stopping, checkpoint],
        logger=[csv_logger, comet_logger],
        limit_train_batches=hyp_params.limit,
        limit_val_batches=hyp_params.limit,
        weights_summary="full",
        weights_save_path="logs/weights",
    )
    trainer.fit(model)

    ck = th.load(checkpoint.best_model_path)
    model.load_state_dict(ck["state_dict"])

    trainer.test(model)
