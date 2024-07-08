import pytorch_lightning as pl
from omegaconf import DictConfig

from .model import CaloPointFlow

from calopointflow.data import CaloDataModule2, CaloDataModule3


def train(model:str, dataset: int, cfg: DictConfig, log_dir: str) -> None:

    calo_point_flow = CaloPointFlow(cfg)

    if dataset == 2:
        calo_data = CaloDataModule2(**cfg.data)
    elif dataset == 3:
        calo_data = CaloDataModule3(**cfg.data)
    else:
        raise NameError(f"Unknown dataset {dataset}")

    logger = pl.loggers.TensorBoardLogger(
        save_dir=log_dir
    )
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="val_loss", mode="min", min_delta=0.00, patience=100
            ),
            pl.callbacks.ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_last=True,
                save_top_k=3,
                filename="{epoch}-{val_loss:.4f}",
            ),
        ],
        logger=logger,
    )

    trainer.fit(calo_point_flow, calo_data)
