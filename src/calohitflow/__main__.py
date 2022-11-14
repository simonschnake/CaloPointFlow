import os

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from calohitflow import CaloHitFlow


def train():
    default_cfg: DictConfig | ListConfig = OmegaConf.load(os.getenv("HOME") + "/calohitflow/configs/" "calochallenge_2_config.yaml")
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(default_cfg, cli_cfg)

    calo_hit_flow = CaloHitFlow(cfg)

    trainer = pl.Trainer(
        callbacks=[
            EarlyStopping(monitor="val_elbo", mode="min", min_delta=0.00, patience=10)
        ],
        **cfg.trainer
    )

    trainer.fit(calo_hit_flow)


def validate():
    raise NotImplementedError


def generate():
    raise NotImplementedError


if __name__ == "__main__":
    train()
