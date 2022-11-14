from email import contentmanager
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.utils import data

from calohitflow.modules import Encoder, Flow, PointFlow
from calohitflow.point_clouds import PointCloudDataset, collate_point_cloud


class CaloHitFlow(pl.LightningModule):
    def __init__(self, config: DictConfig, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.config = config
        self.hparams.update(dict(self.config))

        self.encoder = Encoder(**config.encoder)
        self.point_flow = PointFlow(**config.point_flow)

        self.latent_flow = Flow(**config.latent_flow)

        self.latent_dim = config.encoder.latent_dim

        self.lat_ext_means = torch.tensor([[config.pointcloud.e_hat_mean, config.pointcloud.n_hat_mean]])

        self.lat_ext_stds = torch.tensor([[config.pointcloud.e_hat_std, config.pointcloud.n_hat_std]])

        self.save_hyperparameters()

    def setup(self, stage):
        """Setup Datasets."""
        self.train_set = PointCloudDataset(self.config.data.base_path + self.config.data.train)
        self.val_set = PointCloudDataset(self.config.data.base_path + self.config.data.val)

    def _log(self, x, name, step):
        self.log(
            f"{step}_{name}",
            x.detach().mean(),
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=self.config.model.batch_size,
        )


    def _step(self, batch, step):
        """One step used for training and validation."""

        points, e_in, e_sum, n_hits = batch

        idx_i = 0
        idx = torch.empty(len(n_hits), dtype=torch.int64, device=self.device)

        for i, n in enumerate(n_hits):
            idx_i += n
            idx[i] = idx_i

        # remove pointclouds that have a cummulated size bigger than max_size
        # this is used to filter out peaks in the datasize
        filtered = idx < self.config.model.max_size
        idx = idx[filtered]
        e_in = e_in[filtered]
        e_sum = e_sum[filtered]
        n_hits = n_hits[filtered]

        points = points[:idx[-1]]


        lat_ext = self.process_extention_variables(e_sum, n_hits, e_in)
        points = self.transform_points(points)

        # Encode X
        mu, log_var = self.encoder(points, idx)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # calculate entropy
        d = self.config.encoder.latent_dim
        loss_entropy = d / 2 * (1 + np.log(2 * np.pi))
        loss_entropy += torch.log(std).sum(axis=1)
        loss_entropy = loss_entropy / n_hits
        
        self._log(-loss_entropy, "loss_entropy", step)
        
        z = torch.cat([z, lat_ext], dim=1)

        context = self.transform_input_energy(e_in)
        
        loss_prior = self.latent_flow.log_prob(z, context=context) / n_hits

        self._log(-loss_prior, "loss_prior", step)

        loss_recon = self.point_flow.log_prob(points, z, idx)
        self._log(-loss_recon, "loss_recon", step)

        elbo = -(loss_recon + loss_prior + loss_entropy).mean()

        self._log(elbo, "elbo", step)

        return elbo

    def training_step(self, batch, batch_idx):
        """The training step."""
        return self._step(batch, step="train")

    def validation_step(self, batch, batch_idx):
        """The validation step."""
        return self._step(batch, step="val")

    def _dataloader(self, train: bool) -> data.DataLoader:
        dataset = self.val_set
        if train == True:
            dataset = self.train_set

        return data.DataLoader(
            dataset,
            shuffle=True,
            batch_size=self.config.model.batch_size,
            collate_fn=collate_point_cloud,
            num_workers=self.config.model.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )


    def train_dataloader(self):
        return self._dataloader(train=True)

    def val_dataloader(self):
        return self._dataloader(train=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.model.lr)

    def process_extention_variables(
        self, e_sum: Tensor, n_hits: Tensor, e_in: Tensor
    ) -> Tensor:
        """
        This function contains the ugly hard coded transformation for n_hits and e_sum
        to make them easier handlebale for the flows
        the magic numbers are means and stds precalculated
        """
        e_hat = e_sum / e_in
        n_hat = n_hits + torch.rand_like(n_hits * 1.0)
        n_hat = n_hat / e_in.sqrt()  # poisson

        lat_ext = torch.stack([e_hat, n_hat], dim=1)

        lat_ext = (lat_ext - self.lat_ext_means.to(self.device)) \
            / self.lat_ext_stds.to(self.device)

        return lat_ext

    def deprocess_extention_variables(
        self, lat_ext, e_in: Tensor
        ) -> Tensor:
        """
        This function contains the ugly hard coded transformation for n_hits and e_sum
        to make them easier handlebale for the flows
        the magic numbers are means and stds precalculated
        """
        lat_ext = (lat_ext * self.lat_ext_stds.to(self.device)
            + self.lat_ext_means.to(self.device)) 
            
        e_hat = lat_ext[:, 0]
        n_hat = lat_ext[:, 1]

        e_sum = e_in * e_hat
        n_hits = torch.floor(n_hat * e_in.sqrt()).to(torch.int64)
        
        return e_sum, n_hits

    def transform_points(self, points: Tensor) -> Tensor:
        # add noise to z, alpha and r
        points[:, 0:3] += torch.rand_like(points[:, 0:3])
        # reduce to [0 - 1)
        points[:, 0:3] /= torch.tensor(self.config.pointcloud.dims).view(1, -1).to(self.device)

        if self.config.pointcloud.spherical:
            # calculate x, y out of spherical coordinates and 
            r = points[:, 2].clone()
            alpha = points[:, 1].clone() * 2 * torch.pi
            # switch from alpha, r to x, y
            points[:, 2] = r * torch.sin(alpha)
            points[:, 1] = r * torch.cos(alpha)

        # log the energy e
        points[:, 3] = points[:, 3].log()

        # normalize cloud
        points -= torch.tensor(self.config.pointcloud.means).view(1, -1).to(self.device)
        points /= torch.tensor(self.config.pointcloud.stds).view(1, -1).to(self.device)

        return points

    def detransform_points(self, points: Tensor) -> Tensor:
        # denormalize cloud
        points *= torch.tensor(self.config.pointcloud.stds).view(1, -1).to(self.device)
        points += torch.tensor(self.config.pointcloud.means).view(1, -1).to(self.device)

        # exp the log energy log(e)
        points[:, 3] = points[:, 3].exp()

        if self.config.pointcloud.spherical:
            # calculate spherical coordinates and out of x, y

            r = torch.sqrt(points[:, 1] * points[:, 1] + points[:, 2] * points[:, 2])
            alpha = torch.atan(points[:, 2] / points[:, 1])

            points[:, 2] = r
            points[:, 1] = alpha / (2 * torch.pi)

            r = points[:, 2].clone()
            alpha = points[:, 1].clone() * 2 * torch.pi
            # switch from alpha, r to x, y
            points[:, 2] = r * torch.sin(alpha)
            points[:, 1] = r * torch.cos(alpha)

        # bring back to ints
        torch.clamp_(points[:, 0:3], 0., 1.)

        points[:, 0:3] *= torch.tensor(self.config.pointcloud.dims).view(1, -1).to(self.device)
        # remove noise to z, alpha and r
        points[:, 0:3] = torch.floor(points[:, 0:3])
        
        return points

    def transform_input_energy(self, e_in: Tensor) -> Tensor:
        return (
            e_in.log()
            / self.config.pointcloud.e_in_width 
            - self.config.pointcloud.e_in_shift).unsqueeze(1)