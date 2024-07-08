from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT

from zuko.flows import NSF

import torch
from torch_scatter import scatter
from torch import Tensor
from omegaconf import DictConfig

from calopointflow.freezables import FreezeModule, Dequantization, CDFDequantization, Normalize, MinMaxScale
from calopointflow.utils import logistic_quantile, normal_quantile, logistic_cdf, normal_cdf

from .modules import Encoder, NeuralSplineCouplingFlow, DeepSetFlow

class CaloPointFlow(FreezeModule):
    def __init__(
            self,
            config: DictConfig,
            *args: Any,
            **kwargs: Any) -> None: 
             
        super().__init__(*args, **kwargs)
        self.config = config
        self.hparams.update(dict(self.config))

        if self.config.point_dim == 3:
            self.size = (
                config.dataset.binning.z_bins, 
                config.dataset.binning.r_bins, 
            )

        elif self.config.point_dim == 4:
            self.size = (
                config.dataset.binning.z_bins, 
                config.dataset.binning.alpha_bins, 
                config.dataset.binning.r_bins, 
            )
        else:
            raise NameError(f"Unknown point_dim {self.config.point_dim}")

        self.conditional_flow = NSF(
              features = self.config.conditional_dim,
              context = 1,
              **self.config.conditional_flow
        )

        self.encoder = Encoder(
            point_dim = self.config.point_dim,
            latent_dim = self.config.latent_dim,
            **self.config.encoder
        )

        self.latent_flow = NeuralSplineCouplingFlow(
            features = self.config.latent_dim,
            context = 1 + self.config.conditional_dim,
            **self.config.latent_flow)

        if self.config.deepset_flow:
            self.point_flow = DeepSetFlow(
                features = self.config.point_dim,
                context = 1 + self.config.conditional_dim + self.config.latent_dim,
                **self.config.point_flow)
        else:
            self.point_flow = NeuralSplineCouplingFlow(
                features = self.config.point_dim,
                context = 1 + self.config.conditional_dim + self.config.latent_dim,
                **self.config.point_flow)
            
        if self.config.cdf_dequantization:
            self.register_freezable("dequantize_coords", CDFDequantization(size=self.size))
        else:
            self.register_freezable("dequantize_coords", Dequantization(size=self.size))
             
        self.register_freezable("min_max_scale_nnz", MinMaxScale(1))
        self.register_freezable("min_max_scale_vals", MinMaxScale(1))
        self.register_freezable("normalize_vals", Normalize(1))
        self.register_freezable("normalize_nnz", Normalize(1))
        self.register_freezable("normalize_e_sum", Normalize(1))
        self.register_freezable("normalize_e_in", Normalize(1))

        self._setup_coords()

        self.save_hyperparameters()

    def _step(self, batch, batch_idx, mode="train"):
        coords, vals, nnz, e_in = batch

        idx = torch.repeat_interleave(torch.arange(len(nnz), device=nnz.device), nnz)

        # hardcoded substract the minimum energy in the dataset () 
        if self.config.shift_min_energy:
            vals -= 0.5e-3/0.033

        # calculate e_sum
        e_sum = scatter(vals, idx, reduce='sum') 

        # learn energy fractions
        vals = vals / torch.repeat_interleave(
            e_sum, nnz, dim=0
        )

        points = self.construct_points(coords, vals)
        conditional = self.construct_conditional(nnz, e_sum, e_in)

        e_in = self.transform_e_in(e_in) 

        loss_cond = -self.conditional_flow(e_in).log_prob(conditional).mean()
        loss_cond /= self.config.conditional_dim

        mu, log_var = self.encoder(points, idx)

        # Reparameterization trick
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        latent = eps * std + mu

        # calculate entropy
        d = log_var.size(1)
        # 5.675754132818691 == 2 (1 + log(2 * pi))
        loss_entropy = - (d / 5.675754132818691 + 0.5 * log_var.sum(dim=1))
        loss_entropy = loss_entropy.mean() / self.config.latent_dim

        c = torch.cat([e_in, conditional], dim=1)

        loss_prior = - self.latent_flow(c).log_prob(latent).mean() 
        loss_prior /= self.config.latent_dim 

        c = torch.cat([c, latent], dim=1)
        c = torch.repeat_interleave(c, nnz, dim=0)
        
        if self.config.deepset_flow:
            log_prob = self.point_flow(idx, nnz, c).log_prob(points)
        else:
            log_prob = self.point_flow(c).log_prob(points)
        loss_points = -scatter(log_prob, idx, reduce='mean').mean()
        loss_points /= self.config.point_dim

        loss = loss_cond + loss_entropy + loss_prior + loss_points 

        # Logging
        batch_size = len(nnz)

        self._log(mode, "loss", loss, batch_size)
        self._log(mode, "loss_points", loss_points, batch_size)
        self._log(mode, "loss_cond", loss_cond, batch_size)
        self._log(mode, "loss_prior", loss_prior, batch_size)
        self._log(mode, "loss_entropy", loss_entropy, batch_size)

        return loss

    def _log(self, mode, name, value, batch_size):
        on_step = True if mode == "train" else False
        prog_bar = True if mode == "train" else False
        self.log(f"{mode}_{name}", value.detach(), on_step=on_step, on_epoch=True,
                 prog_bar=prog_bar, logger=True, batch_size=batch_size)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self._step(batch, batch_idx, mode="train")
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self._step(batch, batch_idx, mode="val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.optimizer.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.config.trainer.max_epochs)

        return [optimizer], [scheduler]
    

    def _setup_coords(self):
        binning = self.config.dataset.binning

        coords = []
        z_bins = binning["z_bins"]
        for z in range(z_bins):
            alpha_bins = binning["alpha_bins"]
            if not isinstance(alpha_bins, int):
                alpha_bins = alpha_bins[z] 
            for alpha in range(alpha_bins):
                r_bins = binning["r_bins"]
                if not isinstance(r_bins, int):
                    r_bins = r_bins[z]
                if not isinstance(r_bins, int):
                    r_bins = r_bins[alpha]

                for r in range(r_bins):
                    coords.append([z, alpha, r])

        self.register_buffer("coords", torch.tensor(coords)) 

    def _coord_extent(self, coords: Tensor)  -> Tensor:
        return self.coords[coords.squeeze(1), ...]
        
    def construct_points(self, coords: Tensor, vals: Tensor) -> Tensor:

        coords = self._coord_extent(coords)
        # we do generate alpha, only z and r coordinates
        if self.config.point_dim == 3:
            coords = coords[:, [0, 2]]
 
        coords = self.dequantize_coords(coords)

        if self.config.point_dim == 4:
            alpha = coords[:, 1] * 2 * torch.pi
            r = coords[:, 2]
            coords = torch.stack([
                coords[:, 0],
                torch.cos(alpha) * r / 2 + 0.5,
                torch.sin(alpha) * r / 2 + 0.5,
            ], dim=-1)

        if self.config.cdf_dequantization:
            coords = normal_quantile(coords)
        else:
            coords = logistic_quantile(coords)


        # Learn the energy fractions
        vals = self.min_max_scale_vals(vals)
        vals = logistic_quantile(vals)
        vals = self.normalize_vals(vals)

        points = torch.cat([coords, vals.unsqueeze(-1)], dim=-1)

        return points

    def deconstruct_points(self, points: Tensor) -> tuple[Tensor, Tensor]:

        vals = points[:, -1]
        vals = self.normalize_vals.inverse(vals)
        vals = logistic_cdf(vals)
        vals = self.min_max_scale_vals.inverse(vals)

        coords = points[:, :-1]
        if self.config.cdf_dequantization:
            coords = normal_cdf(coords)
        else:
            coords = logistic_cdf(coords)

        if self.config.point_dim == 4:
            x = coords[:, 1] * 2 - 1
            y = coords[:, 2] * 2 - 1

            coords = torch.stack([
                coords[:, 0],
                torch.atan2(y, x) / (2 * torch.pi),
                torch.sqrt(x**2 + y**2),
            ], dim=-1)

        coords = self.dequantize_coords.inverse(coords)

        return coords, vals

    def construct_conditional(self, nnz: Tensor, e_sum: Tensor, e_in: Tensor) -> Tensor:
        nnz = nnz.float()
        nnz = nnz + torch.rand_like(nnz)
        nnz = nnz.unsqueeze(-1).log() -  e_in.sqrt().log()
        nnz = self.normalize_nnz(nnz) #.unsqueeze(-1)

        e_sum = e_sum.unsqueeze(-1).log() -  e_in.log()
        e_sum = self.normalize_e_sum(e_sum)

        return torch.cat([nnz, e_sum], dim=-1)

    def deconstruct_conditional(self, conditional, e_in: Tensor) -> Tensor:
        nnz = conditional[:, 0]
        e_sum = conditional[:, 1]

        nnz = self.normalize_nnz.inverse(nnz)
        nnz = nnz + e_in.sqrt().log().view(-1)
        nnz = nnz.exp()
        nnz = nnz.floor().int()

        e_sum = self.normalize_e_sum.inverse(e_sum)
        e_sum = e_sum + e_in.log().view(-1)
        e_sum = e_sum.exp()

        return nnz, e_sum

    def transform_e_in(self, e_in):
        e_in = torch.log(e_in)
        e_in = self.normalize_e_in(e_in)
        return e_in