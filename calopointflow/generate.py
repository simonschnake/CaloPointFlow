from typing import Optional
from omegaconf import DictConfig
import torch
import math
import numba
import numpy as np
import h5py
from .model import CaloPointFlow
from .utils import load_state

@numba.jit(nopython=True)
def _generate_shower(shower, x, uni_c, counts, val, alpha_bins, r_bins):
    for i in range(len(uni_c)):
        u = uni_c[i]
        count = counts[i]
        z = u // r_bins 
        r = u % r_bins
        e = val[x == u]
        idx = np.split(
            np.arange(count), 
            np.arange(start=0, stop=count, step=alpha_bins))
        for js in idx:
            c = len(js)
            if c == 0:
                continue
            alpha = np.random.choice(np.arange(alpha_bins), c, replace=False)
            shower[z, alpha, r] += e[js]

def generate(cfg: DictConfig, ckpt_path: str, save_path: Optional[str] = None) -> None:
    cpf = CaloPointFlow(cfg)

    load_state(cpf, ckpt_path)

    z_bins, alpha_bins, r_bins = cfg.dataset.binning.z_bins, cfg.dataset.binning.alpha_bins, cfg.dataset.binning.r_bins

    if torch.cuda.is_available():
        cpf = cpf.cuda()

    cpf.eval()

    incident_energy = torch.exp(
        torch.rand((cfg.generate.dataset_size, 1), dtype=torch.float32)
        * (math.log(1_000_000) - math.log(1_000))
        + math.log(1_000)
    )

    data_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(incident_energy),
        batch_size=cfg.generate.batch_size,
        num_workers=cfg.data.num_workers,
    )

    showers = []

    for batch in data_loader:
        e_in = batch[0].to(cpf.device)
        e_in_hat = cpf.transform_e_in(e_in)

        conditional = cpf.conditional_flow(e_in_hat).sample((1,)).squeeze(0)
        
        nnz, e_sum = cpf.deconstruct_conditional(conditional, e_in)

        c = torch.cat([e_in_hat, conditional], dim=-1)

        latent = cpf.latent_flow(c).sample((1,)).squeeze(0)

        c = torch.cat([c, latent], dim=-1)

        c = torch.repeat_interleave(c, nnz, dim=0)
        idx = torch.repeat_interleave(torch.arange(len(nnz), device=nnz.device), nnz)

        if cfg.deepset_flow:
            points = cpf.point_flow(idx, nnz, c).sample((1,)).squeeze(0)
        else:
            points = cpf.point_flow(c).sample((1,)).squeeze(0)

        coords, vals = cpf.deconstruct_points(points)

        coords = coords.cpu().numpy()
        vals = vals.cpu().numpy()
        nnz = nnz.cpu().numpy()

        start_index = np.zeros(nnz.shape, dtype=nnz.dtype)
        start_index[1:] = np.cumsum(nnz)[:-1]

        c = (coords[:, 0] * r_bins + coords[:, 1])

        shower = np.zeros((len(nnz), z_bins, alpha_bins, r_bins))

        for i, (s, n) in enumerate(zip(start_index, nnz)):
            x = c[s:s+n]
            coord = coords[s:s+n]
            v = vals[s:s+n]
            if cfg.point_dim == 3:
                uni_c, counts = np.unique(x, return_counts=True)
                _generate_shower(shower[i], x, uni_c, counts, v, alpha_bins, r_bins)
            else:
                shower[i, coord[:, 0], coord[:, 1], coord[:, 2]] = v

        shower /= shower.sum(axis=(1,2,3), keepdims=True)
        shower *= e_sum.cpu().view(-1, 1, 1, 1).numpy()

        # hardcoded add the minimum energy in the dataset
        if cfg.shift_min_energy:
            shower[shower > 0] += 0.5e-3/0.033

        showers.append(shower)

    showers = np.concatenate(showers, axis=0)

    if cfg.generate.save_dataset:
        with h5py.File(save_path, "w") as f:
            f.create_dataset(
                "showers", 
                data=showers.astype("<f8").reshape(cfg.generate.dataset_size, -1))
            f.create_dataset(
                "incident_energies", 
                data=incident_energy.numpy().astype("<f8").reshape(cfg.generate.dataset_size, -1))
