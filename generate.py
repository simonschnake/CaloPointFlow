import math
import multiprocessing
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import numba
from omegaconf import OmegaConf
from calopointflow.model import CaloPointFlow
from calopointflow.utils import load_state, load_config
from tqdm import tqdm
import h5py

generation_config = OmegaConf.load("generate_conf.yaml")

DATASET = generation_config.dataset
BATCH_SIZE = generation_config.batch_size
DATASET_SIZE = generation_config.dataset_size
SAVE_DATASET = generation_config.save_dataset
POINT_DIM = generation_config.point_dim
DEEPSET_FLOW = generation_config.deepset_flow
CDF_DEQUANTIZATION = generation_config.cdf_dequantization

NUM_WORKERS = multiprocessing.cpu_count()
CUDA = torch.cuda.is_available()

base_cfg = load_config("base")

if DATASET == 2:
    cfg_str = "d2" 
elif DATASET == 3:
    cfg_str = "d3"
else:
    raise ValueError("Invalid dataset number")
if POINT_DIM == 3:
    cfg_str += "_II"
elif CDF_DEQUANTIZATION:
    cfg_str += "_dsf_cdeq"
elif DEEPSET_FLOW:
    cfg_str += "_dsf"
else:
    cfg_str += "_I"

dataset_cfg = load_config(cfg_str)

cfg = OmegaConf.merge(base_cfg, dataset_cfg)

Z_BINS = cfg.dataset.binning.z_bins
ALPHA_BINS = cfg.dataset.binning.alpha_bins
R_BINS = cfg.dataset.binning.r_bins

cpf = CaloPointFlow(cfg)
load_state(cpf, cfg_str)
if CUDA:
    cpf = cpf.cuda()


incident_energy = torch.exp(
    torch.rand((DATASET_SIZE, 1), dtype=torch.float32)
    * (math.log(1_000_000) - math.log(1_000))
    + math.log(1_000)
)

data_loader = DataLoader(
    TensorDataset(incident_energy),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)

@numba.jit(nopython=True)
def _generate_shower(shower, x, uni_c, counts, val):
    for i in range(len(uni_c)):
        u = uni_c[i]
        count = counts[i]
        z = u // R_BINS 
        r = u % R_BINS
        e = val[x == u]
        idx = np.split(
            np.arange(count), 
            np.arange(start=0, stop=count, step=ALPHA_BINS))
        for js in idx:
            c = len(js)
            if c == 0:
                continue
            alpha = np.random.choice(np.arange(ALPHA_BINS), c, replace=False)
            shower[z, alpha, r] += e[js]

showers = []

for batch in tqdm(data_loader):
    e_in = batch[0].to(cpf.device)
    e_in_hat = cpf.transform_e_in(e_in)

    conditional = cpf.conditional_flow(e_in_hat).sample((1,)).squeeze(0)
    
    nnz, e_sum = cpf.deconstruct_conditional(conditional, e_in)

    c = torch.cat([e_in_hat, conditional], dim=-1)

    latent = cpf.latent_flow(c).sample((1,)).squeeze(0)

    c = torch.cat([c, latent], dim=-1)

    c = torch.repeat_interleave(c, nnz, dim=0)
    idx = torch.repeat_interleave(torch.arange(len(nnz), device=nnz.device), nnz)

    if DEEPSET_FLOW:
        points = cpf.point_flow(idx, nnz, c).sample((1,)).squeeze(0)
    else:
        points = cpf.point_flow(c).sample((1,)).squeeze(0)

    coords, vals = cpf.deconstruct_points(points)

    coords = coords.cpu().numpy()
    vals = vals.cpu().numpy()
    nnz = nnz.cpu().numpy()

    start_index = np.zeros(nnz.shape, dtype=nnz.dtype)
    start_index[1:] = np.cumsum(nnz)[:-1]

    c = (coords[:, 0] * Z_BINS + coords[:, 1])

    shower = np.zeros((len(nnz), Z_BINS, ALPHA_BINS, R_BINS))

    ns = []
    for i, (s, n) in enumerate(zip(start_index, nnz)):
        x = c[s:s+n]
        coord = coords[s:s+n]
        v = vals[s:s+n]
        if POINT_DIM == 3:
            uni_c, counts = np.unique(x, return_counts=True)
            _generate_shower(shower[i], x, uni_c, counts, v)
        else:
            shower[i, coord[:, 0], coord[:, 1], coord[:, 2]] = v

    shower /= shower.sum(axis=(1,2,3), keepdims=True)
    shower *= e_sum.cpu().view(-1, 1, 1, 1).numpy()

    showers.append(shower)

showers = np.concatenate(showers, axis=0)

if SAVE_DATASET:
    with h5py.File(f"/dev/shm/dataset_{cfg_str}.hdf5", "w") as f:
        f.create_dataset(
            "showers", 
            data=showers.astype("<f8").reshape(DATASET_SIZE, -1))
        f.create_dataset(
            "incident_energies", 
            data=incident_energy.numpy().astype("<f8").reshape(DATASET_SIZE, -1))
