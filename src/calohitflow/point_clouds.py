import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.utils import data


class PointCloudDataset(data.Dataset):
    def __init__(self, path: str) -> None:
        super().__init__()

        data = np.load(path + '.npz')
        self.points = torch.from_numpy(data["points"])
        self.pc_range = torch.from_numpy(data["pc_range"])
        self.energies = torch.from_numpy(data["energies"])

    def __getitem__(self, index):
        start, end = self.pc_range[index]
        points = self.points[start:end]

        e_sum = points[:, -1].sum()
        n_hits = torch.tensor(len(points), dtype=torch.int64)

        e_in = self.energies[index]

        return points, e_in, e_sum, n_hits

    def __len__(self):
        return len(self.energies)


def collate_point_cloud(batch_list):
    points = []
    e_in = []
    e_sum = []
    n_hits = []
    for b in batch_list:
        p, ei, es, n = b
        points.append(p)
        e_in.append(ei)
        e_sum.append(es)
        n_hits.append(n)

    points = torch.cat(points)
    e_in = torch.stack(e_in)
    e_sum = torch.stack(e_sum)
    n_hits = torch.stack(n_hits)

    return points, e_in, e_sum, n_hits
