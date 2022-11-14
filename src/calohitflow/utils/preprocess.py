import torch
from torch import Tensor


def voxel_to_point_cloud(voxels: Tensor, energies: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    pc_range = []
    points = []
    reduced_energies = []

    end = 0

    for e, v in zip(energies, voxels):
        x = v.to_sparse_coo().coalesce()

        # there are a small quantity of complete empty showers
        # we skip them
        if x._nnz() == 0:
            continue 

        pc = torch.empty((x._nnz(), 4))

        pc[:, 0:3] = x.indices().T
        pc[:, 3] = x.values()

        start = end
        end = end + x._nnz()

        reduced_energies.append(e)

        pc_range.append([start, end])
        points.append(pc)

    points = torch.cat(points)
    pc_range = torch.tensor(pc_range, dtype=torch.int64)

    reduced_energies = torch.cat(reduced_energies)

    return points, pc_range, reduced_energies