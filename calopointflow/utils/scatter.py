import torch

def scatter(x, idx, dim=0, reduce="sum"):
    shape = list(x.shape)
    shape[dim] = idx.max().item() + 1
    res = torch.zeros(shape, dtype=x.dtype, device=x.device)
    idx_shape = [1] * len(shape)
    idx_shape[dim] = -1
    res.scatter_reduce_(dim=dim, index=idx.view(*idx_shape), src=x, reduce=reduce)
    return res