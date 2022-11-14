import torch
from torch import Tensor

from calohitflow.modules.flow import Flow


class PointFlow(Flow):
    """
    The Point Flow.
    """

    def __init__(
        self,
        features: int = 2,
        n_transforms: int = 8,
        tail_bound: float = 1.0,
        num_bins: int = 4,
        context_features: int = 32,
        context_hidden_features: int = 64,
        hidden_features: int = 30,
        num_blocks: int = 2,
    ) -> None:
        super().__init__(
            features,
            n_transforms,
            tail_bound,
            num_bins,
            context_features,
            context_hidden_features,
            hidden_features,
            num_blocks,
        )

    def log_prob(self, points: Tensor, z: Tensor, idx: Tensor)-> Tensor:
        """Calculate the log_prob of the indivual showers.
        Params:
            points  - The flat points of all point clouds to transform
            z       - The latent vectors
            idx     - The position of all point clouds in the flat points vector.
                      The ith point cloud is in points[idx[i-1]:idx[i]]  
        """

        context = torch.empty((points.size(0), z.size(1)), device=z.device)

        idx_start = 0
        for i, idx_end in enumerate(idx):
            context[idx_start:idx_end] = z[i]
            idx_start = idx_end
        
        log_probs = super().log_prob(points, context)

        res = torch.empty((z.size(0)), device=z.device)

        idx_start = 0
        for i, idx_end in enumerate(idx):
            res[i] = log_probs[idx_start:idx_end].mean()
            idx_start = idx_end

        return res



