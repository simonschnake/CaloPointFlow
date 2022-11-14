""" The encoder part of the network"""

from typing import List, Tuple

import torch
from torch import Tensor, nn


class Encoder(nn.Module):
    """
    The encoder part of the CaloHitFlow.

    It encodes the complete shower. It is permutation invariant.
    """
    def __init__(
        self,
        in_features: int,
        latent_dim: int,
        point_layers: List[int],
        latent_layers: List[int],
    ) -> None:
        """
        Initialize encoder.
        Params:
            in_features   - the number of features the individual points have 
            latent_dim    - the dimensionality of the latent vector $z$
            point_layers  - a list of the features of the indivdual layers in the point encoding part 
                            of the encoder 
            latent_layers - a list of the features of the indivdual layers in the latent encoding part 
                            of the encoder 
        """
        super().__init__()

        prev_features = in_features

        layers = []
        for features in point_layers:
            layers.append(nn.Linear(prev_features, features)),
            layers.append(nn.LeakyReLU(inplace=True)),
            prev_features = features

        layers.append(nn.Linear(prev_features, prev_features)),

        # TODO: think about initialization

        self.point_encoding = nn.Sequential(*layers)
        self.point_features = prev_features

        layers = []
        for features in latent_layers:
            layers.append(nn.Linear(prev_features, features)),
            layers.append(nn.LeakyReLU(inplace=True)),
            prev_features = features

        self.latent_encoding = nn.Sequential(*layers)

        self.mu_mapping = nn.Linear(prev_features, latent_dim)
        self.logvar_mapping = nn.Linear(prev_features, latent_dim)

    def forward(self, points: Tensor, idx: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode the point clouds into the latent space $z$.
        The points are flat transformed by the point encoding and divided into 
        individual point clouds by the positioning in pc_ends.

        Params:
            points  - The flat points of all point clouds to transform
            idx - The position of all point clouds in the flat points vector.
                      The ith point cloud is in points[idx[i-1]:idx[i]]
        """

        points = self.point_encoding(points)

        latent = torch.empty((len(idx), self.point_features),
                             dtype=points.dtype, device=points.device)

        idx_start = 0
        for i, idx_end in enumerate(idx):
            latent[i] = points[idx_start:idx_end].mean(axis=0)
            idx_start = idx_end

        # encode latent
        latent = self.latent_encoding(latent)

        mu = self.mu_mapping(latent)
        log_var = self.logvar_mapping(latent)
        return mu, log_var
