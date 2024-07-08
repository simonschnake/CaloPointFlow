from torch import nn
from torch import Tensor

from calopointflow.modules.deep_sets import DeepSetModule


class Encoder(DeepSetModule):
    """The encoder part of the CaloPointFlow.

    Parameters:
        point_dim (int): The dimension of the points.
        latent_dim (int): The dimension of the latent representation.
        point_layers (list[int]): A list of the dimensions of the hidden layers
                                  of the point transformation layers.
        latent_layers (list[int]): A list of the dimensions of the hidden layers
                                   of the latent transformation layers.  
    """
    def __init__(
            self, 
            point_dim: int, 
            latent_dim: int, 
            point_layers: list[int], 
            latent_layers: list[int],
            reduce: str = "mean") -> None:

        prev_features = point_dim

        point_transformation_layers = []

        for features in point_layers: 
            point_transformation_layers.append(nn.Linear(prev_features, features))
            point_transformation_layers.append(nn.GELU())
            prev_features = features

        point_transformation_layers.append(nn.Linear(prev_features, prev_features))

        self.point_encoding = prev_features

        latent_transformation_layers = []

        for features in latent_layers:
            latent_transformation_layers.append(nn.Linear(prev_features, features))
            latent_transformation_layers.append(nn.GELU())
            prev_features = features

        super().__init__(
            point_net = nn.Sequential(*point_transformation_layers),
            reduced_net = nn.Sequential(*latent_transformation_layers), 
            reduce=reduce
        )

        self.mu_mapping = nn.Linear(prev_features, latent_dim)
        self.logvar_mapping = nn.Linear(prev_features, latent_dim)

    def forward(self, points: Tensor, idx: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute the latent representation of the shower.

        Parameters:
            points (Tensor): The points of the shower.
            idx (Tensor): The cloud indices of the points.

        Returns:
            tuple[Tensor, Tensor]:
                mu     - the mean of the latent distribution
                logvar - the log variance of the latent distribution
        """
        latent = super().forward(points, idx)

        mu = self.mu_mapping(latent)
        logvar = self.logvar_mapping(latent)

        return mu, logvar