import torch
from torch import Tensor, Size, nn

from calopointflow.freezables import Freezable

class MinMaxScale(Freezable):
    """Min-Max Scaling
    This module scales the input tensor to the range [0, 1]. 
    
    Arguments:
        size (Size | tuple[int, ...] | int): The size of the input tensor.
    """
    def __init__(self, size: Size | tuple[int, ...] | int) ->  None:
        super().__init__()
        self.size = size
        self.register_buffer('min', torch.ones(self.size) * torch.inf)
        self.register_buffer('max', torch.ones(self.size) * -torch.inf)

    def forward(self, x: Tensor) -> Tensor:
        """
        Min-Max scale the input tensor to the range [0, 1].

        Parameters:
            x (Tensor): The input tensor to scale.

        Returns:
            Tensor: The scaled tensor.
        """
        # Check if the MinMaxScale is frozen
        if not self.frozen:
            # Update the min and max with the input tensor
            self._update(x)
        
        # Calculate the scaled tensor
        return (x - self.min) / (self.max - self.min)

    def inverse(self, y: Tensor) -> Tensor:
        """
        Scale the input tneosr back to the range [min, max].

        Parameters:
            y (Tensor): The input tensor to scale back.

        Returns:
            Tensor: The scaled tensor.
        """
        return y * (self.max - self.min) + self.min

    def _update(self, x: Tensor) -> None:
        """
        Update the maximum and minimum values with the input tensor.

        Parameters:
            x (Tensor): The input tensor.

        Returns:
            None
        """
        x_max = torch.max(x, dim=0).values
        x_min = torch.min(x, dim=0).values

        self.max = torch.max(self.max, x_max)
        self.min = torch.min(self.min, x_min) 